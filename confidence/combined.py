import numpy as np
import torch
from confidence.base import ConfidenceEstimator
from confidence.temperature import TemperatureScaler
from confidence.mc_dropout import MCDropoutConfidence

class CombinedConfidenceEstimator(ConfidenceEstimator):
    """
    Combined confidence estimator that integrates multiple methods
    for more reliable confidence estimation.

    Supports both character-level and word-level confidence calibration.

    Args:
        model: MCDropoutCRNN model
        converter: CTC label converter
        device: Device to use
        level: 'word' or 'char' - level at which to apply calibration
    """
    def __init__(self, model, converter, device, level='word'):
        super().__init__(model, device)
        self.converter = converter
        self.level = level

        # Initialize component estimators
        self.temperature_scaler = TemperatureScaler(model, device, converter, level=level)
        self.mc_dropout = MCDropoutConfidence(model, converter, device, num_samples=30, level=level)

        # Default weights for combining confidences
        self.temp_weight = 0.4
        self.mc_weight = 0.6

        # Calibration status
        self.calibrated = False

    def calibrate(self, val_loader):
        """
        Calibrate all confidence estimators.

        Args:
            val_loader: Validation data loader
        """
        print(f"Calibrating confidence estimators (level: {self.level})...")

        # Calibrate temperature scaling
        self.temperature_scaler.calibrate(val_loader)

        # MC dropout doesn't need explicit calibration
        self.mc_dropout.calibrate(val_loader)

        self.calibrated = True
        print("Calibration complete!")

    def set_weights(self, temp_weight=0.4, mc_weight=0.6):
        """
        Set weights for combining different confidence estimators.

        Args:
            temp_weight: Weight for temperature scaling confidence
            mc_weight: Weight for MC dropout confidence
        """
        total = temp_weight + mc_weight
        self.temp_weight = temp_weight / total
        self.mc_weight = mc_weight / total
        print(f"Weights set: Temperature={self.temp_weight:.2f}, MC Dropout={self.mc_weight:.2f}")

    def get_confidence(self, images):
        """
        Get combined confidence estimates.

        Args:
            images: Input image batch

        Returns:
            Dictionary with predictions and multiple confidence metrics
        """
        if not self.calibrated:
            print("Warning: Using uncalibrated confidence estimator")

        # Get standard model output first
        self.model.eval()
        with torch.no_grad():
            logits = self.model(images)

        # Get temperature scaling confidence
        temp_results = self.temperature_scaler.get_confidence(logits)

        # Get MC dropout confidence
        mc_results = self.mc_dropout.get_confidence(images)

        # Combine confidences with weighted average
        combined_confidences = []
        for i in range(len(temp_results['confidences'])):
            temp_conf = temp_results['confidences'][i]
            mc_conf = mc_results['confidences'][i]
            combined = self.temp_weight * temp_conf + self.mc_weight * mc_conf
            combined_confidences.append(combined)

        # We use MC dropout predictions as our final predictions (more robust)
        return {
            'predictions': mc_results['predictions'],
            'temperature_confidences': temp_results['confidences'],
            'mc_confidences': mc_results['confidences'],
            'combined_confidences': combined_confidences,
            'epistemic_uncertainties': mc_results['epistemic_uncertainties'],
            'aleatoric_uncertainties': mc_results['aleatoric_uncertainties'],
            'prediction_distribution': mc_results.get('prediction_distribution', {})
        }
