import torch
import torch.nn.functional as F
from .base import ConfidenceEstimator

class UncalibratedConfidence(ConfidenceEstimator):
    """
    Raw uncalibrated confidence from the model's softmax outputs.

    This method simply uses the raw probabilities from the model's
    softmax layer as confidence scores without any calibration.

    Args:
        model: Recognition model
        device: Device to use
        converter: Label converter for decoding
        normalize_confidence: Whether to normalize confidence scores
    """
    def __init__(self, model, device, converter, normalize_confidence=True):
        super().__init__(model, device, converter)
        self.name = "Uncalibrated"
        # Uncalibrated confidence doesn't need calibration
        self.calibrated = True
        self.normalize_confidence = normalize_confidence

    def calibrate(self, val_loader):
        """
        No calibration needed for uncalibrated confidence.

        Args:
            val_loader: Validation data loader (not used)
        """
        # Already "calibrated" (there's nothing to calibrate)
        pass

    def get_confidence(self, images):
        """
        Get uncalibrated confidence scores from the model.

        Args:
            images: Input image batch

        Returns:
            Dictionary with predictions and raw confidence scores
        """
        # Get model outputs
        logits, decoded = self._get_model_predictions(images)

        # Convert to probabilities
        probs = F.softmax(logits, dim=2)

        # Get the highest probability at each timestep
        max_probs, _ = probs.max(2)

        # Compute confidence for each sequence
        batch_size = probs.size(1)
        confidences = []

        for b in range(batch_size):
            # Get sequence probabilities for this batch item
            seq_probs = max_probs[:, b]

            # Compute product of probabilities (using log space for numerical stability)
            confidence = self.calculate_confidence(seq_probs)

            confidences.append(confidence)

        return {
            'predictions': decoded,
            'confidences': confidences,
            'raw_probabilities': probs.cpu().numpy(),
            'method': self.name
        }
