import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from .base import ConfidenceEstimator
from .visualization import plot_temperature_ece_curve

class TemperatureScaling(ConfidenceEstimator):
    """
    Temperature scaling for calibrating model confidence.

    This implementation supports both length-normalized and unnormalized confidence,
    and uses ECE minimization as the optimization objective.

    Args:
        model: The model to calibrate
        device: Device to run calibration on (cuda/cpu)
        converter: Converter to convert logits to probabilities
        normalize_confidence: Whether to use length-normalized confidence
    """
    def __init__(self, model, device, converter, normalize_confidence=True):
        super().__init__(model, device, converter)
        self.name = "Temperature Scaling"
        self.temperature = nn.Parameter(torch.ones(1).to(device))
        self.calibrated = False
        self.normalize_confidence = normalize_confidence

    def calibrate(self, val_loader, min_temp=1.0, max_temp=1.5, num_temps=10):
        """
        Calibrate temperature using grid search to minimize ECE.

        Args:
            val_loader: Validation data loader
            min_temp: Minimum temperature to try
            max_temp: Maximum temperature to try
            num_temps: Number of temperature values to try
        """
        self.model.eval()
        print(f"Calibrating {self.name} using validation data...")

        # Temperature values to try
        temp_values = np.linspace(min_temp, max_temp, num_temps)
        best_ece = float('inf')
        best_temp = 1.0

        # Store results for visualization
        all_temps = []
        all_eces = []

        # Try different temperature values
        for temp in tqdm(temp_values, desc="Testing temperatures"):
            # Calculate confidences and correctness using this temperature
            all_confidences = []
            all_correctness = []

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Evaluating T={temp:.2f}", leave=False):
                    images = images.to(self.device)

                    # Forward pass
                    logits = self.model(images)

                    # Apply this temperature
                    temp_logits = logits / temp
                    probs = F.softmax(temp_logits, dim=2)
                    max_probs, _ = probs.max(2)

                    # Decode predictions
                    decoded = self.converter.decode(temp_logits)

                    # Word-level calibration
                    batch_size = probs.size(1)
                    for b in range(batch_size):
                        # Get sequence probabilities
                        seq_probs = max_probs[:, b]

                        # Calculate confidence based on normalization setting
                        confidence = self.calculate_confidence(seq_probs)

                        # Check if prediction is correct
                        is_correct = 1 if decoded[b] == labels[b] else 0

                        all_confidences.append(confidence)
                        all_correctness.append(is_correct)

            # Calculate ECE for this temperature
            ece = self._calculate_ece(all_confidences, all_correctness)

            # Store for visualization
            all_temps.append(temp)
            all_eces.append(ece)

            print(f"T={temp:.4f}, ECE={ece:.4f}")

            # Update best temperature
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
                print(f"New best temperature: {best_temp:.4f}, ECE: {best_ece:.4f}")

        # Create visualization directory
        vis_dir = os.path.join(os.getcwd(), 'output', 'confidence_evaluation_v2', 'Temperature_Scaling')
        os.makedirs(vis_dir, exist_ok=True)

        # Plot temperature vs ECE curve
        plot_temperature_ece_curve(
            all_temps, all_eces, best_temp, best_ece,
            save_path=os.path.join(vis_dir, 'temperature_ece_curve.png')
        )

        # Set the best temperature
        self.temperature.data = torch.tensor([best_temp], device=self.device)
        print(f"Final calibrated temperature: {best_temp:.4f}, ECE: {best_ece:.4f}")

        # Print type of confidence used
        conf_type = "length-normalized (geometric mean)" if self.normalize_confidence else "unnormalized (product)"
        print(f"Confidence type used: {conf_type}")

        self.calibrated = True
        return best_temp

    def get_confidence(self, images):
        """
        Get calibrated confidence scores for predictions.

        Args:
            images: Input image batch

        Returns:
            Dictionary with decoded texts and confidence scores
        """
        if not self.calibrated:
            print(f"Warning: Using {self.name} before calibration")

        # Get model outputs
        logits, _ = self._get_model_predictions(images)

        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=2)

        # Get the highest probability and index at each timestep
        max_probs, _ = probs.max(2)

        # Decode predicted sequences
        decoded = self.converter.decode(scaled_logits)

        # Compute confidence for each sequence
        batch_size = probs.size(1)
        confidences = []

        for b in range(batch_size):
            # Get sequence probabilities for this batch item
            seq_probs = max_probs[:, b]

            # Calculate confidence
            confidence = self.calculate_confidence(seq_probs)
            confidences.append(confidence)

        return {
            'predictions': decoded,
            'confidences': confidences,
            'temperature': self.temperature.item(),
            'method': self.name
        }

    def _calculate_ece(self, confidences, correctness, num_bins=10):
        """Calculate Expected Calibration Error."""
        if not confidences:
            return 1.0  # Return worst ECE if no data

        # Convert to numpy arrays
        confidences = np.array(confidences)
        correctness = np.array(correctness)

        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Initialize arrays for bin statistics
        bin_accuracies = np.zeros(num_bins)
        bin_confidences = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)

        # Calculate accuracy and confidence for each bin
        for i in range(num_bins):
            bin_mask = (bin_indices == i)
            if np.any(bin_mask):
                bin_counts[i] = np.sum(bin_mask)
                bin_accuracies[i] = np.mean(correctness[bin_mask])
                bin_confidences[i] = np.mean(confidences[bin_mask])

        # Calculate ECE
        ece = 0
        total_samples = len(confidences)
        for i in range(num_bins):
            if bin_counts[i] > 0:
                ece += (bin_counts[i] / total_samples) * abs(bin_accuracies[i] - bin_confidences[i])

        return ece
