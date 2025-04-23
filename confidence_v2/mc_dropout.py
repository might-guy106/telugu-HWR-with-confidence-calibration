import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import Counter
from .base import ConfidenceEstimator

class MCDropoutConfidence(ConfidenceEstimator):
    """
    Monte Carlo Dropout for confidence/uncertainty estimation.

    This method performs multiple forward passes with dropout enabled,
    resulting in an ensemble of predictions that can be used to estimate
    the model's uncertainty.

    Args:
        model: Model with dropout layers
        device: Device to run inference on (cuda/cpu)
        converter: Label converter for decoding predictions
        num_samples: Number of Monte Carlo samples to generate
    """
    def __init__(self, model, device, converter, num_samples=30, normalize_confidence=True):
        super().__init__(model, device, converter, normalize_confidence)
        self.name = "MC Dropout"
        self.num_samples = num_samples
        self.calibrated = True  # MC Dropout doesn't need explicit calibration

    def calibrate(self, val_loader):
        """
        MC Dropout doesn't need explicit calibration.

        Args:
            val_loader: Validation data loader (not used)
        """
        # Verify the model has dropout layers
        has_dropout = False
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
                has_dropout = True
                break

        if not has_dropout:
            print("Warning: Model doesn't have dropout layers, MC Dropout may not be effective")
        else:
            print(f"Model has dropout layers, MC Dropout is ready with {self.num_samples} samples")

        self.calibrated = True

    def get_confidence(self, images):
        """
        Estimate confidence and uncertainty using Monte Carlo Dropout.

        Args:
            images: Input image batch [batch_size, channels, height, width]

        Returns:
            Dictionary with prediction and confidence information
        """
        images = images.to(self.device)
        batch_size = images.size(0)

        # Store all sampled outputs
        all_outputs = []
        all_decoded = []
        all_probs = []

        # Generate multiple predictions with dropout enabled
        with torch.no_grad():
            for _ in tqdm(range(self.num_samples), desc="MC Sampling", leave=False):
                # Enable dropout during inference
                self.model.apply(self.enable_dropout)

                # Forward pass
                outputs = self.model(images)
                all_outputs.append(outputs)

                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=2)
                all_probs.append(probs)

                # Decode predictions for this sample
                decoded = self.converter.decode(outputs)
                all_decoded.append(decoded)

        # Stack all outputs: [num_samples, seq_len, batch_size, vocab_size]
        stacked_outputs = torch.stack(all_outputs)
        stacked_probs = torch.stack(all_probs)

        # Mean across samples: [seq_len, batch_size, vocab_size]
        mean_outputs = torch.mean(stacked_outputs, dim=0)
        mean_probs = torch.mean(stacked_probs, dim=0)

        # Decode mean predictions
        mean_decoded = self.converter.decode(mean_outputs)

        # Calculate confidence metrics
        confidences = []
        epistemic_uncertainties = []  # Model uncertainty
        aleatoric_uncertainties = []  # Data uncertainty
        prediction_distributions = []

        # Process each sample in the batch
        for b in range(batch_size):
            # Word-level confidence
            # Count prediction occurrences for this sample
            sample_predictions = [decoded[b] for decoded in all_decoded]
            prediction_counts = Counter(sample_predictions)

            # Get the most frequent prediction and its frequency
            most_common = prediction_counts.most_common(1)[0]
            most_common_pred, most_common_count = most_common

            # Calculate prediction probability (most common count / total samples)
            confidence = most_common_count / self.num_samples
            confidences.append(confidence)

            # Store distribution of predictions for this sample
            prediction_distributions.append(dict(prediction_counts))

            # Calculate uncertainties
            # Epistemic uncertainty: variance of probability predictions across samples
            sample_probs = stacked_probs[:, :, b, :]  # [num_samples, seq_len, vocab_size]
            var_probs = torch.var(sample_probs, dim=0)  # [seq_len, vocab_size]
            epistemic = torch.mean(var_probs).item()  # Average over sequence and vocab
            epistemic_uncertainties.append(epistemic)

            # Aleatoric uncertainty: expected entropy of predictions
            entropies = []
            for s in range(self.num_samples):
                # Calculate entropy of each prediction
                sample_prob = sample_probs[s]  # [seq_len, vocab_size]
                entropy = -torch.sum(sample_prob * torch.log(sample_prob + 1e-10), dim=1)  # [seq_len]
                mean_entropy = torch.mean(entropy).item()  # Average over sequence
                entropies.append(mean_entropy)

            aleatoric = np.mean(entropies)
            aleatoric_uncertainties.append(aleatoric)

        return {
            'predictions': mean_decoded,
            'confidences': confidences,
            'epistemic_uncertainties': epistemic_uncertainties,
            'aleatoric_uncertainties': aleatoric_uncertainties,
            'prediction_distributions': prediction_distributions,
            'method': self.name
        }
