import torch
import torch.nn.functional as F
from confidence.base import ConfidenceEstimator

class UncalibratedConfidence(ConfidenceEstimator):
    """
    Raw uncalibrated confidence from the model's softmax outputs.

    Args:
        model: Recognition model
        converter: Label converter for decoding
        device: Device to use
    """
    def __init__(self, model, converter, device):
        super().__init__(model, device)
        self.converter = converter
        # Uncalibrated confidence doesn't need calibration
        self.calibrated = True

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
        self.model.eval()
        with torch.no_grad():
            # Get model outputs
            logits = self.model(images)

            # Convert to probabilities
            probs = F.softmax(logits, dim=2)

            # Get the highest probability at each timestep
            max_probs, max_indices = probs.max(2)

            # Decode predicted sequences
            decoded = self.converter.decode(logits)

            # Compute confidence for each sequence
            batch_size = probs.size(1)
            confidences = []

            for b in range(batch_size):
                # Get sequence probabilities for this batch item
                seq_probs = max_probs[:, b]

                # Compute product of probabilities (using log space for numerical stability)
                log_probs = torch.log(seq_probs + 1e-10)
                sum_log_prob = torch.sum(log_probs)
                confidence = torch.exp(sum_log_prob).item()

                confidences.append(confidence)

            return {
                'predictions': decoded,
                'confidences': confidences
            }
