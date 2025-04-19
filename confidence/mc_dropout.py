import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from confidence.base import ConfidenceEstimator

class MCDropoutConfidence(ConfidenceEstimator):
    """
    Monte Carlo Dropout for confidence/uncertainty estimation.

    This implementation supports both character-level and word-level evaluation
    and uses simple product for word-level confidence calculation.

    Args:
        model: MCDropoutCRNN model
        converter: Label converter for decoding predictions
        device: Device to run inference on (cuda/cpu)
        num_samples: Number of Monte Carlo samples to generate
        level: 'word' or 'char' - level at which to calculate confidence
    """
    def __init__(self, model, converter, device, num_samples=30, level='word'):
        super().__init__(model, device)
        self.converter = converter
        self.num_samples = max(30, num_samples)  # Ensure at least 30 samples
        self.level = level

    def calibrate(self, val_loader):
        """
        MC Dropout doesn't need explicit calibration.

        Args:
            val_loader: Validation data loader (not used)
        """
        # MC Dropout doesn't require explicit calibration
        self.calibrated = True
        return

    def get_confidence(self, images):
        """
        Estimate confidence and uncertainty using Monte Carlo Dropout.

        Args:
            images: Input image batch [batch_size, channels, height, width]

        Returns:
            Dictionary with prediction and confidence information
        """
        self.model.eval()
        images = images.to(self.device)
        batch_size = images.size(0)

        # Store all sampled outputs
        all_outputs = []
        all_decoded = []

        # Generate multiple predictions with dropout enabled
        with torch.no_grad():
            for _ in tqdm(range(self.num_samples), desc="MC Sampling", leave=False):
                outputs = self.model(images, apply_dropout=True)
                all_outputs.append(outputs)

                # Decode predictions for this sample
                decoded = self.converter.decode(outputs)
                all_decoded.append(decoded)

        # Stack all outputs: [num_samples, seq_len, batch_size, vocab_size]
        stacked_outputs = torch.stack(all_outputs)

        # Mean across samples: [seq_len, batch_size, vocab_size]
        mean_outputs = torch.mean(stacked_outputs, dim=0)

        # Decode mean predictions
        mean_decoded = self.converter.decode(mean_outputs)

        # Calculate confidence metrics
        confidences = []
        aleatoric_uncertainties = []
        epistemic_uncertainties = []
        predictions_count = {}

        # Process each sample in the batch
        for b in range(batch_size):
            if self.level == 'word':
                # Word-level confidence
                # Count prediction occurrences for this sample
                sample_predictions = [decoded[b] for decoded in all_decoded]
                prediction_counts = {}

                for pred in sample_predictions:
                    prediction_counts[pred] = prediction_counts.get(pred, 0) + 1

                # Get the most frequent prediction and its frequency
                predictions_count[b] = prediction_counts
                most_common = max(prediction_counts.items(), key=lambda x: x[1])
                most_common_pred, most_common_count = most_common

                # Calculate prediction probability (most common count / total samples)
                confidence = most_common_count / self.num_samples
                confidences.append(confidence)

                # Calculate uncertainties from the probabilities at each timestep
                sample_epistemic_uncertainty = 0
                sample_aleatoric_uncertainty = 0

                # Get probabilities at each sequence position
                for t in range(stacked_outputs.shape[1]):
                    try:
                        # Get probabilities for all classes at this position
                        position_probs = F.softmax(stacked_outputs[:, t, b, :], dim=1)

                        # Variance across samples (epistemic uncertainty)
                        var_probs = torch.var(position_probs, dim=0)
                        epistemic = torch.mean(var_probs).item()

                        # Expected entropy (aleatoric uncertainty)
                        entropy = -torch.mean(torch.sum(position_probs * torch.log(position_probs + 1e-10), dim=1))
                        aleatoric = entropy.item()

                        sample_epistemic_uncertainty += epistemic
                        sample_aleatoric_uncertainty += aleatoric
                    except RuntimeWarning:
                        # Handle cases with too few samples
                        pass

                # Average uncertainties over sequence length
                seq_len = stacked_outputs.shape[1]
                sample_epistemic_uncertainty /= seq_len
                sample_aleatoric_uncertainty /= seq_len

                epistemic_uncertainties.append(sample_epistemic_uncertainty)
                aleatoric_uncertainties.append(sample_aleatoric_uncertainty)

            else:  # Character-level
                # Initialize character-level metrics
                char_confidences = []
                char_epistemic = []
                char_aleatoric = []

                # For each position in the sequence
                for t in range(stacked_outputs.shape[1]):
                    # Get probabilities for all samples at this position
                    position_probs = F.softmax(stacked_outputs[:, t, b, :], dim=1)

                    # Get the most probable class at each timestep for each sample
                    _, sample_classes = position_probs.max(dim=1)

                    # Count occurrences of each class
                    class_counts = {}
                    for cls in sample_classes:
                        cls_item = cls.item()
                        class_counts[cls_item] = class_counts.get(cls_item, 0) + 1

                    # Get most common class
                    most_common_class = max(class_counts.items(), key=lambda x: x[1])
                    most_common_count = most_common_class[1]

                    # Character-level confidence
                    char_conf = most_common_count / self.num_samples
                    char_confidences.append(char_conf)

                    # Calculate uncertainties
                    try:
                        # Variance across samples (epistemic uncertainty)
                        var_probs = torch.var(position_probs, dim=0)
                        epistemic = torch.mean(var_probs).item()

                        # Expected entropy (aleatoric uncertainty)
                        entropy = -torch.mean(torch.sum(position_probs * torch.log(position_probs + 1e-10), dim=1))
                        aleatoric = entropy.item()

                        char_epistemic.append(epistemic)
                        char_aleatoric.append(aleatoric)
                    except RuntimeWarning:
                        # Handle cases with too few samples
                        char_epistemic.append(0.0)
                        char_aleatoric.append(0.0)

                # For word-level confidence, use product of character confidences
                log_confidences = [np.log(conf + 1e-10) for conf in char_confidences]
                word_confidence = np.exp(sum(log_confidences))

                confidences.append(word_confidence)
                epistemic_uncertainties.append(np.mean(char_epistemic))
                aleatoric_uncertainties.append(np.mean(char_aleatoric))

        return {
            'predictions': mean_decoded,
            'confidences': confidences,
            'epistemic_uncertainties': epistemic_uncertainties,
            'aleatoric_uncertainties': aleatoric_uncertainties,
            'prediction_distribution': predictions_count
        }
