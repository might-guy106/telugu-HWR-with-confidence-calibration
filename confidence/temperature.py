import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from confidence.base import ConfidenceEstimator
from confidence.calibration import calculate_ece

class TemperatureScaler(ConfidenceEstimator):
    """
    Temperature scaling for calibrating model confidence.

    This implementation supports both character-level and word-level calibration,
    and uses ECE minimization as the optimization objective.

    Args:
        model: The model to calibrate
        device: Device to run calibration on (cuda/cpu)
        converter: Converter to convert logits to probabilities
        level: 'word' or 'char' - level at which to apply calibration
    """
    def __init__(self, model, device, converter, level='word'):
        super().__init__(model, device)
        self.converter = converter
        self.level = level
        self.temperature = nn.Parameter(torch.ones(1).to(device))
        self.calibrated = False

    def calibrate(self, valid_loader, min_temp=0.1, max_temp=5.0, num_temps=20):
        """
        Calibrate temperature using grid search to minimize ECE.

        Args:
            valid_loader: Validation data loader
            min_temp: Minimum temperature to try
            max_temp: Maximum temperature to try
            num_temps: Number of temperature values to try
        """
        print(f"Calibrating temperature using {self.level}-level optimization...")
        self.model.eval()
        temp_values = torch.linspace(min_temp, max_temp, num_temps)
        best_ece = float('inf')
        best_temp = 1.0

        # Create results table
        results = []

        # Try different temperature values
        for temp in tqdm(temp_values, desc="Testing temperatures"):
            temp_val = temp.item()

            # Calculate confidences and correctness using this temperature
            all_confidences = []
            all_correctness = []

            with torch.no_grad():
                for images, labels in valid_loader:
                    images = images.to(self.device)

                    # Forward pass
                    logits = self.model(images)

                    # Apply this temperature
                    temp_logits = logits / temp
                    probs = F.softmax(temp_logits, dim=2)
                    max_probs, _ = probs.max(2)

                    # Decode predictions
                    decoded = self.converter.decode(temp_logits)

                    if self.level == 'word':
                        # Word-level calibration
                        batch_size = probs.size(1)
                        for b in range(batch_size):
                            # Get sequence probabilities
                            seq_probs = max_probs[:, b]

                            # Calculate confidence as product of probabilities (in log space)
                            log_probs = torch.log(seq_probs + 1e-10)
                            sum_log_prob = torch.sum(log_probs)
                            confidence = torch.exp(sum_log_prob).item()

                            # Check if prediction is correct
                            is_correct = 1 if decoded[b] == labels[b] else 0

                            all_confidences.append(confidence)
                            all_correctness.append(is_correct)

                    else:  # Character-level calibration
                        # For each position in the sequence
                        seq_len = probs.size(0)
                        batch_size = probs.size(1)

                        for b in range(batch_size):
                            # Check if prediction is correct at word level
                            word_correct = decoded[b] == labels[b]

                            # Get sequence probabilities for this batch item
                            for t in range(seq_len):
                                confidence = max_probs[t, b].item()

                                # We use word correctness as a proxy for character correctness
                                # This is an approximation since CTC alignment is ambiguous
                                all_confidences.append(confidence)
                                all_correctness.append(1 if word_correct else 0)

            # Calculate ECE for this temperature
            ece, _, _, _, _ = calculate_ece(all_confidences, all_correctness)

            # Store results
            results.append((temp_val, ece))

            if ece < best_ece:
                best_ece = ece
                best_temp = temp_val

        # Print results in a nice table
        print("\nTemperature Scaling Results:")
        print("-" * 30)
        print(f"{'Temperature':<15} {'ECE':<15}")
        print("-" * 30)
        for temp, ece in results:
            print(f"{temp:<15.4f} {ece:<15.4f}")
        print("-" * 30)

        # Set temperature to best value
        self.temperature.data = torch.tensor([best_temp], device=self.device)
        print(f"Best temperature: {best_temp:.4f}, ECE: {best_ece:.4f}")
        self.calibrated = True

    def get_scaled_logits(self, logits):
        """
        Apply temperature scaling to logits.

        Args:
            logits: Model output logits [seq_len, batch_size, vocab_size]

        Returns:
            Scaled logits with the same shape
        """
        if not self.calibrated:
            print("Warning: Using temperature scaler before calibration")

        # Apply temperature scaling
        return logits / self.temperature

    def get_confidence(self, logits):
        """
        Get calibrated confidence scores for predictions.

        Args:
            logits: Model output logits [seq_len, batch_size, vocab_size]

        Returns:
            Dictionary with decoded texts and confidence scores
        """
        # Apply temperature scaling
        scaled_logits = self.get_scaled_logits(logits)

        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=2)

        # Get the highest probability and index at each timestep
        max_probs, max_indices = probs.max(2)

        # Decode predicted sequences
        decoded = self.converter.decode(scaled_logits)

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
