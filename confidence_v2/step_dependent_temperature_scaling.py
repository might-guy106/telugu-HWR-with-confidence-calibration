import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .visualization import plot_temperature_by_position


from .base import ConfidenceEstimator

class StepDependentTemperatureScaling(ConfidenceEstimator):
    """
    Step Dependent Temperature Scaling (STS) for calibrating sequence model confidence.

    STS applies different temperature values to different positions in the sequence,
    which can better handle the varying confidence patterns at different sequence steps.

    Args:
        model: The model to calibrate
        device: Device to run calibration on (cuda/cpu)
        converter: Converter to convert logits to probabilities
        max_seq_len: Maximum sequence length to consider
        tau: Meta-parameter controlling position sharing (positions > tau share same temperature)
        initial_temp: Initial temperature value for all positions
    """

    def __init__(self, model, device, converter, max_seq_len=50, tau=10, initial_temp=1.0, agg_method='geometric_mean'):
        super().__init__(model, device, converter, agg_method)
        self.name = "Step Dependent T-Scaling"
        self.max_seq_len = max_seq_len
        self.tau = min(tau, max_seq_len)

        # Initialize temperature parameters
        self.temperatures = nn.Parameter(
            torch.ones(self.tau + 1, device=device) * initial_temp
        )

        self.calibrated = False

    def calibrate(self, val_loader,min_temp=1.0, max_temp=1.5, num_temps=10):
        """
        Calibrate step-dependent temperatures using validation data.

        Args:
            val_loader: Validation data loader
        """
        self.model.eval()
        print(f"Calibrating {self.name} using validation data...")

        # Collect validation predictions and ground truth
        print("Collecting validation predictions...")
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Processing validation data"):
                images = images.to(self.device)

                # Get model outputs
                outputs = self.model(images)

                # Store outputs and labels
                all_logits.append(outputs)
                all_labels.extend(labels)

        # For each temperature value in a reasonable range, calculate ECE
        # We'll use grid search for each position independently
        temp_range = np.linspace(min_temp, max_temp, num_temps)

        # Function to calculate correctness
        def get_correctness(preds, true_labels):
            return [1 if p == l else 0 for p, l in zip(preds, true_labels)]

        # Track best temperatures for each position
        best_temps = torch.ones(self.tau + 1, device=self.device)
        best_eces = torch.ones(self.tau + 1, device=self.device) * float('inf')

        print(f"Optimizing temperatures for {self.tau + 1} positions...")

        # We'll optimize each position independently
        for pos in range(self.tau + 1):
            pos_name = f"position {pos}" if pos < self.tau else f"positions >{self.tau-1}"
            print(f"Optimizing temperature for {pos_name}...")

            eces_by_temp = []

            for temp in tqdm(temp_range, desc=f"Testing temperatures for {pos_name}", leave=False):
                # Create a temporary set of temperatures with current position's temp modified
                temps = best_temps.clone()
                temps[pos] = temp

                # Apply these temperatures to get predictions
                all_confidences = []
                all_correctness = []

                for logits_batch in all_logits:
                    batch_size = logits_batch.size(1)
                    seq_len = logits_batch.size(0)

                    # Apply step-dependent temperatures
                    scaled_logits = torch.zeros_like(logits_batch)

                    for j in range(seq_len):
                        # Use position-specific temp or shared temp for positions > tau
                        temp_idx = min(j, self.tau)
                        scaled_logits[j] = logits_batch[j] / temps[temp_idx]

                    # Get predictions
                    preds = self.converter.decode(scaled_logits)

                    # Get probabilities
                    probs = F.softmax(scaled_logits, dim=2)
                    max_probs, _ = probs.max(2)

                    # Calculate confidences
                    for b in range(batch_size):
                        # Get sequence probabilities
                        seq_probs = max_probs[:, b]

                        # Use calculate_confidence from base class
                        confidence = self.calculate_confidence(seq_probs)

                        # Store confidence and correctness for this sample
                        idx = len(all_confidences)
                        if idx < len(all_labels):
                            all_confidences.append(confidence)
                            label_correct = 1 if preds[b] == all_labels[idx] else 0
                            all_correctness.append(label_correct)

                # Calculate ECE for this temperature
                ece = self._calculate_ece(all_confidences, all_correctness)
                eces_by_temp.append(ece)

                # Update if this is the best temperature for this position
                if ece < best_eces[pos].item():
                    best_eces[pos] = torch.tensor(ece, device=self.device)
                    best_temps[pos] = torch.tensor(temp, device=self.device)
                    print(f"New best temp for {pos_name}: {temp:.4f}, ECE: {ece:.4f}")

            # Store temperature vs ECE curve for visualization
            if pos == 0:  # Just save for the first position to avoid too many plots
                self._plot_temp_ece_curve(temp_range, eces_by_temp, best_temps[pos].item(),
                                      best_eces[pos].item(), pos)

        # Set the best temperatures
        self.temperatures.data = best_temps
        # Create visualization of temperatures by position
        vis_dir = os.path.join(os.getcwd(), 'output', 'confidence_evaluation_v2', 'Step_Dependent_T_Scaling')
        os.makedirs(vis_dir, exist_ok=True)

        plot_temperature_by_position(
            best_temps.detach().cpu().numpy(),
            self.tau,
            save_path=os.path.join(vis_dir, 'temperature_by_position.png')
        )
        print(f"Calibration complete. Temperatures: {best_temps.detach().cpu().numpy()}")


        self.calibrated = True
        # Add at the end of the calibrate method
        conf_type = "length-normalized (geometric mean)" if self.normalize_confidence else "unnormalized (product)"
        print(f"Confidence type used: {conf_type}")
        return best_temps.cpu().numpy()

    def get_confidence(self, images, agg_method=None):
        """
        Get calibrated confidence scores using step-dependent temperature scaling.

        Args:
            images: Input image batch
            agg_method: Aggregation method for confidence scores

        Returns:
            Dictionary with predictions and confidence scores
        """
        if not self.calibrated:
            print(f"Warning: Using {self.name} before calibration")

        # Get model outputs
        logits, _ = self._get_model_predictions(images)

        # Apply step-dependent temperature scaling
        batch_size = logits.size(1)
        seq_len = logits.size(0)

        scaled_logits = torch.zeros_like(logits)

        for j in range(seq_len):
            # Use position-specific temp or shared temp for positions > tau
            temp_idx = min(j, self.tau)
            scaled_logits[j] = logits[j] / self.temperatures[temp_idx]

        # Get probabilities
        probs = F.softmax(scaled_logits, dim=2)
        max_probs, _ = probs.max(2)

        # Decode predictions
        decoded = self.converter.decode(scaled_logits)

        # Compute confidence for each sequence
        confidences = []

        for b in range(batch_size):
            # Get sequence probabilities for this batch item
            seq_probs = max_probs[:, b]

            # Calculate confidence with specified aggregation method
            confidence = self.calculate_confidence(seq_probs, agg_method)
            confidences.append(confidence)

        return {
            'predictions': decoded,
            'confidences': confidences,
            'temperatures': self.temperatures.detach().cpu().numpy(),
            'method': self.name,
            'agg_method': agg_method or self.agg_method
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

    def _plot_temp_ece_curve(self, temps, eces, best_temp, best_ece, position):
        """Plot temperature vs ECE curve for a specific position."""
        import matplotlib.pyplot as plt
        import os

        # Create directory
        vis_dir = os.path.join(os.getcwd(), 'output', 'confidence_evaluation_v2', 'Step_Dependent_T_Scaling')
        os.makedirs(vis_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))

        # Plot ECE vs temperature
        plt.plot(temps, eces, 'b-', linewidth=2, alpha=0.7)
        plt.scatter(temps, eces, color='blue', s=30, alpha=0.5)

        # Highlight best temperature
        plt.scatter([best_temp], [best_ece], color='red', s=100,
                    label=f'Best T={best_temp:.2f}, ECE={best_ece:.4f}')

        # Add vertical line at T=1.0
        plt.axvline(x=1.0, color='green', linestyle='--', alpha=0.7,
                   label='T=1.0 (No scaling)')

        # Format plot
        position_name = f"Position {position}" if position < self.tau else f"Positions >{self.tau-1}"
        plt.title(f'Temperature vs ECE for {position_name}')
        plt.xlabel('Temperature')
        plt.ylabel('Expected Calibration Error (ECE)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save plot
        plt.savefig(os.path.join(vis_dir, f'temp_ece_pos{position}.png'), dpi=300)
        plt.close()
