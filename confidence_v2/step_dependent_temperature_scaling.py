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
    """

    def __init__(self, model, device, converter, max_seq_len=50, tau=10, initial_temp=1.0, agg_method='min'):
        super().__init__(model, device, converter, agg_method)
        self.name = "Step Dependent T-Scaling"
        self.max_seq_len = max_seq_len
        self.tau = min(tau, max_seq_len)

        # Initialize temperature parameters
        self.temperatures = nn.Parameter(
            torch.ones(self.tau + 1, device=device) * initial_temp
        )

        self.calibrated = False

    def calibrate(self, val_loader, output_dir=None,  initial_lr=0.1, min_lr=0.001, num_iterations=100):
        """
        Calibrate step-dependent temperatures using validation data with adaptive learning rates.
        """
        self.model.eval()
        print(f"Calibrating {self.name} using joint optimization with adaptive learning rate...")

        # Collect validation predictions and ground truth (unchanged)
        print("Collecting validation predictions...")
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Processing validation data"):
                images = images.to(self.device)
                outputs = self.model(images)
                all_logits.append(outputs)
                all_labels.extend(labels)

        # Initialize temperatures with increasing values (unchanged)
        init_temps = torch.ones(self.tau + 1, device=self.device)
        for i in range(self.tau + 1):
            init_temps[i] = 1.0 + (i * 0.05)  # Start with 1.0, 1.05, 1.10, etc.

        temperatures = nn.Parameter(init_temps)
        print(f"Initial temperatures: {temperatures.data}")

        # Use Adam optimizer with initial learning rate
        optimizer = optim.Adam([temperatures], lr=initial_lr)

        # Add adaptive learning rate scheduler
        # ReduceLROnPlateau reduces LR when improvement stalls
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',          # Minimize ECE
            factor=0.5,          # Reduce LR by half when triggered
            patience=3,          # Wait 3 iterations without improvement
            threshold=0.0001,    # Minimum significant improvement
            min_lr=min_lr,       # Don't go below this LR
        )

        print(f"Starting optimization with initial learning rate: {initial_lr}...")

        # Optimization loop (with tracking)
        best_ece = float('inf')
        best_temps = temperatures.clone().detach()
        patience = 8  # Increased patience since we have adaptive LR
        patience_counter = 0
        min_improvement = 0.0005  # Minimum meaningful improvement

        # Lists to track progress
        all_temps_history = []  # Will store temperature values at each iteration
        all_eces_history = []   # Will store ECE values at each iteration
        all_lrs_history = []    # Will store learning rates at each iteration

        for iteration in tqdm(range(num_iterations), desc="Optimization progress"):
            optimizer.zero_grad()

            # Calculate ECE with current temperatures (implementation unchanged)
            all_confidences = []
            all_correctness = []

            # Process each batch of logits
            for logits_batch in all_logits:
                batch_size = logits_batch.size(1)
                seq_len = logits_batch.size(0)

                # Apply step-dependent temperatures
                scaled_logits = torch.zeros_like(logits_batch)
                for j in range(seq_len):
                    temp_idx = min(j, self.tau)
                    # Safe temperature value
                    temp_value = torch.clamp(temperatures[temp_idx], min=0.1)
                    scaled_logits[j] = logits_batch[j] / temp_value

                # Get predictions (detached from graph for correctness calculation)
                with torch.no_grad():
                    preds = self.converter.decode(scaled_logits)

                # Get softmax probabilities (with grad)
                probs = F.softmax(scaled_logits, dim=2)
                max_probs, _ = probs.max(2)

                # Process each sample in batch
                for b in range(batch_size):
                    seq_probs = max_probs[:, b]

                     # Get confidence (with grad)
                    if self.agg_method == 'geometric_mean':
                        # Length-normalized confidence
                        log_probs = torch.log(seq_probs + 1e-10)
                        confidence = torch.exp(torch.mean(log_probs))
                    elif self.agg_method == 'product':
                        # Product of probabilities
                        confidence = torch.prod(seq_probs)
                    elif self.agg_method == 'min':
                        # Minimum probability
                        confidence = torch.min(seq_probs)
                    else:
                        confidence = torch.mean(seq_probs)

                    # Store confidence and correctness
                    idx = len(all_confidences)
                    if idx < len(all_labels):
                        all_confidences.append(confidence)
                        label_correct = 1 if preds[b] == all_labels[idx] else 0
                        all_correctness.append(label_correct)

            # Calculate ECE with current temperatures
            ece = self._differentiable_ece(all_confidences, all_correctness)
            ece_value = ece.item()

            # Store current values for tracking
            all_temps_history.append(temperatures.detach().cpu().numpy().copy())
            all_eces_history.append(ece_value)
            all_lrs_history.append(optimizer.param_groups[0]['lr'])

            # Backward pass and optimization
            ece.backward()
            if iteration == 0:
                print("Temperature gradients:", temperatures.grad)
            optimizer.step()

            # Step the scheduler based on current ECE
            scheduler.step(ece_value)
            current_lr = optimizer.param_groups[0]['lr']

            # Print current values
            print(f"Iteration {iteration}: ECE={ece_value:.4f}, LR={current_lr:.6f}, Temps={temperatures.detach().cpu().numpy()}")

            # Check for improvement
            improvement = best_ece - ece_value
            if improvement > min_improvement:
                best_ece = ece_value
                best_temps = temperatures.clone().detach()
                print(f"Iteration {iteration}: New best ECE: {best_ece:.4f} (improved by {improvement:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No significant improvement, patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping due to lack of improvement")
                    break

            # Check if learning rate is too small
            if current_lr <= min_lr * 1.1:
                print(f"Learning rate {current_lr:.6f} too small, stopping optimization")
                break

        # Set the best temperatures to our class parameter
        self.temperatures.data = best_temps.data
        print(f"Calibration complete. Best ECE: {best_ece:.4f}")
        print(f"Final temperatures: {self.temperatures.detach().cpu().numpy()}")

        # Create visualization of temperatures by position (unchanged)
        if output_dir:
            vis_dir = os.path.join(output_dir, 'Step_Dependent_T_Scaling')
        else:
            # Fallback to default path if no output directory is provided
            vis_dir = os.path.join(os.getcwd(), 'output', 'confidence_evaluation_min', 'Step_Dependent_T_Scaling')
        os.makedirs(vis_dir, exist_ok=True)

        plot_temperature_by_position(
            best_temps.detach().cpu().numpy(),
            self.tau,
            save_path=os.path.join(vis_dir, 'temperature_by_position.png')
        )

        # Plot optimization progress including learning rate
        self._plot_optimization_progress_with_lr(
            all_temps_history, all_eces_history, all_lrs_history,
            best_temp=best_temps.cpu().numpy(),
            save_path=os.path.join(vis_dir, 'optimization_progress.png')
        )

        self.calibrated = True
        return best_temps.cpu().numpy()

    def _plot_optimization_progress_with_lr(self, temperatures, eces, lrs, best_temp, save_path):
        """Plot the optimization progress including learning rate."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 10))

        # Plot ECE progress
        plt.subplot(2, 2, 1)
        plt.plot(eces, marker='o', linestyle='-', markersize=3)
        plt.title('ECE Improvement During Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Expected Calibration Error')
        plt.grid(True, alpha=0.3)

        # Plot learning rate progress
        plt.subplot(2, 2, 2)
        plt.plot(lrs, marker='s', linestyle='-', color='green', markersize=3)
        plt.title('Learning Rate Adaptation')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.yscale('log')  # Log scale for better visibility
        plt.grid(True, alpha=0.3)

        # Plot final temperatures
        plt.subplot(2, 2, 3)
        positions = np.arange(len(best_temp))
        plt.bar(positions, best_temp)
        plt.title('Optimized Temperatures by Position')
        plt.xlabel('Sequence Position')
        plt.ylabel('Temperature Value')
        plt.grid(True, alpha=0.3, axis='y')

        # Plot temperature evolution (heatmap)
        if len(temperatures) > 1:
            plt.subplot(2, 2, 4)
            temps_array = np.array(temperatures)
            plt.imshow(
                temps_array.T,  # Transpose to show positions on y-axis
                aspect='auto',
                interpolation='none',
                cmap='viridis'
            )
            plt.colorbar(label='Temperature Value')
            plt.title('Temperature Evolution Over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Position')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _differentiable_ece(self, confidences, correctness, num_bins=10):
        """Calculate differentiable ECE with tensor inputs."""
        # Handle tensor vs list inputs
        if isinstance(confidences, list):
            if all(isinstance(c, torch.Tensor) for c in confidences):
                # Stack tensors directly (preserving gradients)
                stacked_confidences = torch.stack(confidences)
            else:
                # Convert to tensor if needed (this branch should not happen after fix)
                stacked_confidences = torch.tensor(confidences, device=self.device, requires_grad=True)
        else:
            # Already a tensor
            stacked_confidences = confidences

        correctness_tensor = torch.tensor(correctness, dtype=torch.float, device=self.device)

        # Bin edges
        bin_edges = torch.linspace(0, 1, num_bins+1, device=self.device)

        # Initialize ECE with requires_grad=True
        ece = torch.tensor(0.0, device=self.device)

        # Process each bin
        for i in range(num_bins):
            # Find confidence values in this bin
            low, high = bin_edges[i], bin_edges[i+1]
            in_bin = ((stacked_confidences >= low) & (stacked_confidences < high))

            # If there are elements in this bin
            if torch.any(in_bin):
                # Extract values in the bin
                bin_conf = stacked_confidences[in_bin]
                bin_corr = correctness_tensor[in_bin]

                # Calculate bin statistics (avoid in-place operations)
                bin_size = in_bin.sum().float()
                bin_conf_mean = torch.mean(bin_conf) if len(bin_conf) > 0 else torch.tensor(0.0, device=self.device)
                bin_acc = torch.mean(bin_corr) if len(bin_corr) > 0 else torch.tensor(0.0, device=self.device)

                # Add to ECE using operations that preserve the graph
                weight = (bin_size / len(stacked_confidences))
                calibration_error = torch.abs(bin_acc - bin_conf_mean)
                ece = ece + weight * calibration_error

        return ece

    def get_confidence(self, images, agg_method=None):
        """
        Get calibrated confidence scores using step-dependent temperature scaling.
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
