import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.calibration import calibration_curve
import seaborn as sns
from sklearn.metrics import precision_recall_curve

def calculate_ece(confidences, correctness, num_bins=10):
    """
    Calculate Expected Calibration Error.

    Args:
        confidences: List of confidence scores
        correctness: List of binary values (1 for correct, 0 for incorrect)
        num_bins: Number of bins for binning the confidence scores

    Returns:
        Tuple of (ECE, bin accuracies, bin confidences, bin counts, bin positions)
    """
    # Convert inputs to numpy arrays for processing
    confidences = np.array(confidences)
    correctness = np.array(correctness)

    # Create bins and digitize confidences
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
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

    bin_positions = (bin_lowers + bin_uppers) / 2

    return ece, bin_accuracies, bin_confidences, bin_counts, bin_positions


def calculate_mce(bin_accuracies, bin_confidences, bin_counts):
    """
    Calculate Maximum Calibration Error.

    Args:
        bin_accuracies: Accuracy in each bin
        bin_confidences: Average confidence in each bin
        bin_counts: Number of samples in each bin

    Returns:
        Maximum calibration error value
    """
    mce = 0
    for i in range(len(bin_counts)):
        if bin_counts[i] > 0:
            mce = max(mce, abs(bin_accuracies[i] - bin_confidences[i]))
    return mce


def brier_score(confidences, correctness):
    """
    Calculate Brier Score (mean squared error between confidence and correctness).

    Args:
        confidences: List of confidence scores
        correctness: List of binary values (1 for correct, 0 for incorrect)

    Returns:
        Brier score value
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    return np.mean((confidences - correctness) ** 2)

def plot_temperature_ece_curve(temperatures, eces, best_temp, best_ece, save_path=None, figsize=(10, 6)):
    """
    Plot the relationship between temperature values and ECE.

    Args:
        temperatures: List of temperature values
        eces: List of corresponding ECE values
        best_temp: Optimal temperature value
        best_ece: Best (minimum) ECE value
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Plot the ECE vs temperature curve
    plt.plot(temperatures, eces, 'b-', linewidth=2, alpha=0.7)
    plt.scatter(temperatures, eces, color='blue', s=30, alpha=0.5)

    # Highlight the best temperature
    plt.scatter([best_temp], [best_ece], color='red', s=100,
                label=f'Best T={best_temp:.2f}, ECE={best_ece:.4f}')

    # Add a vertical line at T=1.0 (no scaling)
    plt.axvline(x=1.0, color='green', linestyle='--', alpha=0.7,
               label='T=1.0 (No scaling)')

    # Set axis labels and title
    plt.xlabel('Temperature')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('Effect of Temperature on Calibration Error')

    # Add grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best')

    # Set x-axis limits with some padding
    max_temp = max(temperatures)
    plt.xlim(0, max_temp * 1.1)

    # Set y-axis to start from 0
    plt.ylim(0, max(eces) * 1.1)

    # Add annotations
    plt.annotate(f'Best ECE: {best_ece:.4f}',
                xy=(best_temp, best_ece),
                xytext=(best_temp + 0.5, best_ece + 0.02),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)

    # Finalize and save
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_temperature_by_position(temperatures, tau, save_path=None, figsize=(12, 6)):
    """
    Plot the relationship between sequence position and best temperature values.

    Args:
        temperatures: Array of temperature values by position
        tau: Meta-parameter indicating the position after which temperatures are shared
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Get positions
    positions = np.arange(len(temperatures))

    # Make the temperature for positions > tau visually distinct
    colors = ['blue' if pos < tau else 'red' for pos in positions]

    # Plot bar chart of temperatures by position
    bars = plt.bar(positions, temperatures, color=colors, alpha=0.7, width=0.6)

    # Add horizontal line at T=1.0 (no scaling)
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='T=1.0 (No scaling)')

    # Add vertical line at position tau
    if tau < len(temperatures) - 1:
        plt.axvline(x=tau - 0.5, color='red', linestyle='--', alpha=0.7,
                   label=f'τ = {tau} (Shared temperatures)')

    # Customize labels and title
    plt.xlabel('Sequence Position')
    plt.ylabel('Temperature')
    plt.title('Optimal Temperature by Sequence Position')

    # Limit y-axis to make differences more visible
    max_temp = max(temperatures) * 1.1
    plt.ylim(0, max(5.0, max_temp))  # Don't go below 5.0 to ensure visibility

    # Add exact values above each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')

    # Add legend
    plt.legend()

    # Add tick marks for each position
    plt.xticks(positions)

    # Add grid lines
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation for shared region
    if tau < len(temperatures) - 1:
        plt.annotate('Shared Temperature\nRegion',
                    xy=(len(temperatures) - 1, temperatures[-1]),
                    xytext=(tau + 1, temperatures[-1] + 0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    # Finalize and save
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_enhanced_reliability_diagram(confidences, correctness, title="Reliability Diagram",
                                      save_path=None, num_bins=10, figsize=(10, 6)):
    """
    Create simplified reliability diagram with only bin analysis and accuracy label.

    Args:
        confidences: List of confidence scores
        correctness: List of binary values (1 for correct, 0 for incorrect)
        title: Plot title
        save_path: Path to save the plot
        num_bins: Number of bins for binning
        figsize: Figure size

    Returns:
        Tuple of (ECE, MCE, Brier Score)
    """
    # Calculate calibration metrics
    ece, bin_accuracies, bin_confidences, bin_counts, bin_positions = calculate_ece(
        confidences, correctness, num_bins)

    mce = calculate_mce(bin_accuracies, bin_confidences, bin_counts)
    bs = brier_score(confidences, correctness)

    # Create figure
    plt.figure(figsize=figsize)

    # Set width and colors
    width = 1.0 / num_bins
    conf_color = 'lightblue'
    acc_color = 'navy'
    gap_color_pos = 'red'
    gap_color_neg = 'green'

    # Plot confidence bars
    plt.bar(bin_positions, bin_confidences, width=width, alpha=0.8, color=conf_color,
            edgecolor='black', label='Confidence')

    # Plot accuracy bars
    plt.bar(bin_positions, bin_accuracies, width=width, alpha=0.6, color=acc_color,
            edgecolor='black', label='Accuracy')

    # Plot gaps
    for i, (pos, conf, acc) in enumerate(zip(bin_positions, bin_confidences, bin_accuracies)):
        if bin_counts[i] > 0:
            gap = acc - conf
            gap_color = gap_color_pos if gap >= 0 else gap_color_neg
            if abs(gap) > 0.05:  # Only show significant gaps
                plt.plot([pos, pos], [conf, acc], color=gap_color, linewidth=2)
                plt.text(pos, (conf + acc) / 2, f"{gap:.2f}",
                         ha='center', va='center', fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    # Add sample counts as labels
    for i, count in enumerate(bin_counts):
        if count > 0:
            plt.text(bin_positions[i], 0.05, f"{int(count)}",
                     ha='center', va='bottom', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.6))

    # Perfect calibration reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')

    # Format plot
    plt.title(f"{title}\nECE: {ece:.4f}, MCE: {mce:.4f}, Brier Score: {bs:.4f}")
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')  # Changed from 'Frequency' to 'Accuracy'
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)

    # Finalize and save
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return ece, mce, bs

def plot_confidence_comparison(all_results, save_path=None, figsize=(14, 10)):
    """
    Compare different confidence methods.

    Args:
        all_results: List of dictionaries containing results for each method
                    Each dict should have 'method', 'ece', 'mce', 'brier_score', etc.
        save_path: Path to save the plot
        figsize: Figure size
    """
    methods = [r['method'] for r in all_results]
    eces = [r['ece'] for r in all_results]
    mces = [r['mce'] for r in all_results]
    briers = [r['brier_score'] for r in all_results]

    # Create figure
    plt.figure(figsize=figsize)

    # ECE comparison
    plt.subplot(3, 1, 1)
    bars = plt.bar(methods, eces, color='blue', alpha=0.7)
    plt.title('Expected Calibration Error (ECE)')
    plt.ylabel('ECE (lower is better)')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.4f}',
                 ha='center', va='bottom', fontsize=9)

    # MCE comparison
    plt.subplot(3, 1, 2)
    bars = plt.bar(methods, mces, color='red', alpha=0.7)
    plt.title('Maximum Calibration Error (MCE)')
    plt.ylabel('MCE (lower is better)')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.4f}',
                 ha='center', va='bottom', fontsize=9)

    # Brier score comparison
    plt.subplot(3, 1, 3)
    bars = plt.bar(methods, briers, color='green', alpha=0.7)
    plt.title('Brier Score')
    plt.ylabel('Brier Score (lower is better)')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.002, f'{height:.4f}',
                 ha='center', va='bottom', fontsize=9)

    # Finalize and save
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_uncertainty_analysis(epistemic, aleatoric, correctness, save_path=None, figsize=(12, 10)):
    """
    Analyze relationship between uncertainty measures and correctness.

    Args:
        epistemic: List of epistemic uncertainty values
        aleatoric: List of aleatoric uncertainty values
        correctness: List of binary values (1 for correct, 0 for incorrect)
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert to numpy arrays
    epistemic = np.array(epistemic)
    aleatoric = np.array(aleatoric)
    correctness = np.array(correctness)

    # Create figure
    plt.figure(figsize=figsize)

    # Scatter plot of uncertainties
    plt.subplot(2, 2, 1)
    correct_mask = (correctness == 1)
    plt.scatter(epistemic[correct_mask], aleatoric[correct_mask], alpha=0.5,
                color='green', label='Correct')
    plt.scatter(epistemic[~correct_mask], aleatoric[~correct_mask], alpha=0.5,
                color='red', label='Incorrect')
    plt.xlabel('Epistemic Uncertainty')
    plt.ylabel('Aleatoric Uncertainty')
    plt.title('Uncertainty Distribution by Correctness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Epistemic uncertainty distribution
    plt.subplot(2, 2, 2)
    plt.hist([epistemic[correct_mask], epistemic[~correct_mask]], bins=20, alpha=0.7,
            label=['Correct', 'Incorrect'], color=['green', 'red'])
    plt.xlabel('Epistemic Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Epistemic Uncertainty Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Aleatoric uncertainty distribution
    plt.subplot(2, 2, 3)
    plt.hist([aleatoric[correct_mask], aleatoric[~correct_mask]], bins=20, alpha=0.7,
            label=['Correct', 'Incorrect'], color=['green', 'red'])
    plt.xlabel('Aleatoric Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Aleatoric Uncertainty Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Total uncertainty
    plt.subplot(2, 2, 4)
    total_uncertainty = epistemic + aleatoric

    # Plot ROC curve or precision-recall
    precision, recall, _ = precision_recall_curve(1 - correctness, total_uncertainty)
    plt.plot(recall, precision, color='blue', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Error Detection Using Total Uncertainty')
    plt.grid(True, alpha=0.3)

    # Finalize and save
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
