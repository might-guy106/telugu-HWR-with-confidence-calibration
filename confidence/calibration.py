import numpy as np
import matplotlib.pyplot as plt

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


def plot_reliability_diagram(confidences, correctness, title="Reliability Diagram",
                            save_path=None, num_bins=10):
    """
    Plot reliability diagram to visualize calibration.

    Args:
        confidences: List of confidence scores
        correctness: List of binary values (1 for correct, 0 for incorrect)
        title: Plot title
        save_path: Path to save the plot
        num_bins: Number of bins for binning

    Returns:
        Tuple of (ECE, MCE)
    """
    ece, bin_accuracies, bin_confidences, bin_counts, bin_positions = calculate_ece(
        confidences, correctness, num_bins)

    mce = calculate_mce(bin_accuracies, bin_confidences, bin_counts)

    plt.figure(figsize=(10, 8))

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    # Plot accuracies as bars
    width = 1.0 / num_bins
    plt.bar(bin_positions, bin_accuracies, width=width, alpha=0.3, color='b',
            edgecolor='black', label='Accuracy')

    # Plot gaps between accuracy and confidence
    for i, (acc, conf) in enumerate(zip(bin_accuracies, bin_confidences)):
        if bin_counts[i] > 0 and abs(acc - conf) > 0.05:
            plt.plot([bin_positions[i], bin_positions[i]], [acc, conf], 'r-', linewidth=2)

    # Plot average confidences
    plt.plot(bin_positions, bin_confidences, 'ro-', linewidth=2, label='Average Confidence')

    # Add bin counts as labels
    for i, count in enumerate(bin_counts):
        if count > 0:
            plt.text(bin_positions[i], 0.05, f"{int(count)}",
                     ha='center', va='bottom', fontsize=8)

    plt.title(f"{title}\nECE: {ece:.4f}, MCE: {mce:.4f}")
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return ece, mce


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
