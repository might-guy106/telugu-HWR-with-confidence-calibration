import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import argparse
import torch
import json
import logging
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TeluguHWRDataset
from data.transforms import get_trocr_transforms
from models.trocr import TrOCRModel
from utils.metrics import calculate_metrics
from confidence_v2.trocr_confidence import (
    TrOCRBaseConfidence,
    TrOCRTemperatureScaling,
    TrOCRMCDropout
)
from confidence_v2.visualization import (
    plot_enhanced_reliability_diagram,
    plot_confidence_comparison
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def evaluate_confidence_method(estimator, test_loader, save_dir):
    """
    Evaluate a confidence estimation method.

    Args:
        estimator: ConfidenceEstimator instance
        test_loader: Test data loader
        save_dir: Directory to save results

    Returns:
        Dictionary with evaluation results
    """
    method_name = estimator.name
    agg_method = estimator.agg_method
    method_dir = os.path.join(save_dir, method_name.replace(' ', '_'))
    os.makedirs(method_dir, exist_ok=True)

    logger.info(f"Evaluating {method_name} with {agg_method} aggregation...")

    # Lists to store results
    all_predictions = []
    all_ground_truth = []
    all_correctness = []
    all_confidences = []
    all_uncertainties = []
    all_pred_dists = []

    # Process each batch
    for images, labels in tqdm(test_loader, desc=f"Evaluating {method_name}"):
        # Get predictions and confidence scores
        results = estimator.get_confidence(images)

        # Extract predictions and confidences
        predictions = results['predictions']
        confidences = results['confidences']

        # Store results
        all_predictions.extend(predictions)
        all_ground_truth.extend(labels)

        # Calculate correctness (1 for correct, 0 for incorrect)
        batch_correctness = [1 if p == gt else 0 for p, gt in zip(predictions, labels)]
        all_correctness.extend(batch_correctness)
        all_confidences.extend(confidences)

        # Store uncertainty values if available
        if 'uncertainties' in results:
            all_uncertainties.extend(results['uncertainties'])

        # Store prediction distributions if available
        if 'prediction_distributions' in results:
            all_pred_dists.extend(results['prediction_distributions'])

    # Calculate accuracy
    accuracy = sum(all_correctness) / len(all_correctness) * 100
    logger.info(f"{method_name} accuracy: {accuracy:.2f}%")

    # Create reliability diagram
    ece, mce, bs = plot_enhanced_reliability_diagram(
        all_confidences,
        all_correctness,
        title=f"{method_name} Calibration ({agg_method})",
        save_path=os.path.join(method_dir, "reliability_diagram.png")
    )

    logger.info(f"{method_name} metrics - ECE: {ece:.4f}, MCE: {mce:.4f}, Brier: {bs:.4f}")

    # Compile results
    results = {
        'method': method_name,
        'aggregation_method': agg_method,
        'accuracy': accuracy,
        'ece': ece,
        'mce': mce,
        'brier_score': bs,
        'examples': []
    }

    # Add examples
    for i in range(min(20, len(all_predictions))):
        example = {
            'ground_truth': all_ground_truth[i],
            'prediction': all_predictions[i],
            'correct': bool(all_correctness[i]),
            'confidence': all_confidences[i]
        }

        if all_uncertainties and i < len(all_uncertainties):
            example['uncertainty'] = all_uncertainties[i]

        if all_pred_dists and i < len(all_pred_dists):
            example['prediction_distribution'] = all_pred_dists[i]

        results['examples'].append(example)

    # Save results to JSON
    with open(os.path.join(method_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    return results

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {args.output_dir}")

    # Create transforms for TrOCR
    test_transform = get_trocr_transforms('test', args.img_height, args.img_width)
    val_transform = get_trocr_transforms('val', args.img_height, args.img_width)

    # Load datasets
    logger.info("Loading datasets...")
    val_dataset = TeluguHWRDataset(
        data_file=os.path.join(args.data_root, args.val_file),
        root_dir=args.data_root,
        transform=val_transform,
        max_samples=args.val_samples
    )

    test_dataset = TeluguHWRDataset(
        data_file=os.path.join(args.data_root, args.test_file),
        root_dir=args.data_root,
        transform=test_transform,
        max_samples=args.test_samples
    )

    # Create data loaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Create TrOCR model
    logger.info(f"Loading TrOCR model from {args.model_path}")
    model = TrOCRModel.from_pretrained(args.model_path, device=device)
    model.eval()

    # Use the aggregation method from command line args
    agg_method = args.agg_method
    logger.info(f"Using aggregation method: {agg_method}")

    # Create confidence estimators
    logger.info("Creating confidence estimators...")
    base_confidence = TrOCRBaseConfidence(model, device, agg_method=agg_method)
    temperature_scaling = TrOCRTemperatureScaling(model, device, agg_method=agg_method)
    mc_dropout = TrOCRMCDropout(model, device, num_samples=args.num_samples, agg_method=agg_method)

    # Calibrate estimators using validation data
    logger.info("Calibrating confidence estimators...")
    temperature_scaling.calibrate(val_loader)

    # Evaluate each method on test data
    logger.info("Evaluating calibrated methods on test data...")
    all_test_results = []

    # We'll evaluate these estimators
    estimators_to_evaluate = [base_confidence, temperature_scaling]
    if args.mc_dropout:
        estimators_to_evaluate.append(mc_dropout)

    for estimator in estimators_to_evaluate:
        result = evaluate_confidence_method(estimator, test_loader, args.output_dir)
        all_test_results.append(result)

    # Create comparative visualization
    plot_confidence_comparison(
        all_test_results,
        save_path=os.path.join(args.output_dir, "trocr_comparison.png")
    )

    logger.info("TrOCR confidence evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TrOCR Confidence for Telugu HWR")

    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root directory for dataset')
    parser.add_argument('--val_file', type=str, required=True,
                      help='Validation annotation file')
    parser.add_argument('--test_file', type=str, required=True,
                      help='Test annotation file')
    parser.add_argument('--output_dir', type=str, default='./output/trocr_confidence_evaluation',
                      help='Output directory for saving results')
    parser.add_argument('--val_samples', type=int, default=None,
                      help='Maximum validation samples to use (None for all)')
    parser.add_argument('--test_samples', type=int, default=None,
                      help='Maximum test samples to use (None for all)')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained TrOCR model')
    parser.add_argument('--img_height', type=int, default=384,
                      help='Input image height')
    parser.add_argument('--img_width', type=int, default=384,
                      help='Input image width')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')

    # Confidence parameters
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples for MC Dropout')
    parser.add_argument('--mc_dropout', action='store_true',
                      help='Use MC Dropout model for evaluation')
    parser.add_argument('--agg_method', type=str, default='min',
                      choices=['geometric_mean', 'product', 'min'],
                      help='Aggregation method for confidence scores')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--cuda', action='store_true',
                      help='Use CUDA if available')

    args = parser.parse_args()
    main(args)
