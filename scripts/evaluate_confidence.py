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
from data.transforms import get_transforms
from models.crnn import CRNN
from models.mc_dropout_crnn import MCDropoutCRNN
from utils.ctc_decoder import CTCLabelConverter
from confidence_v2.uncalibrated import UncalibratedConfidence
from confidence_v2.temperature_scaling import TemperatureScaling
from confidence_v2.mc_dropout import MCDropoutConfidence
from confidence_v2.step_dependent_temperature_scaling import StepDependentTemperatureScaling
from confidence_v2.visualization import (
    plot_enhanced_reliability_diagram,
    plot_confidence_comparison,
    plot_uncertainty_analysis
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_vocabulary(vocab_file):
    """Load vocabulary from file"""
    vocab = set()
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.strip()
            if char:  # Skip empty lines
                vocab.add(char)
    return vocab

def evaluate_confidence_method(estimator, test_loader, save_dir, agg_method='geometric_mean'):
    """
    Evaluate a single confidence estimation method with a specific aggregation method.

    Args:
        estimator: ConfidenceEstimator instance
        test_loader: Test data loader
        save_dir: Directory to save results
        agg_method: Aggregation method for confidence calculation ('geometric_mean', 'product', or 'min')

    Returns:
        Dictionary with evaluation results
    """
    method_name = estimator.name
    method_dir = os.path.join(save_dir, method_name.replace(' ', '_'), agg_method)
    os.makedirs(method_dir, exist_ok=True)

    logger.info(f"Evaluating {method_name} with {agg_method} aggregation...")

    # Set the aggregation method for this evaluation
    original_agg_method = estimator.agg_method
    estimator.agg_method = agg_method

    # Lists to store results
    all_predictions = []
    all_ground_truth = []
    all_correctness = []
    all_confidences = []
    all_epistemic_unc = []
    all_pred_dists = []

    # Process each batch
    for images, labels in tqdm(test_loader, desc=f"Evaluating {method_name} ({agg_method})"):
        # Get predictions and confidence scores
        results = estimator.get_confidence(images, agg_method=agg_method)

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

        # Store epistemic uncertainty values if available
        if 'epistemic_uncertainties' in results:
            all_epistemic_unc.extend(results['epistemic_uncertainties'])

        # Store prediction distributions if available
        if 'prediction_distributions' in results:
            all_pred_dists.extend(results['prediction_distributions'])

    # Calculate accuracy
    accuracy = sum(all_correctness) / len(all_correctness) * 100
    logger.info(f"{method_name} ({agg_method}) accuracy: {accuracy:.2f}%")

    # Create reliability diagram
    ece, mce, bs = plot_enhanced_reliability_diagram(
        all_confidences,
        all_correctness,
        title=f"{method_name} Calibration ({agg_method})",
        save_path=os.path.join(method_dir, "reliability_diagram.png")
    )

    logger.info(f"{method_name} ({agg_method}) metrics - ECE: {ece:.4f}, MCE: {mce:.4f}, Brier: {bs:.4f}")

    # Plot uncertainty analysis if available
    if all_epistemic_unc:
        plot_uncertainty_analysis(
            all_epistemic_unc,
            [0] * len(all_epistemic_unc),  # Placeholder for aleatoric (not available)
            all_correctness,
            save_path=os.path.join(method_dir, "uncertainty_analysis.png")
        )

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

        if all_epistemic_unc:
            example['epistemic_uncertainty'] = all_epistemic_unc[i]

        if all_pred_dists:
            example['prediction_distribution'] = all_pred_dists[i]

        results['examples'].append(example)

    # Save results to JSON
    with open(os.path.join(method_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    # Restore the original aggregation method
    estimator.agg_method = original_agg_method

    return results

def evaluate_all_aggregation_methods(estimator, test_loader, save_dir, agg_methods):
    """
    Evaluate a confidence estimator with all specified aggregation methods.

    Args:
        estimator: ConfidenceEstimator instance
        test_loader: Test data loader
        save_dir: Directory to save results
        agg_methods: List of aggregation methods to evaluate

    Returns:
        Dictionary with best aggregation method and all results
    """
    all_results = []
    best_result = None
    best_agg_method = None
    best_ece = float('inf')

    for agg_method in agg_methods:
        result = evaluate_confidence_method(estimator, test_loader, save_dir, agg_method)
        all_results.append(result)

        # Track the best method based on ECE
        if result['ece'] < best_ece:
            best_ece = result['ece']
            best_result = result
            best_agg_method = agg_method

    return {
        'best_agg_method': best_agg_method,
        'best_ece': best_ece,
        'all_results': all_results
    }

def predict_with_selection(images, estimators, selection_method='best_confidence', agg_method='geometric_mean'):
    """
    Make predictions using flexible selection among multiple estimators.

    Args:
        images: Input images
        estimators: List of confidence estimators
        selection_method: Method for selecting predictions
            - 'best_confidence': Use prediction with highest confidence
            - 'ensemble': Majority voting weighted by confidence
            - Or specify an estimator name
        agg_method: Aggregation method for confidence calculation

    Returns:
        Dictionary with combined predictions and confidences
    """
    # Check if selection method is a specific estimator
    if selection_method not in ['best_confidence', 'ensemble']:
        # Only use the specified estimator
        for estimator in estimators:
            if estimator.name == selection_method:
                # Get predictions from just this estimator
                results = estimator.get_confidence(images, agg_method=agg_method)

                return {
                    'predictions': results['predictions'],
                    'confidences': results['confidences'],
                    'selected_methods': [estimator.name] * len(results['predictions']),
                    'agg_method': agg_method
                }

        # If we reach here, the specified estimator was not found
        raise ValueError(f"Estimator '{selection_method}' not found in available estimators")

    # For 'best_confidence' or 'ensemble', we need all estimators' predictions
    all_results = [estimator.get_confidence(images, agg_method=agg_method) for estimator in estimators]

    # Extract batch size
    batch_size = len(all_results[0]['predictions'])

    # Initialize final predictions and confidences
    final_predictions = []
    final_confidences = []
    selected_methods = []

    # Process each sample in the batch
    for i in range(batch_size):
        if selection_method == 'best_confidence':
            # Find estimator with highest confidence
            best_confidence = -1
            best_prediction = None
            best_estimator = None

            for results, estimator in zip(all_results, estimators):
                confidence = results['confidences'][i]
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_prediction = results['predictions'][i]
                    best_estimator = estimator.name

            final_predictions.append(best_prediction)
            final_confidences.append(best_confidence)
            selected_methods.append(best_estimator)

        elif selection_method == 'ensemble':
            # Weighted voting based on confidence
            prediction_votes = {}

            for results in all_results:
                prediction = results['predictions'][i]
                confidence = results['confidences'][i]

                if prediction not in prediction_votes:
                    prediction_votes[prediction] = 0

                prediction_votes[prediction] += confidence

            # Get prediction with highest weighted votes
            best_prediction = max(prediction_votes.items(), key=lambda x: x[1])
            final_prediction = best_prediction[0]
            final_confidence = best_prediction[1] / len(estimators)  # Normalize

            final_predictions.append(final_prediction)
            final_confidences.append(final_confidence)
            selected_methods.append('Ensemble')

    return {
        'predictions': final_predictions,
        'confidences': final_confidences,
        'selected_methods': selected_methods,
        'agg_method': agg_method
    }

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {args.output_dir}")

    # Load vocabulary
    vocab = load_vocabulary(args.vocab_file)
    logger.info(f"Loaded vocabulary with {len(vocab)} characters")

    # Create converter
    converter = CTCLabelConverter(vocab)

    # Create model
    logger.info(f"Loading model from {args.model_path}")
    if args.mc_dropout:
        model = MCDropoutCRNN(
            img_height=args.img_height,
            num_channels=1,
            num_classes=converter.vocab_size,
            rnn_hidden_size=args.hidden_size,
            dropout_rate=args.dropout_rate
        )
        logger.info("Using MCDropoutCRNN model")
    else:
        model = CRNN(
            img_height=args.img_height,
            num_channels=1,
            num_classes=converter.vocab_size,
            rnn_hidden_size=args.hidden_size
        )
        logger.info("Using standard CRNN model")

    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    # Create data loaders
    logger.info("Creating data loaders...")
    val_transform = get_transforms('val', args.img_height, args.img_width)
    test_transform = get_transforms('test', args.img_height, args.img_width)

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

    # Define aggregation methods to evaluate
    agg_methods = ['geometric_mean', 'product', 'min']
    logger.info(f"Testing {len(agg_methods)} aggregation methods: {', '.join(agg_methods)}")

    # Create confidence estimators with initial aggregation method
    logger.info("Creating confidence estimators...")
    initial_agg_method = args.agg_method
    logger.info(f"Initial aggregation method: {initial_agg_method}")

    uncalibrated = UncalibratedConfidence(model, device, converter, agg_method=initial_agg_method)
    temperature = TemperatureScaling(model, device, converter, agg_method=initial_agg_method)
    mc_dropout = MCDropoutConfidence(model, device, converter, num_samples=args.num_samples, agg_method=initial_agg_method)
    step_temp = StepDependentTemperatureScaling(
        model, device, converter,
        max_seq_len=50, tau=10, initial_temp=1.0,
        agg_method=initial_agg_method
    )

    # Calibrate estimators using validation data
    logger.info("Calibrating confidence estimators...")

    # Calibrate Temperature Scaling
    logger.info("Calibrating Temperature Scaling...")
    calib_result_temp = temperature.calibrate(val_loader)

    # MC dropout just needs to verify dropout layers
    logger.info("Initializing MC Dropout...")
    mc_dropout.calibrate(val_loader)

    # Calibrate Step Dependent T-Scaling
    logger.info("Calibrating Step-Dependent Temperature Scaling...")
    calib_result_step = step_temp.calibrate(val_loader)

    # Store all estimators, calibrated parameters, and best aggregation methods
    estimators = [uncalibrated, temperature, mc_dropout, step_temp]
    calibration_results = {}
    best_agg_methods = {}

    # Evaluate each method with all aggregation methods
    all_evaluation_results = []

    for estimator in estimators:
        logger.info(f"Evaluating {estimator.name} with all aggregation methods...")
        result = evaluate_all_aggregation_methods(estimator, test_loader, args.output_dir, agg_methods)
        best_agg_methods[estimator.name] = result['best_agg_method']

        # Add calibration parameters if available
        if estimator.name == "Temperature Scaling":
            calibration_results[estimator.name] = {
                'temperature': temperature.temperature.item(),
                'best_aggregation_method': result['best_agg_method']
            }
        elif estimator.name == "Step Dependent T-Scaling":
            calibration_results[estimator.name] = {
                'temperatures': step_temp.temperatures.detach().cpu().numpy().tolist(),
                'tau': step_temp.tau,
                'best_aggregation_method': result['best_agg_method']
            }
        elif estimator.name == "MC Dropout":
            calibration_results[estimator.name] = {
                'num_samples': mc_dropout.num_samples,
                'best_aggregation_method': result['best_agg_method'],
                'note': "MC Dropout estimates only epistemic uncertainty, not aleatoric"
            }
        else:
            calibration_results[estimator.name] = {
                'best_aggregation_method': result['best_agg_method']
            }

        # Collect best results for comparison
        all_evaluation_results.append(result['all_results'][0])  # Add best result for comparison

    # Create comparative visualizations
    plot_confidence_comparison(
        all_evaluation_results,
        save_path=os.path.join(args.output_dir, "comparison.png")
    )

    # Create directory for saving calibration parameters
    calib_dir = os.path.join(args.output_dir, 'calibration_params')
    os.makedirs(calib_dir, exist_ok=True)
    logger.info(f"Saving calibration parameters to {calib_dir}")

    # Save all calibration results
    with open(os.path.join(calib_dir, 'calibration_results.json'), 'w') as f:
        json.dump(calibration_results, f, indent=4)
    logger.info("Saved calibration results")

    # Save temperature scaling parameters separately for easier loading
    with open(os.path.join(calib_dir, 'temperature_scaling.json'), 'w') as f:
        json.dump({
            'temperature': temperature.temperature.item(),
            'calibration_method': 'temperature_scaling',
            'best_aggregation_method': best_agg_methods.get('Temperature Scaling', 'geometric_mean')
        }, f, indent=4)

    # Save step-dependent parameters
    with open(os.path.join(calib_dir, 'step_dependent_scaling.json'), 'w') as f:
        json.dump({
            'temperatures': step_temp.temperatures.detach().cpu().numpy().tolist(),
            'tau': step_temp.tau,
            'calibration_method': 'step_dependent_scaling',
            'best_aggregation_method': best_agg_methods.get('Step Dependent T-Scaling', 'geometric_mean')
        }, f, indent=4)

    # Save recommended aggregation methods separately
    with open(os.path.join(calib_dir, 'aggregation_methods.json'), 'w') as f:
        json.dump(best_agg_methods, f, indent=4)

    # If prediction selection method is specified, evaluate combined predictions
    if args.prediction_source != 'none':
        logger.info(f"Evaluating with prediction selection: {args.prediction_source}")

        all_selection_results = []

        # Test each aggregation method for prediction selection
        for agg in agg_methods:
            logger.info(f"Testing selection with aggregation method: {agg}")

            selected_predictions = []
            selected_confidences = []
            selected_methods = []
            ground_truth = []
            correctness = []

            for images, labels in tqdm(test_loader, desc=f"Prediction Selection ({agg})"):
                # Get predictions using selection method
                results = predict_with_selection(images, estimators, args.prediction_source, agg)

                # Store results
                selected_predictions.extend(results['predictions'])
                selected_confidences.extend(results['confidences'])
                selected_methods.extend(results['selected_methods'])
                ground_truth.extend(labels)

                # Calculate correctness
                batch_correctness = [1 if p == gt else 0 for p, gt in
                                   zip(results['predictions'], labels)]
                correctness.extend(batch_correctness)

            # Calculate accuracy
            accuracy = sum(correctness) / len(correctness) * 100

            # Evaluate calibration
            selection_dir = os.path.join(args.output_dir, f"Selection_{args.prediction_source}", agg)
            os.makedirs(selection_dir, exist_ok=True)

            ece, mce, bs = plot_enhanced_reliability_diagram(
                selected_confidences,
                correctness,
                title=f"Selection: {args.prediction_source} ({agg})",
                save_path=os.path.join(selection_dir, "reliability_diagram.png")
            )

            # Compile results
            selection_results = {
                'method': f"Selection: {args.prediction_source}",
                'aggregation_method': agg,
                'accuracy': accuracy,
                'ece': ece,
                'mce': mce,
                'brier_score': bs,
                'examples': []
            }

            # Add selection statistics
            method_counts = {}
            for method in selected_methods:
                if method not in method_counts:
                    method_counts[method] = 0
                method_counts[method] += 1

            selection_results['selection_stats'] = {
                'method_counts': method_counts,
                'total_samples': len(selected_methods)
            }

            # Add examples
            for i in range(min(20, len(selected_predictions))):
                selection_results['examples'].append({
                    'ground_truth': ground_truth[i],
                    'prediction': selected_predictions[i],
                    'confidence': selected_confidences[i],
                    'selected_method': selected_methods[i],
                    'correct': bool(correctness[i])
                })

            # Save results
            with open(os.path.join(selection_dir, "results.json"), 'w') as f:
                json.dump(selection_results, f, indent=4)

            # Add to results for comparison
            all_selection_results.append(selection_results)

        # Find best aggregation method for selection
        best_selection_agg = min(all_selection_results, key=lambda x: x['ece'])['aggregation_method']

        # Add the best selection result to overall results for comparison
        best_selection_result = next(r for r in all_selection_results if r['aggregation_method'] == best_selection_agg)
        all_evaluation_results.append(best_selection_result)

        # Save best selection method
        with open(os.path.join(calib_dir, 'selection_method.json'), 'w') as f:
            json.dump({
                'selection_method': args.prediction_source,
                'best_aggregation_method': best_selection_agg
            }, f, indent=4)

        # Create updated comparison with selection
        plot_confidence_comparison(
            all_evaluation_results,
            save_path=os.path.join(args.output_dir, "comparison_with_selection.png")
        )

    logger.info("Confidence evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Confidence Evaluation for Telugu HWR")

    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root directory for dataset')
    parser.add_argument('--val_file', type=str, required=True,
                      help='Validation annotation file')
    parser.add_argument('--test_file', type=str, required=True,
                      help='Test annotation file')
    parser.add_argument('--vocab_file', type=str, required=True,
                      help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str, default='./output/confidence_evaluation_v2',
                      help='Output directory for saving results')
    parser.add_argument('--val_samples', type=int, default=None,
                      help='Maximum validation samples to use (None for all)')
    parser.add_argument('--test_samples', type=int, default=None,
                      help='Maximum test samples to use (None for all)')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--img_height', type=int, default=64,
                      help='Input image height')
    parser.add_argument('--img_width', type=int, default=256,
                      help='Input image width')
    parser.add_argument('--hidden_size', type=int, default=256,
                      help='RNN hidden size')
    parser.add_argument('--mc_dropout', action='store_true',
                      help='Use MC Dropout model for evaluation')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                      help='Dropout rate for MC Dropout model')

    # Confidence parameters
    parser.add_argument('--num_samples', type=int, default=30,
                      help='Number of samples for MC Dropout')
    parser.add_argument('--prediction_source', type=str, default='none',
                      choices=['none', 'best_confidence', 'ensemble', 'Uncalibrated',
                              'Temperature Scaling', 'MC Dropout', 'Step Dependent T-Scaling'],
                      help='Method for selecting predictions')
    parser.add_argument('--agg_method', type=str, default='geometric_mean',
                      choices=['geometric_mean', 'product', 'min'],
                      help='Default aggregation method for confidence scores')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--cuda', action='store_true',
                      help='Use CUDA if available')

    args = parser.parse_args()
    main(args)
