import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import argparse
import torch
import json
import logging
from torch.utils.data import DataLoader

from data.dataset import TeluguHWRDataset
from data.transforms import get_transforms
from models.mc_dropout_crnn import MCDropoutCRNN
from utils.ctc_decoder import CTCLabelConverter
from confidence.temperature import TemperatureScaler
from confidence.mc_dropout import MCDropoutConfidence
from confidence.combined import CombinedConfidenceEstimator
from confidence.uncalibrated import UncalibratedConfidence
from confidence.calibration import plot_reliability_diagram, brier_score

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

def evaluate_confidence(model, data_loader, converter, device, confidence_estimator, save_dir):
    """
    Evaluate confidence estimation methods.

    Args:
        model: Model to evaluate
        data_loader: Test data loader
        converter: Label converter
        device: Device to use
        confidence_estimator: Confidence estimator
        save_dir: Directory to save results

    Returns:
        Dictionary with evaluation results
    """
    # Lists to store results
    all_predictions = []
    all_ground_truth = []
    all_correctness = []
    all_uncalib_conf = []  # Uncalibrated confidence
    all_temp_conf = []
    all_mc_conf = []
    all_combined_conf = []
    all_epistemic_unc = []
    all_aleatoric_unc = []

    logger.info("Evaluating all confidence methods...")

    # Create uncalibrated confidence estimator
    uncalib_estimator = UncalibratedConfidence(model, converter, device)

    # Process each batch
    batch_count = 0
    for images, labels in data_loader:
        batch_count += 1
        if batch_count % 10 == 0:
            logger.info(f"Processing batch {batch_count}/{len(data_loader)}")

        images = images.to(device)

        # Get uncalibrated confidence first
        uncalib_results = uncalib_estimator.get_confidence(images)
        uncalib_predictions = uncalib_results['predictions']
        all_uncalib_conf.extend(uncalib_results['confidences'])

        # Get calibrated confidence methods
        calib_results = confidence_estimator.get_confidence(images)
        calib_predictions = calib_results['predictions']

        # Store predictions and ground truth
        all_predictions.extend(calib_predictions)  # Use the calibrated predictions for official results
        all_ground_truth.extend(labels)

        # Calculate correctness for both methods
        # We need to store correctness for BOTH uncalibrated and calibrated predictions
        batch_correctness = [1 if p == gt else 0 for p, gt in zip(calib_predictions, labels)]
        uncalib_correctness = [1 if p == gt else 0 for p, gt in zip(uncalib_predictions, labels)]

        all_correctness.extend(batch_correctness)

        # Store confidence values
        all_temp_conf.extend(calib_results['temperature_confidences'])
        all_mc_conf.extend(calib_results['mc_confidences'])
        all_combined_conf.extend(calib_results['combined_confidences'])

        # Store uncertainty values
        all_epistemic_unc.extend(calib_results['epistemic_uncertainties'])
        all_aleatoric_unc.extend(calib_results['aleatoric_uncertainties'])

    # Calculate accuracy
    accuracy = sum(all_correctness) / len(all_correctness) * 100
    logger.info(f"Model accuracy on test set: {accuracy:.2f}%")

    # Evaluate calibration for uncalibrated model
    logger.info("Evaluating Uncalibrated model calibration...")
    uncalib_ece, uncalib_mce = plot_reliability_diagram(
        all_uncalib_conf,
        all_correctness,  # Using the calibrated model's correctness for consistency
        title="Uncalibrated Model",
        save_path=os.path.join(save_dir, "uncalibrated_calibration.png")
    )
    uncalib_brier = brier_score(all_uncalib_conf, all_correctness)

    # Evaluate calibration for each method
    logger.info("Evaluating Temperature Scaling calibration...")
    temp_ece, temp_mce = plot_reliability_diagram(
        all_temp_conf,
        all_correctness,
        title="Temperature Scaling Calibration",
        save_path=os.path.join(save_dir, "temp_scaling_calibration.png")
    )

    logger.info("Evaluating MC Dropout calibration...")
    mc_ece, mc_mce = plot_reliability_diagram(
        all_mc_conf,
        all_correctness,
        title="MC Dropout Calibration",
        save_path=os.path.join(save_dir, "mc_dropout_calibration.png")
    )

    logger.info("Evaluating Combined calibration...")
    combined_ece, combined_mce = plot_reliability_diagram(
        all_combined_conf,
        all_correctness,
        title="Combined Calibration",
        save_path=os.path.join(save_dir, "combined_calibration.png")
    )

    # Calculate Brier scores
    temp_brier = brier_score(all_temp_conf, all_correctness)
    mc_brier = brier_score(all_mc_conf, all_correctness)
    combined_brier = brier_score(all_combined_conf, all_correctness)

    # Compile and save results
    results = {
        "accuracy": accuracy,
        "temperature_scaling": {
            "ece": temp_ece,
            "mce": temp_mce,
            "brier_score": temp_brier
        },
        "mc_dropout": {
            "ece": mc_ece,
            "mce": mc_mce,
            "brier_score": mc_brier
        },
        "combined": {
            "ece": combined_ece,
            "mce": combined_mce,
            "brier_score": combined_brier
        },
        "examples": [],
        "uncalibrated": {
            "ece": uncalib_ece,
            "mce": uncalib_mce,
            "brier_score": uncalib_brier
        }
    }

    # Add some example predictions with confidences
    for i in range(min(20, len(all_predictions))):
        results["examples"].append({
            "ground_truth": all_ground_truth[i],
            "prediction": all_predictions[i],
            "correct": bool(all_correctness[i]),
            "temp_confidence": all_temp_conf[i],
            "mc_confidence": all_mc_conf[i],
            "combined_confidence": all_combined_conf[i],
            "epistemic_uncertainty": all_epistemic_unc[i],
            "aleatoric_uncertainty": all_aleatoric_unc[i]
        })

    # Save results to JSON file
    logger.info(f"Saving evaluation results to {save_dir}")
    with open(os.path.join(save_dir, "confidence_evaluation.json"), 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"\nConfidence evaluation results:")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Uncalibrated - ECE: {uncalib_ece:.4f}, MCE: {uncalib_mce:.4f}, Brier: {uncalib_brier:.4f}")
    logger.info(f"Temperature Scaling - ECE: {temp_ece:.4f}, MCE: {temp_mce:.4f}, Brier: {temp_brier:.4f}")
    logger.info(f"MC Dropout - ECE: {mc_ece:.4f}, MCE: {mc_mce:.4f}, Brier: {mc_brier:.4f}")
    logger.info(f"Combined - ECE: {combined_ece:.4f}, MCE: {combined_mce:.4f}, Brier: {combined_brier:.4f}")

    return results

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

    # Create model with MC Dropout support
    logger.info("Creating model with MC Dropout support...")
    model = MCDropoutCRNN(
        img_height=args.img_height,
        num_channels=1,
        num_classes=converter.vocab_size,
        rnn_hidden_size=args.hidden_size,
        dropout_rate=args.dropout_rate
    )

    # Load model weights
    logger.info(f"Loading model weights from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # Direct state dict
    model = model.to(device)

    if 'epoch' in checkpoint:
        logger.info(f"Loaded model trained for {checkpoint['epoch'] + 1} epochs")

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

    # Create confidence estimator
    logger.info(f"Creating combined confidence estimator (level: {args.calib_level})...")
    confidence_estimator = CombinedConfidenceEstimator(
        model, converter, device, level=args.calib_level
    )

    # Calibrate confidence estimators using validation data
    logger.info("Calibrating confidence estimators...")
    confidence_estimator.calibrate(val_loader)
    logger.info("Calibration completed")

    # Set weights for combined estimator (optional)
    if args.temp_weight is not None and args.mc_weight is not None:
        logger.info(f"Setting estimator weights: temp={args.temp_weight}, mc={args.mc_weight}")
        confidence_estimator.set_weights(args.temp_weight, args.mc_weight)

    # Evaluate confidence on test data
    logger.info("Evaluating confidence methods on test data...")
    results = evaluate_confidence(
        model=model,
        data_loader=test_loader,
        converter=converter,
        device=device,
        confidence_estimator=confidence_estimator,
        save_dir=args.output_dir
    )

    logger.info(f"Confidence evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Confidence Estimation for Telugu Handwriting Recognition")

    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for dataset')
    parser.add_argument('--val_file', type=str, required=True,
                        help='Validation annotation file')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test annotation file')
    parser.add_argument('--vocab_file', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str, default='./confidence_evaluation',
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
                        help='LSTM hidden size')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate for MC Dropout')

    # Confidence estimation parameters
    parser.add_argument('--temp_weight', type=float, default=None,
                        help='Weight for temperature scaling in combined estimator')
    parser.add_argument('--mc_weight', type=float, default=None,
                        help='Weight for MC dropout in combined estimator')
    parser.add_argument('--calib_level', type=str, choices=['word', 'char'], default='word',
                        help='Calibration level (word or character)')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')

    args = parser.parse_args()
    main(args)
