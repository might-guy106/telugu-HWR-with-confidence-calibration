import torch
import numpy as np
import torch.nn.functional as F
from abc import ABC, abstractmethod


class ConfidenceEstimator(ABC):
    """
    Base class for confidence estimators.

    This class defines the interface that all confidence estimators must implement.
    It provides common functionality and ensures consistent behavior across different
    confidence estimation methods.

    Args:
        model: The model used for predictions
        device: Device to run inference on (cuda/cpu)
        converter: Converter for decoding predictions
        normalize_confidence: Whether to normalize confidence scores
    """
    def __init__(self, model, device, converter, normalize_confidence=True):
        self.model = model
        self.device = device
        self.converter = converter
        self.calibrated = False
        self.name = "BaseEstimator"
        self.normalize_confidence = normalize_confidence

    @abstractmethod
    def calibrate(self, val_loader):
        """
        Calibrate the confidence estimator using validation data.

        Args:
            val_loader: Validation data loader
        """
        pass

    @abstractmethod
    def get_confidence(self, images):
        """
        Get confidence scores for predictions.

        Args:
            images: Input data (e.g., images)

        Returns:
            Dictionary with predictions and confidence scores
        """
        pass

    def predict(self, images):
        """
        Make predictions with confidence scores.

        Args:
            images: Input data (e.g., images)

        Returns:
            Dictionary with predictions and confidence scores
        """
        return self.get_confidence(images)

    def enable_dropout(self, m):
        """
        Enable dropout during inference for a model module.

        Args:
            m: Model module
        """
        if type(m) == torch.nn.Dropout or type(m) == torch.nn.Dropout2d:
            m.train()

    def _get_model_predictions(self, images, apply_dropout=False):
        """
        Get raw model predictions.

        Args:
            images: Input images
            apply_dropout: Whether to apply dropout during inference

        Returns:
            Model outputs and decoded predictions
        """
        # Move images to device
        images = images.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # If applying dropout, enable it for specific layers
        if apply_dropout:
            self.model.apply(self.enable_dropout)

        with torch.no_grad():
            # Forward pass
            outputs = self.model(images)

            # Decode predictions
            decoded = self.converter.decode(outputs)

        return outputs, decoded

    # Add a utility method to calculate confidence
    def calculate_confidence(self, seq_probs):
        """
        Calculate confidence based on sequence probabilities.

        Args:
            seq_probs: Sequence probabilities tensor

        Returns:
            Confidence score
        """
        if self.normalize_confidence:
            # Length-normalized confidence (geometric mean)
            log_probs = torch.log(seq_probs + 1e-10)
            avg_log_prob = torch.mean(log_probs)
            confidence = torch.exp(avg_log_prob).item()
        else:
            # Product of probabilities (original method)
            log_probs = torch.log(seq_probs + 1e-10)
            sum_log_prob = torch.sum(log_probs)
            confidence = torch.exp(sum_log_prob).item()

        return confidence
