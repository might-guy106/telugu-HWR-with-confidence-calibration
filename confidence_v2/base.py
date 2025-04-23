import torch
import numpy as np
import torch.nn.functional as F
from abc import ABC, abstractmethod


class ConfidenceEstimator(ABC):
    """
    Base class for confidence estimators.
    """
    def __init__(self, model, device, converter, agg_method='geometric_mean'):
        self.model = model
        self.device = device
        self.converter = converter
        self.calibrated = False
        self.name = "BaseEstimator"
        self.agg_method = agg_method

    @abstractmethod
    def calibrate(self, val_loader):
        """Calibrate the confidence estimator using validation data."""
        pass

    @abstractmethod
    def get_confidence(self, images, agg_method=None):
        """Get confidence scores for predictions."""
        pass

    def predict(self, images, agg_method=None):
        """Make predictions with confidence scores."""
        return self.get_confidence(images, agg_method)

    def enable_dropout(self, m):
        """Enable dropout during inference for a model module."""
        if type(m) == torch.nn.Dropout or type(m) == torch.nn.Dropout2d:
            m.train()

    def _get_model_predictions(self, images, apply_dropout=False):
        """Get raw model predictions."""
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

    def calculate_confidence(self, seq_probs, agg_method=None):
        """
        Calculate confidence based on sequence probabilities using one of three methods.

        Args:
            seq_probs: Sequence probabilities tensor
            agg_method: Aggregation method ('geometric_mean', 'product', or 'min')

        Returns:
            Confidence score
        """
        # Use provided method or default from instance
        method = agg_method or self.agg_method

        if method == 'geometric_mean':
            # Length-normalized confidence (geometric mean)
            log_probs = torch.log(seq_probs + 1e-10)
            avg_log_prob = torch.mean(log_probs)
            confidence = torch.exp(avg_log_prob).item()
        elif method == 'product':
            # Product of probabilities (unnormalized)
            log_probs = torch.log(seq_probs + 1e-10)
            sum_log_prob = torch.sum(log_probs)
            confidence = torch.exp(sum_log_prob).item()
        elif method == 'min':
            # Minimum probability (pessimistic approach)
            confidence = torch.min(seq_probs).item()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return confidence
