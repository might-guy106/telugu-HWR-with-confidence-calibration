class ConfidenceEstimator:
    """
    Base class for confidence estimators.

    This abstract class defines the interface for confidence estimation methods.

    Args:
        model: The model used for predictions
        device: Device to run inference on (cuda/cpu)
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.calibrated = False

    def calibrate(self, val_loader):
        """
        Calibrate the confidence estimator using validation data.

        Args:
            val_loader: Validation data loader
        """
        raise NotImplementedError

    def get_confidence(self, inputs):
        """
        Get confidence scores for predictions.

        Args:
            inputs: Input data (e.g., images)

        Returns:
            Dictionary with predictions and confidence scores
        """
        raise NotImplementedError
