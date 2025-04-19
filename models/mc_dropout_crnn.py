import torch
import torch.nn as nn
from models.crnn import CRNN
from models.components.rnn import BidirectionalLSTMWithDropout

class MCDropoutCRNN(CRNN):
    """
    CRNN with Monte Carlo Dropout support for uncertainty estimation.

    This model extends the basic CRNN by enabling dropout during inference
    to estimate model uncertainty.

    Args:
        img_height (int): Height of input images
        num_channels (int): Number of input channels (1 for grayscale)
        num_classes (int): Number of output classes (vocabulary size + blank)
        rnn_hidden_size (int): Hidden size for RNN layers
        dropout_rate (float): Dropout probability
    """
    def __init__(self, img_height, num_channels, num_classes, rnn_hidden_size=256, dropout_rate=0.2):
        # Initialize with parent constructor
        super(MCDropoutCRNN, self).__init__(img_height, num_channels, num_classes, rnn_hidden_size)

        # Store dropout rate
        self.dropout_rate = dropout_rate

        # Replace RNN layers with dropout-enabled versions
        self.rnn = nn.Sequential(
            BidirectionalLSTMWithDropout(
                512 * self.cnn_output_height,
                rnn_hidden_size,
                rnn_hidden_size,
                dropout=dropout_rate
            ),
            BidirectionalLSTMWithDropout(
                rnn_hidden_size,
                rnn_hidden_size,
                num_classes,
                dropout=dropout_rate
            )
        )

    def forward(self, x, apply_dropout=False):
        """
        Forward pass with optional dropout enabled at inference time.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            apply_dropout: Whether to apply dropout at inference time

        Returns:
            Output tensor of shape (seq_length, batch_size, num_classes)
        """
        # Store original training state
        training_state = self.training

        # Enable dropout if requested (even during inference)
        if apply_dropout and not self.training:
            # Set model to train mode to enable dropout, but restore later
            self.train()

        try:
            # Apply CNN layers
            conv = self.cnn_layers(x)

            # Reshape for RNN: (batch_size, channels, height, width) -> (batch_size, width, channels*height)
            batch_size = conv.size(0)
            conv = conv.permute(0, 3, 1, 2)  # (batch_size, width, channels, height)
            conv = conv.reshape(batch_size, -1, 512 * self.cnn_output_height)

            # Apply RNN layers with optional dropout
            if isinstance(self.rnn[0], BidirectionalLSTMWithDropout):
                # Forward through first LSTM with dropout
                features = self.rnn[0](conv, apply_dropout)
                # Forward through second LSTM with dropout
                output = self.rnn[1](features, apply_dropout)
            else:
                # Standard forward pass through RNN layers
                output = self.rnn(conv)

            # For CTC loss, we need (sequence_length, batch_size, num_classes)
            output = output.permute(1, 0, 2)

            return output

        finally:
            # Restore original training state if it was modified
            if apply_dropout and not training_state:
                self.eval()
