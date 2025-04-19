import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM module for sequence modeling.

    Args:
        input_size (int): Input feature size
        hidden_size (int): LSTM hidden layer size
        output_size (int): Output feature size
        dropout (float): Dropout probability (default: 0.0)
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(BidirectionalLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, sequence_length, output_size)
        """
        outputs, _ = self.lstm(x)
        outputs = self.dropout(outputs)
        outputs = self.linear(outputs)
        return outputs


class BidirectionalLSTMWithDropout(BidirectionalLSTM):
    """
    Enhanced Bidirectional LSTM with dropout that can be enabled during inference
    for Monte Carlo uncertainty estimation.

    Args:
        input_size (int): Input feature size
        hidden_size (int): LSTM hidden layer size
        output_size (int): Output feature size
        dropout (float): Dropout probability (default: 0.2)
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(BidirectionalLSTMWithDropout, self).__init__(
            input_size, hidden_size, output_size, dropout
        )
        self.dropout_rate = dropout

    def forward(self, x, apply_dropout=False):
        """
        Forward pass with optional dropout.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            apply_dropout (bool): Whether to apply dropout even during inference

        Returns:
            Output tensor of shape (batch_size, sequence_length, output_size)
        """
        outputs, _ = self.lstm(x)

        # Apply dropout either during training or when explicitly requested
        if self.training or apply_dropout:
            outputs = self.dropout(outputs)

        outputs = self.linear(outputs)
        return outputs
