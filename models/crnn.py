import torch
import torch.nn as nn
from models.components.rnn import BidirectionalLSTM

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for Telugu handwriting recognition.

    This model combines a CNN for feature extraction with bidirectional LSTM
    layers for sequence modeling.

    Args:
        img_height (int): Height of input images
        num_channels (int): Number of input channels (1 for grayscale)
        num_classes (int): Number of output classes (vocabulary size + blank)
        rnn_hidden_size (int): Hidden size for RNN layers
    """
    def __init__(self, img_height, num_channels, num_classes, rnn_hidden_size=256):
        super(CRNN, self).__init__()

        # Make sure input height is compatible with the architecture
        assert img_height % 16 == 0, f"Image height must be divisible by 16, got {img_height}"

        self.img_height = img_height
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size

        # CNN layers for feature extraction
        self.cnn_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x H/2 x W/2

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x H/4 x W/4

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Output: 256 x H/8 x W/4

            # Layer 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Output: 512 x H/16 x W/4

            # Layer 5
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # Output: 512 x H/16-1 x W/4-1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Calculate output height of CNN
        self.cnn_output_height = img_height // 16 - 1

        # RNN layers (bidirectional LSTM)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512 * self.cnn_output_height, rnn_hidden_size, rnn_hidden_size, dropout=0.2),
            BidirectionalLSTM(rnn_hidden_size, rnn_hidden_size, num_classes, dropout=0.2)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (sequence_length, batch_size, num_classes)
        """
        # Apply CNN layers to extract features
        conv = self.cnn_layers(x)

        # Reshape for RNN: (batch_size, channels, height, width) -> (batch_size, width, channels*height)
        batch_size = conv.size(0)
        conv = conv.permute(0, 3, 1, 2)  # (batch_size, width, channels, height)
        conv = conv.reshape(batch_size, -1, 512 * self.cnn_output_height)  # (batch_size, width, channels*height)

        # Apply RNN layers
        output = self.rnn(conv)  # (batch_size, width, num_classes)

        # For CTC loss, we need (sequence_length, batch_size, num_classes)
        output = output.permute(1, 0, 2)

        return output
