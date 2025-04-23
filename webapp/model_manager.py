import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Add project root to path
webapp_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(webapp_dir)
sys.path.insert(0, project_root)

from data.transforms import get_transforms
from models.crnn import CRNN
# from models.mc_dropout_crnn import MCDropoutCRNN
from utils.ctc_decoder import CTCLabelConverter
from confidence_v2.step_dependent_temperature_scaling import StepDependentTemperatureScaling
from confidence_v2.temperature_scaling import TemperatureScaling
from confidence_v2.mc_dropout import MCDropoutConfidence
from confidence_v2.uncalibrated import UncalibratedConfidence

class ModelManager:
    """
    Manages model loading, preprocessing, and inference for the web application.
    """
    def __init__(self):
        # Model paths and settings
        self.model_path = os.path.join(project_root, 'output', 'crnn', 'best_cer_model.pth')
        self.vocab_file = os.path.join(project_root, 'output', 'vocabulary.txt')
        self.img_height = 64
        self.img_width = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load vocabulary
        self.vocab = self._load_vocabulary()
        print(f"Loaded vocabulary with {len(self.vocab)} characters")

        # Create converter
        self.converter = CTCLabelConverter(self.vocab)

        # Load model
        self.model = self._load_model()

        # Create transforms
        self.transform = get_transforms('test', self.img_height, self.img_width)

        # Initialize confidence estimators
        self.confidence_estimators = {
            'uncalibrated': UncalibratedConfidence(self.model, self.device, self.converter),
            'temperature': TemperatureScaling(self.model, self.device, self.converter),
            'step_dependent': StepDependentTemperatureScaling(self.model, self.device, self.converter),
            'mc_dropout': MCDropoutConfidence(self.model, self.device, self.converter, num_samples=10),
        }

        # Temperatures for the calibrated models (these would come from calibration)
        # In a real implementation, you'd load these from saved calibration results
        self._init_calibration()

        print("Model manager initialized successfully")

    def _load_vocabulary(self):
        """Load vocabulary from file"""
        vocab = set()
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip()
                if char:  # Skip empty lines
                    vocab.add(char)
        return vocab

    def _load_model(self):
        """Load the CRNN model"""
        # Create model
        model = CRNN(
            img_height=self.img_height,
            num_channels=1,
            num_classes=self.converter.vocab_size,
            rnn_hidden_size=256,
            # dropout_rate=0.2
        )

        # Load weights
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(self.device)
            model.eval()
            print(f"Model loaded from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # For demonstration, we'll continue without a model
            # In production, you might want to handle this differently
            return model

    def _init_calibration(self):
        """Initialize calibration for confidence estimators"""
        # In a real implementation, you would load calibration parameters
        # from previously saved files

        # For demonstration, we'll set some example values
        if isinstance(self.confidence_estimators['temperature'], TemperatureScaling):
            # Set a fixed temperature value
            self.confidence_estimators['temperature'].temperature.data = torch.tensor([1.5], device=self.device)
            self.confidence_estimators['temperature'].calibrated = True

        if isinstance(self.confidence_estimators['step_dependent'], StepDependentTemperatureScaling):
            # Set fixed temperature values for each position
            temps = torch.ones(11, device=self.device) * 1.2
            # Make some positions cooler/hotter for demonstration
            temps[0] = 1.5  # First position usually needs higher temperature
            temps[1:5] = 1.3  # Middle positions
            temps[5:] = 1.1  # Later positions
            self.confidence_estimators['step_dependent'].temperatures.data = temps
            self.confidence_estimators['step_dependent'].calibrated = True

        # MC Dropout doesn't need calibration

    def preprocess_image(self, image):
        """Preprocess the image for model input"""
        # Ensure we're working with a PIL image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Apply transformations
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Basic transformation if transform not available
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
            image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def recognize(self, image, method='step_dependent'):
        """
        Recognize text in the image and calculate confidence

        Args:
            image: PIL image
            method: Confidence estimation method ('uncalibrated', 'temperature',
                   'step_dependent', or 'mc_dropout')

        Returns:
            Dictionary with recognition results and confidence info
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)

        # Select confidence estimator
        if method not in self.confidence_estimators:
            method = 'step_dependent'  # Default method

        estimator = self.confidence_estimators[method]

        # Get prediction with confidence
        result = estimator.get_confidence(image_tensor)

        # Extract basic information
        prediction = result['predictions'][0]  # Get first item (batch size 1)
        confidence = result['confidences'][0]

        # Get character-level confidence
        char_confidences = self._get_character_confidences(image_tensor, estimator, prediction)

        # Generate visualizations
        heatmap_b64 = self._generate_confidence_heatmap(prediction, char_confidences)

        # Format the result
        formatted_result = {
            'text': prediction,
            'overall_confidence': float(confidence),
            'character_confidences': [float(c) for c in char_confidences],
            'confidence_heatmap': heatmap_b64,
            'method': method
        }

        # Add method-specific data
        if method == 'step_dependent' and 'temperatures' in result:
            formatted_result['temperatures'] = result['temperatures'].tolist() if hasattr(result['temperatures'], 'tolist') else result['temperatures']
        elif method == 'temperature' and 'temperature' in result:
            formatted_result['temperature'] = float(result['temperature'])
        elif method == 'mc_dropout':
            # Add uncertainty information if available
            if 'epistemic_uncertainties' in result and 'aleatoric_uncertainties' in result:
                formatted_result['epistemic_uncertainty'] = float(result['epistemic_uncertainties'][0])
                formatted_result['aleatoric_uncertainty'] = float(result['aleatoric_uncertainties'][0])
            if 'prediction_distributions' in result:
                formatted_result['prediction_distribution'] = result['prediction_distributions'][0]

        return formatted_result

    def _get_character_confidences(self, image_tensor, estimator, prediction):
        """Extract character-level confidences using improved CTC alignment"""
        with torch.no_grad():
            # Get model output
            logits = self.model(image_tensor.to(self.device))

            # Apply temperature scaling if needed
            if hasattr(estimator, 'temperature'):
                scaled_logits = logits / estimator.temperature
            elif hasattr(estimator, 'temperatures'):
                # Apply step-dependent temperature scaling
                scaled_logits = torch.zeros_like(logits)
                for j in range(logits.size(0)):
                    temp_idx = min(j, estimator.tau)
                    scaled_logits[j] = logits[j] / estimator.temperatures[temp_idx]
            else:
                scaled_logits = logits

            # Get predictions with character-level confidences and positions
            decoded_results = self.converter.decode_with_confidence(scaled_logits)

            if len(decoded_results) > 0:
                decoded_text, char_confidences, char_positions = decoded_results[0]

                # If our decoded text matches the prediction, use its confidences
                if decoded_text == prediction:
                    return char_confidences

            # Fallback to the old method if there's a mismatch
            probs = torch.nn.functional.softmax(scaled_logits, dim=2)
            max_probs, _ = probs.max(2)
            max_probs = max_probs[:, 0].cpu().numpy()

            # Simple linear interpolation
            if len(prediction) > 0:
                confidence_values = np.interp(
                    np.linspace(0, len(max_probs) - 1, len(prediction)),
                    np.arange(len(max_probs)),
                    max_probs
                )
                return confidence_values.tolist()
            else:
                return []

    def _generate_confidence_heatmap(self, text, confidences):
        """Generate a heatmap visualization of character confidences"""
        if not text or not confidences:
            return None

        # Create figure
        plt.figure(figsize=(0.6 * len(text), 0.8))

        # Set a font that supports Telugu
        # Try to use either Noto Sans Telugu or a system font with Unicode support
        try:
            plt.rcParams['font.family'] = 'Noto Sans Telugu, DejaVu Sans, sans-serif'
        except:
            print("Warning: Could not set Telugu font for matplotlib")

        # Create data for heatmap
        data = np.array([confidences])

        # Create heatmap with characters as annotations
        chars = np.array([[c for c in text]])
        ax = sns.heatmap(data, cmap='RdYlGn', vmin=0, vmax=1,
                         annot=chars, fmt='',
                         cbar=False, linewidths=1, linecolor='black',
                         annot_kws={'fontsize': 12})

        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')

        # Save to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()

        # Convert to base64 for embedding in HTML
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')

        return f"data:image/png;base64,{img_str}"
