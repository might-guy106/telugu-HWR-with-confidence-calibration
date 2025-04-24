import torch
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TrOCRModel(torch.nn.Module):
    """
    TrOCR model for Telugu handwriting recognition.

    This model uses a Vision Transformer encoder and a Text Transformer decoder
    from the Hugging Face transformers library.

    Args:
        pretrained_model_name (str): Name or path of the pretrained model
        max_length (int): Maximum sequence length for generation
        is_fine_tuned (bool): Whether the model is fine-tuned for Telugu
    """

    def __init__(self, pretrained_model_name="microsoft/trocr-base-handwritten",
                 max_length=128, is_fine_tuned=False, device='cuda'):
        super(TrOCRModel, self).__init__()

        self.max_length = max_length
        self.device = device

        # Load model and processor
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name).to(device)
        self.processor = TrOCRProcessor.from_pretrained(pretrained_model_name)

        # Configure model generation parameters
        self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.processor.tokenizer.vocab_size
        self.model.config.max_length = max_length
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4

        # Update encoder config to handle different image formats
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'encoder'):
            if hasattr(self.model.config.encoder, 'num_channels'):
                # We keep the 3 channels for the encoder but preprocess our grayscale images
                # to match this expected input format
                print(f"TrOCR encoder expects {self.model.config.encoder.num_channels} channels")

        # Configure for Telugu-specific requirements if fine-tuned
        if is_fine_tuned:
            # Telugu-specific configurations would be applied here
            pass

    def forward(self, images, labels=None):
        """
        Forward pass for training or inference.

        Args:
            images: Input images tensor [batch_size, C, H, W]
            labels: Optional text labels for training

        Returns:
            Dictionary containing model outputs
        """
        # Process images
        pixel_values = self._preprocess_images(images)

        # Prepare labels for decoder input if provided
        if labels is not None:
            # Tokenize the labels
            tokenized_labels = self.processor.tokenizer(
                labels,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)

            # Get outputs from the model
            outputs = self.model(
                pixel_values=pixel_values,
                decoder_input_ids=tokenized_labels,
                labels=tokenized_labels,
                return_dict=True
            )
        else:
            # For inference without labels
            # We need empty decoder_input_ids
            batch_size = pixel_values.shape[0]
            decoder_input_ids = torch.ones(
                (batch_size, 1),
                dtype=torch.long,
                device=self.device
            ) * self.model.config.decoder_start_token_id

            outputs = self.model(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )

        return outputs

    def generate(self, images):
        """
        Generate text predictions with confidence scores.

        Args:
            images: Input images tensor [batch_size, C, H, W]

        Returns:
            Dictionary with predictions and confidence information
        """
        # Process images
        pixel_values = self._preprocess_images(images)

        # Generate text with scores for confidence estimation
        outputs = self.model.generate(
            pixel_values,
            max_length=self.max_length,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=4
        )

        # Decode predictions
        predicted_ids = outputs.sequences
        predicted_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # Calculate token-level confidences
        token_confidences = self._calculate_token_confidences(outputs)

        return {
            'logits': outputs.scores if hasattr(outputs, 'scores') else None,
            'predicted_ids': predicted_ids,
            'predicted_text': predicted_text,
            'token_confidences': token_confidences
        }

    def _preprocess_images(self, images):
        """
        Convert input tensor to TrOCR's expected format.

        Args:
            images: Input tensor [batch_size, C, H, W]

        Returns:
            Preprocessed pixel_values tensor
        """
        if isinstance(images, np.ndarray):
            # Handle numpy array input (from PIL images)
            # Convert grayscale to RGB if needed
            if len(images.shape) == 3 and images.shape[2] == 1:  # [H, W, 1]
                images = np.repeat(images, 3, axis=2)  # Convert to RGB
            elif len(images.shape) == 2:  # [H, W]
                images = np.stack([images, images, images], axis=2)  # Convert to RGB

            return self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)

        elif isinstance(images, torch.Tensor):
            # Handle tensor input
            if images.dim() == 4:  # [batch_size, C, H, W]
                # Convert to numpy and ensure RGB format
                images_np = []
                for i in range(images.size(0)):
                    # Convert to numpy array
                    if images.size(1) == 1:  # Grayscale
                        # Convert to [H, W, 1]
                        img = images[i].permute(1, 2, 0).cpu().numpy()
                        # Convert to RGB by repeating channel
                        img = np.repeat(img, 3, axis=2)
                    else:
                        # Already RGB
                        img = images[i].permute(1, 2, 0).cpu().numpy()

                    # Scale to 0-255 if needed
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)

                    images_np.append(img)

                # Process through the TrOCR processor
                pixel_values = self.processor(images=images_np, return_tensors="pt").pixel_values.to(self.device)
                return pixel_values
            else:
                raise ValueError(f"Unexpected tensor dimensions: {images.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(images)}")

    def _calculate_token_confidences(self, outputs):
        """
        Calculate token-level confidences from generation outputs.

        Args:
            outputs: Model generation outputs

        Returns:
            List of token confidence lists for each sample
        """
        token_confidences = []

        # Handle different output formats
        if hasattr(outputs, 'scores') and outputs.scores:
            # Get the scores (logits) for each generated token
            for i in range(len(outputs.sequences)):
                sequence = outputs.sequences[i]

                # Skip special tokens for confidence calculation
                special_tokens = {
                    self.processor.tokenizer.bos_token_id,
                    self.processor.tokenizer.eos_token_id,
                    self.processor.tokenizer.pad_token_id
                }

                # Track confidence for each token
                sequence_confidences = []

                # For each position in the sequence (skipping the first which is BOS)
                for t, token_id in enumerate(sequence[1:], 0):  # Skip BOS token
                    # Skip other special tokens
                    if token_id.item() in special_tokens:
                        continue

                    # Get the score for this position
                    if t < len(outputs.scores):
                        # Get the logits for this position
                        logits = outputs.scores[t][i]

                        # Apply softmax to get probabilities
                        probs = torch.nn.functional.softmax(logits, dim=-1)

                        # Get the probability of the chosen token
                        token_prob = probs[token_id].item()
                        sequence_confidences.append(token_prob)

                token_confidences.append(sequence_confidences)

        return token_confidences

    def save_pretrained(self, output_dir):
        """
        Save the model and processor to a directory.

        Args:
            output_dir: Directory to save the model to
        """
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_dir, device='cuda'):
        """
        Load a pretrained model from a directory.

        Args:
            model_dir: Directory containing the model
            device: Device to load the model onto

        Returns:
            Loaded TrOCRModel instance
        """
        model = cls(pretrained_model_name=model_dir, device=device)
        return model
