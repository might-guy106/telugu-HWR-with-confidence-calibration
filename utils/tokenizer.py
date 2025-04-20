import torch
import numpy as np

class PARSeqTokenizer:
    """
    Tokenizer for PARSeq model.

    Args:
        vocab: Character vocabulary
        max_length: Maximum sequence length
    """

    def __init__(self, vocab: str, max_length: int = 25):
        self.vocab = vocab
        self.max_length = max_length

        # Create mappings
        self.char_to_idx = {char: idx + 3 for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx + 3: char for idx, char in enumerate(vocab)}

        # Special tokens
        self.pad_id = 0
        self.sos_id = 1  # Start of sequence
        self.eos_id = 2  # End of sequence

        # Add special tokens to mapping
        self.idx_to_char[self.pad_id] = ""
        self.idx_to_char[self.sos_id] = "<s>"
        self.idx_to_char[self.eos_id] = "</s>"

    def encode(self, batch_texts: list[str]) -> tuple[np.ndarray, list[int]]:
        """
        Encode a batch of texts to token indices with debug printing.
        """
        batch_size = len(batch_texts)
        seq_lengths = [min(len(text), self.max_length) for text in batch_texts]
        max_seq_len = max(seq_lengths)

        # Create tensor with padding
        encoded = np.ones((batch_size, max_seq_len + 2), dtype=np.int64) * self.pad_id

        # Add SOS token at the beginning
        encoded[:, 0] = self.sos_id

        # Encode each text
        for i, text in enumerate(batch_texts):
            # Debug the first few examples
            if i < 2:  # Print first 2 examples
                print(f"Encoding text: '{text}'")

            # Limit to max_length
            text = text[:self.max_length]

            # Convert characters to indices
            char_indices = []
            for char in text:
                if char in self.char_to_idx:
                    char_indices.append(self.char_to_idx[char])
                else:
                    print(f"Warning: Character '{char}' not in vocabulary, skipping")

            # Add to encoded tensor
            encoded[i, 1:len(char_indices)+1] = char_indices

            # Add EOS token
            encoded[i, len(char_indices)+1] = self.eos_id

            # Debug the first few examples
            if i < 2:  # Print first 2 examples
                print(f"Encoded indices: {encoded[i]}")

        return encoded, seq_lengths

    def decode(self, token_indices: torch.Tensor) -> list[str]:
        """
        Decode token indices to texts with better error handling.
        """
        texts = []

        # Handle both tensor and numpy array inputs
        if isinstance(token_indices, torch.Tensor):
            token_indices = token_indices.cpu().numpy()

        # Handle both 1D and 2D inputs
        if token_indices.ndim == 0:
            # Single integer - wrap in list
            token_indices = np.array([[token_indices]])
        elif token_indices.ndim == 1:
            # 1D array - add batch dimension
            token_indices = token_indices.reshape(1, -1)

        for indices in token_indices:
            chars = []

            # Ensure indices is iterable
            if not hasattr(indices, '__iter__'):
                indices = [indices]

            for idx in indices:
                # Stop at EOS token
                if idx == self.eos_id:
                    break
                # Skip special tokens
                if idx not in (self.pad_id, self.sos_id) and idx in self.idx_to_char:
                    chars.append(self.idx_to_char[idx])

            decoded_text = "".join(chars)
            texts.append(decoded_text)

        return texts
