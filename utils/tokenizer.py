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

    def encode(self, batch_texts):
        """Encode text to token indices, ensuring proper handling of special tokens."""
        batch_size = len(batch_texts)
        max_seq_len = max(len(text) for text in batch_texts)
        max_seq_len = min(max_seq_len, self.max_length)

        # Create tensor for the batch with padding
        encoded = torch.full((batch_size, max_seq_len + 2), self.pad_id, dtype=torch.long)

        # Add SOS token at the beginning
        encoded[:, 0] = self.sos_id

        # Encode each text
        for i, text in enumerate(batch_texts):
            text = text[:max_seq_len]  # Truncate if too long
            for j, char in enumerate(text):
                if char in self.char_to_idx:
                    encoded[i, j+1] = self.char_to_idx[char]

            # Add EOS token
            encoded[i, len(text)+1] = self.eos_id

        return encoded, [len(text)+2 for text in batch_texts]  # +2 for SOS and EOS

    def decode(self, token_indices):
        """Decode with NaN protection and better handling of special tokens."""
        texts = []

        # Handle tensor input
        if isinstance(token_indices, torch.Tensor):
            token_indices = token_indices.cpu().numpy()

        for seq in token_indices:
            # Skip initial SOS token
            if seq[0] == self.sos_id:
                seq = seq[1:]

            # Find first EOS token and truncate
            eos_pos = np.where(seq == self.eos_id)[0]
            if len(eos_pos) > 0:
                seq = seq[:eos_pos[0]]

            # Convert tokens to characters, skipping pad tokens
            chars = []
            for idx in seq:
                if idx != self.pad_id and idx in self.idx_to_char:
                    chars.append(self.idx_to_char[idx])

            texts.append(''.join(chars))

        return texts
