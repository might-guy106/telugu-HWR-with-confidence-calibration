import torch
from collections import defaultdict
from utils.ctc_alignment import decode_with_timestamps

class CTCLabelConverter:
    """
    Convert between text and CTC labels.

    This class handles encoding text strings to label indices for CTC loss
    and decoding model outputs back to text strings.

    Args:
        char_set: Set of characters in the vocabulary
        blank_idx: Index of the CTC blank character
    """
    def __init__(self, char_set, blank_idx=0):
        """
        Initialize the converter with a character set.

        Args:
            char_set: Set of characters in the vocabulary
            blank_idx: Index of the CTC blank character
        """
        self.blank_idx = blank_idx

        # Create character mappings
        self.char2idx = {char: idx + 1 for idx, char in enumerate(sorted(char_set))}  # +1 for blank token
        self.idx2char = {idx + 1: char for idx, char in enumerate(sorted(char_set))}
        self.idx2char[blank_idx] = ''  # blank token

        self.vocab_size = len(self.char2idx) + 1  # +1 for blank token

    def encode(self, text_batch):
        """
        Convert text strings to label indices for CTC loss.

        Args:
            text_batch: List of strings

        Returns:
            Tuple of (torch.IntTensor, torch.IntTensor):
                - Label indices (concatenated)
                - Label lengths
        """
        indices = []
        lengths = []

        for text in text_batch:
            # Check if text contains any valid characters
            valid_chars = [char for char in text if char in self.char2idx]

            if not valid_chars:
                # Handle empty or invalid text
                indices.append(self.blank_idx)  # Use blank as fallback
                lengths.append(1)
            else:
                # Normal encoding
                text_indices = [self.char2idx[char] for char in text if char in self.char2idx]
                indices.extend(text_indices)
                lengths.append(len(text_indices))

        return torch.IntTensor(indices), torch.IntTensor(lengths)

    def decode(self, output, output_lengths=None):
        """
        Decode model output to text with CTC decoding.

        Args:
            output: Tensor of shape [seq_len, batch_size, vocab_size]
            output_lengths: Optional lengths of output sequences

        Returns:
            List of decoded text strings
        """
        # Get the highest probability character at each timestep
        _, max_indices = torch.max(output, dim=2)

        # Convert to numpy for easier processing
        max_indices = max_indices.cpu().numpy()

        batch_size = max_indices.shape[1]
        decoded_text_batch = []

        for b in range(batch_size):
            sequence = max_indices[:, b]

            # If output lengths are provided, only take valid timesteps
            if output_lengths is not None and b < len(output_lengths):
                sequence = sequence[:output_lengths[b]]

            # CTC decoding: remove duplicates and blanks
            result = []
            prev_idx = -1

            for idx in sequence:
                if idx != self.blank_idx and idx != prev_idx:  # If not blank and not a repeat
                    result.append(self.idx2char.get(idx, ''))
                prev_idx = idx if idx != self.blank_idx else prev_idx

            decoded_text_batch.append(''.join(result))

        return decoded_text_batch

    def decode_beam_search(self, log_probs, beam_width=5):
        """
        Decode using beam search to find better transcriptions.

        Args:
            log_probs: Log probabilities from model [seq_len, batch_size, vocab_size]
            beam_width: Size of the beam

        Returns:
            List of decoded text strings
        """
        # Currently only supports batch size of 1
        batch_size = log_probs.shape[1]
        decoded_results = []

        for batch_idx in range(batch_size):
            sequence = log_probs[:, batch_idx, :]
            sequence = sequence.cpu().numpy()

            # Initialize beam with empty string
            beams = [([], 0)]  # (prefix, accumulated log probability)

            for t in range(sequence.shape[0]):
                new_beams = []

                for prefix, score in beams:
                    for c in range(sequence.shape[1]):
                        new_score = score + sequence[t, c]
                        new_prefix = prefix + [c]
                        new_beams.append((new_prefix, new_score))

                # Sort and keep top beam_width beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]

            # CTC decoding for the best beam
            best_beam = beams[0][0]
            decoded = self._ctc_decode_list(best_beam)
            decoded_results.append(decoded)

        return decoded_results

    def _ctc_decode_list(self, sequence):
        """
        Helper method to decode a sequence of indices with CTC rules.

        Args:
            sequence: List of character indices

        Returns:
            Decoded text string
        """
        result = []
        prev_idx = -1

        for idx in sequence:
            if idx != self.blank_idx and idx != prev_idx:
                result.append(self.idx2char.get(idx, ''))
            prev_idx = idx if idx != self.blank_idx else prev_idx

        return ''.join(result)

    def decode_with_confidence(self, output):
        """
        Decode model output to text with character-level confidences and positions.

        Args:
            output: Tensor of shape [seq_len, batch_size, vocab_size]

        Returns:
            List of tuples (decoded_text, character_confidences, character_positions)
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=2)

        # Decode with timestamps
        return decode_with_timestamps(probs, self)
