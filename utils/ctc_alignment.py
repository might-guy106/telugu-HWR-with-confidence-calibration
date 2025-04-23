import torch
import numpy as np

def decode_with_timestamps(probs, converter):
    """
    Decode CTC output and track which timesteps produce each character,
    using the minimum confidence when characters repeat.

    Args:
        probs: Probability tensor of shape [seq_len, batch_size, vocab_size]
        converter: CTCLabelConverter instance

    Returns:
        List of tuples (decoded_text, character_confidences, character_positions)
    """
    # Get the highest probability character at each timestep
    max_probs, max_indices = torch.max(probs, dim=2)

    # Convert to numpy for easier processing
    max_indices = max_indices.cpu().numpy()
    max_probs = max_probs.cpu().numpy()

    batch_size = max_indices.shape[1]
    results = []

    for b in range(batch_size):
        sequence = max_indices[:, b]
        prob_sequence = max_probs[:, b]

        # CTC decoding with tracking of repeated characters
        decoded_text = []
        confidences = []
        timesteps = []  # Track which timestep produced each character

        # For tracking repeated characters
        current_char = None
        current_char_positions = []
        current_char_probs = []

        # Process the sequence
        for t, (idx, prob) in enumerate(zip(sequence, prob_sequence)):
            # Skip blank labels
            if idx == converter.blank_idx:
                continue

            # Get character for this index
            char = converter.idx2char.get(idx, '')

            # If character changes, save the previous one
            if current_char is not None and char != current_char:
                # Find minimum probability among all occurrences
                min_prob = min(current_char_probs)
                # Find the position of that minimum (first if multiple)
                min_prob_pos = current_char_positions[current_char_probs.index(min_prob)]

                decoded_text.append(current_char)
                confidences.append(min_prob)
                timesteps.append(min_prob_pos)

                # Reset for the new character
                current_char = char
                current_char_positions = [t]
                current_char_probs = [prob]
            # If this is the same character or first character
            else:
                if current_char is None:
                    current_char = char
                    current_char_positions = [t]
                    current_char_probs = [prob]
                else:
                    # Same character, just add to the current group
                    current_char_positions.append(t)
                    current_char_probs.append(prob)

        # Don't forget the last character
        if current_char is not None:
            min_prob = min(current_char_probs)
            min_prob_pos = current_char_positions[current_char_probs.index(min_prob)]

            decoded_text.append(current_char)
            confidences.append(min_prob)
            timesteps.append(min_prob_pos)

        results.append((''.join(decoded_text), confidences, timesteps))

    return results
