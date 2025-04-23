import torch
import numpy as np

def decode_with_timestamps(probs, converter):
    """
    Decode CTC output and track which timesteps produce each character.

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

        # CTC decoding with timestep tracking
        decoded_text = []
        confidences = []
        timesteps = []  # Track which timestep produced each character

        prev_idx = -1
        for t, (idx, prob) in enumerate(zip(sequence, prob_sequence)):
            # Apply CTC rules (skip blanks and duplicates)
            if idx != converter.blank_idx and idx != prev_idx:
                char = converter.idx2char.get(idx, '')
                decoded_text.append(char)
                confidences.append(prob)
                timesteps.append(t)  # Record the timestep

            prev_idx = idx if idx != converter.blank_idx else prev_idx

        results.append((''.join(decoded_text), confidences, timesteps))

    return results
