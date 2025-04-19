def levenshtein_distance(s1, s2):
    """
    Calculate Levenshtein distance between two sequences.
    Works for both strings (character-level) and lists (word-level).

    Args:
        s1: First sequence (string or list)
        s2: Second sequence (string or list)

    Returns:
        Edit distance (int)
    """
    # Convert lists to tuple if needed (for hashable keys in memoization)
    if isinstance(s1, list):
        s1 = tuple(s1)
    if isinstance(s2, list):
        s2 = tuple(s2)

    # Create a 2D matrix to store results of subproblems
    rows = len(s1) + 1
    cols = len(s2) + 1

    # Initialize the matrix
    dp = [[0 for _ in range(cols)] for _ in range(rows)]

    # Fill the first row and column
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    # Fill the rest of the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],   # deletion
                                   dp[i][j-1],    # insertion
                                   dp[i-1][j-1])  # substitution

    # Return the final distance
    return dp[rows-1][cols-1]

def character_error_rate(pred_text, gt_text):
    """
    Calculate Character Error Rate (CER).

    Args:
        pred_text: Predicted text
        gt_text: Ground truth text

    Returns:
        CER value between 0 and 1
    """
    if not gt_text:  # Handle empty ground truth
        return 1.0 if pred_text else 0.0

    edit_distance = levenshtein_distance(pred_text, gt_text)
    return edit_distance / len(gt_text)

def word_error_rate(pred_text, gt_text):
    """
    Calculate Word Error Rate (WER).

    Args:
        pred_text: Predicted text
        gt_text: Ground truth text

    Returns:
        WER value between 0 and 1
    """
    pred_words = pred_text.split()
    gt_words = gt_text.split()

    if not gt_words:  # Handle empty ground truth
        return 1.0 if pred_words else 0.0

    edit_distance = levenshtein_distance(pred_words, gt_words)
    return edit_distance / len(gt_words)

def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics for a batch of predictions.

    Args:
        predictions: List of predicted text strings
        targets: List of ground truth text strings

    Returns:
        Dictionary containing accuracy, CER, WER, and other metrics
    """
    total_char_edits = 0
    total_chars = 0
    total_word_edits = 0
    total_words = 0
    correct_words = 0

    for pred, target in zip(predictions, targets):
        # Character-level metrics
        char_edits = levenshtein_distance(pred, target)
        total_char_edits += char_edits
        total_chars += len(target)

        # Word-level metrics
        pred_words = pred.split()
        target_words = target.split()
        word_edits = levenshtein_distance(pred_words, target_words)
        total_word_edits += word_edits
        total_words += len(target_words)

        # Exact word match
        if pred == target:
            correct_words += 1

    # Calculate final metrics
    cer = (total_char_edits / total_chars) * 100 if total_chars > 0 else 0
    wer = (total_word_edits / total_words) * 100 if total_words > 0 else 0
    accuracy = (correct_words / len(predictions)) * 100 if predictions else 0

    return {
        'accuracy': accuracy,
        'character_error_rate': cer,
        'word_error_rate': wer,
        'total_samples': len(predictions),
        'correct_words': correct_words
    }
