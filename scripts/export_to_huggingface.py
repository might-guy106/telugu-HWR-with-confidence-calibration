import os
import sys
import json
import torch
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from models.crnn import CRNN
from utils.ctc_decoder import CTCLabelConverter

def export_to_huggingface(model_path, output_dir, vocab_file, img_height=64, hidden_size=256):
    """
    Export model to Hugging Face format.

    Args:
        model_path: Path to the trained model
        output_dir: Directory to save the model in Hugging Face format
        vocab_file: Path to vocabulary file
        img_height: Input image height
        hidden_size: RNN hidden size
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load vocabulary
    vocab = set()
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.strip()
            if char:
                vocab.add(char)

    # Create converter
    converter = CTCLabelConverter(vocab)

    # Load model
    model = CRNN(
        img_height=img_height,
        num_channels=1,
        num_classes=converter.vocab_size,
        rnn_hidden_size=hidden_size
    )

    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

    # Create config.json
    config = {
        'model_type': 'crnn',
        'img_height': img_height,
        'num_channels': 1,
        'num_classes': converter.vocab_size,
        'rnn_hidden_size': hidden_size,
        'vocab_size': len(vocab) + 1,  # +1 for blank token
        'architectures': ['CRNN'],
    }

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Save vocabulary
    with open(os.path.join(output_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(vocab)))

    print(f"Model exported to Hugging Face format in {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Export model to Hugging Face format')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save HF model')
    parser.add_argument('--vocab_file', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--img_height', type=int, default=64, help='Input image height')
    parser.add_argument('--hidden_size', type=int, default=256, help='RNN hidden size')

    args = parser.parse_args()

    export_to_huggingface(
        args.model_path,
        args.output_dir,
        args.vocab_file,
        args.img_height,
        args.hidden_size
    )
