# Telugu Handwriting Recognition with Confidence Calibration

This repository contains an end-to-end system for Telugu handwritten word recognition with confidence calibration. The system supports multiple model architectures (CRNN and PARSeq) and various confidence estimation techniques.

## Project Structure

```
telugu-hwr/
├── data/                      # Data handling and preprocessing
│   ├── dataset.py             # Dataset class
│   ├── transforms.py          # Image transformations
│   └── analysis.py            # Data analysis and visualization
│
├── models/                    # Model architectures
│   ├── components/            # Model building blocks
│   │   ├── rnn.py             # RNN modules
│   │   └── ...                # Other components
│   ├── crnn.py                # CRNN model
│   ├── mc_dropout_crnn.py     # CRNN with Monte Carlo dropout
│   └── parseq.py              # PARSeq model
│
├── confidence/                # Confidence estimation methods
│   ├── base.py                # Base confidence estimator
│   ├── calibration.py         # Calibration metrics
│   ├── temperature.py         # Temperature scaling
│   ├── mc_dropout.py          # Monte Carlo dropout
│   ├── combined.py            # Combined confidence estimator
│   └── uncalibrated.py        # Uncalibrated confidence
│
├── utils/                     # Utility functions
│   ├── ctc_decoder.py         # CTC decoder
│   ├── metrics.py             # Evaluation metrics
│   └── visualization.py       # Visualization utilities
│
├── trainers/                  # Model trainers
│   ├── base_trainer.py        # Base trainer
│   ├── crnn_trainer.py        # CRNN model trainer
│   └── parseq_trainer.py      # PARSeq model trainer
│
├── scripts/                   # Training and evaluation scripts
│   ├── train_crnn.py          # Train CRNN model
│   ├── train_parseq.py        # Train PARSeq model
│   ├── evaluate.py            # Evaluate models
│   └── evaluate_confidence.py # Evaluate confidence methods
│
└── configs/                   # Configuration files
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/telugu-hwr.git
cd telugu-hwr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The system expects data in the following format:

- A text file with lines in the format: `image_path label`
- Images should be in the directory specified by `data_root`

## Training

### CRNN Model

```bash
python scripts/train_crnn.py \
    --data_root "/home/pankaj/Desktop/698r project/my implementations/telugu-hwr/datasets/telugu datset" \
    --train_file train.txt \
    --val_file val.txt \
    --vocab_file ../output/vocabulary.txt \
    --output_dir ../output/crnn \
    --max_samples 100 \
    --val_samples 20 \
    --img_height 64 \
    --img_width 256 \
    --batch_size 32 \
    --epochs 30 \
    --learning_rate 0.001 \
    --cuda
```

Add `--mc_dropout` flag to train with Monte Carlo dropout for uncertainty estimation.

### PARSeq Model

```bash
python scripts/train_parseq.py \
    --data_root "/home/pankaj/Desktop/698r project/my implementations/telugu-hwr/datasets/telugu datset" \
    --train_file train.txt \
    --val_file val.txt \
    --vocab_file ../output/vocabulary.txt \
    --output_dir ../output/parseq \
    --max_samples 100 \
    --val_samples 20 \
    --img_height 32 \
    --img_width 128 \
    --max_length 35 \
    --num_permutations 6 \
    --batch_size 32 \
    --epochs 1 \
    --learning_rate 0.0007 \
    --cuda
```

## Evaluation

### Standard Evaluation

```bash
python scripts/evaluate.py \
    --data_root "/home/pankaj/Desktop/698r project/datasets/telugu datset" \
    --test_file test.txt \
    --vocab_file ../output/crnn/vocabulary.txt \
    --model_path ../output/crnn/best_cer_model.pth \
    --model_type crnn \
    --output_dir ../output/evaluation/crnn \
    --cuda
```

For PARSeq models, use `--model_type parseq`.

### Confidence Evaluation

```bash
python scripts/evaluate_confidence.py \
    --data_root "/home/pankaj/Desktop/698r project/datasets/telugu datset" \
    --val_file val.txt \
    --test_file test.txt \
    --vocab_file ../output/crnn/vocabulary.txt \
    --model_path ../output/crnn/best_cer_model.pth \
    --output_dir ../output/confidence_evaluation \
    --batch_size 32 \
    --dropout_rate 0.2 \
    --calib_level word \
    --cuda
```

## Acknowledgements

- This project was developed as part of research on handwriting recognition with confidence calibration.
- The PARSeq implementation is based on the paper "Scene Text Recognition with Permuted Autoregressive Sequence Models" by Bautista et al.
```

## Testing the Restructured Code

To test the restructured code, you can run the test script:

```python
# telugu-hwr/scripts/test_structure.py
# (Already implemented above)
```

Make sure to update the paths in the script to point to your data before running:

```bash
cd telugu-hwr
python scripts/test_structure.py
```

This will perform a small training run to ensure all components are working together properly.

## Conclusion

You have now completed the full restructuring of the Telugu Handwriting Recognition project. The new structure is modular, maintainable, and follows good software engineering practices:

1. Clear separation of concerns with modules for data, models, confidence estimation, training, and evaluation
2. Well-documented code with docstrings explaining purpose and parameters
3. Consistent interfaces between components
4. Command-line scripts for easy use
5. Comprehensive README with usage instructions

To fully complete the project, you might want to also:

1. Create a `requirements.txt` file listing all dependencies
2. Add more detailed documentation for specific components
3. Include example configurations for different scenarios
4. Add unit tests for critical components

This restructured project should be much easier to maintain, extend, and use for future research or applications.
