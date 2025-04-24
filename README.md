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
├── confidence_v2/             # Enhanced confidence estimation methods
│   ├── base.py                # Base confidence estimator
│   ├── mc_dropout.py          # Monte Carlo dropout
│   ├── step_dependent_temperature_scaling.py # Position-specific calibration
│   ├── temperature_scaling.py # Temperature scaling
│   ├── uncalibrated.py        # Uncalibrated confidence
│   └── visualization.py       # Calibration visualization utilities
│
├── utils/                     # Utility functions
│   ├── ctc_decoder.py         # CTC decoder
│   ├── metrics.py             # Evaluation metrics
│   ├── tokenizer.py           # Tokenizer for PARSeq model
│   └── visualization.py       # Result visualization utilities
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
│   ├── evaluate_confidence_v2.py # Enhanced confidence evaluation
│   └── test_structure.py      # Test code structure functionality
│
└── requirements.txt           # Project dependencies
```

## Key Features

- **Multiple Model Architectures**:
  - CRNN with CTC loss for sequence recognition
  - PARSeq (Permuted Autoregressive Sequence) model with transformer architecture

- **Advanced Confidence Calibration**:
  - Temperature Scaling for basic calibration
  - Monte Carlo Dropout for uncertainty estimation
  - Step-Dependent Temperature Scaling for position-specific calibration
  - Length-normalized confidence scores (geometric mean) for better sequence confidence

- **Comprehensive Evaluation**:
  - Character Error Rate (CER) and Word Error Rate (WER) metrics
  - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
  - Brier Score for probabilistic assessment
  - Visualization tools for calibration analysis

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
    --vocab_file output/vocabulary.txt \
    --output_dir output/crnn \
    --img_height 64 \
    --img_width 256 \
    --batch_size 32 \
    --epochs 20 \
    --learning_rate 0.001 \
    --cuda
```

<!-- Add `--mc_dropout` flag to train with Monte Carlo dropout for uncertainty estimation. -->

### PARSeq Model

```bash
python scripts/train_parseq.py \
    --data_root "/home/pankaj/Desktop/698r project/my implementations/telugu-hwr/datasets/telugu datset" \
    --train_file train.txt \
    --val_file val.txt \
    --vocab_file output/vocabulary.txt \
    --output_dir output/parseq \
    --img_height 32 \
    --img_width 128 \
    --max_length 35 \
    --num_permutations 6 \
    --batch_size 32 \
    --epochs 30 \
    --learning_rate 0.0007 \
    --cuda
```

## Evaluation

### Standard Evaluation

```bash
python scripts/evaluate.py \
    --data_root "/home/pankaj/Desktop/698r project/my implementations/telugu-hwr/datasets/telugu datset" \
    --test_file test.txt \
    --vocab_file output/vocabulary.txt \
    --model_path output/crnn/best_cer_model.pth \
    --model_type crnn \
    --output_dir output/evaluation/crnn \
    --cuda
```

For PARSeq models, use `--model_type parseq`.

### Confidence Calibration Evaluation

```bash
python scripts/evaluate_confidence.py \
    --data_root "/home/pankaj/Desktop/698r project/my implementations/telugu-hwr/datasets/telugu datset" \
    --val_file val.txt \
    --test_file test.txt \
    --vocab_file output/vocabulary.txt \
    --model_path output/crnn/best_cer_model.pth \
    --output_dir output/confidence_evaluation_v8 \
    --batch_size 16 \
    --num_samples 30 \
    --val_samples 1000 \
    --test_samples 1000 \
    --agg_method min \
    --cuda
```

## Results and Visualizations

After running the evaluation scripts, you can find various visualizations in the specified output directory:

- **Reliability Diagrams**: Show the relationship between confidence and accuracy
- **Temperature vs. ECE Curves**: For temperature scaling optimization
- **Position-specific Temperature Values**: For Step-Dependent Temperature Scaling
- **Calibration Comparison**: Comparing different calibration methods

## Creating a Vocabulary File

If you don't have a vocabulary file, you can generate one from your dataset using the data analysis tools:

```bash
python -c "
from data.dataset import TeluguHWRDataset
from data.analysis import analyze_dataset
import os

data_root = '/path/to/dataset'
train_dataset = TeluguHWRDataset(
    data_file=os.path.join(data_root, 'train.txt'),
    root_dir=data_root,
    transform=None
)

vocab = analyze_dataset(train_dataset, save_dir='output')
print(f'Created vocabulary with {len(vocab)} characters')
"
```

## Acknowledgements

- This project was developed as part of research on handwriting recognition with confidence calibration.
- The PARSeq implementation is based on the paper "Scene Text Recognition with Permuted Autoregressive Sequence Models" by Bautista et al.
```


## Temporary

```bash
python scripts/train_crnn.py \
    --data_root "/home/GNN-NIDS/pankaj/telugu-hwr/datasets/telugu datset" \
    --train_file train.txt \
    --val_file val.txt \
    --vocab_file output/vocabulary.txt \
    --output_dir output/crnn \
    --img_height 64 \
    --img_width 256 \
    --batch_size 32 \
    --epochs 20 \
    --learning_rate 0.001 \
    --cuda
```
```bash
python scripts/train_trocr.py \
    --data_root "/home/GNN-NIDS/pankaj/telugu-hwr/datasets/telugu datset" \
    --train_file train.txt \
    --val_file val.txt \
    --output_dir output/trocr \
    --pretrained_model "microsoft/trocr-base-handwritten" \
    --img_height 384 \
    --img_width 384 \
    --batch_size 32 \
    --max_samples 1000 \
    --val_samples 200 \
    --epochs 10 \
    --learning_rate 5e-5 \
    --cuda
```

```bash
python scripts/evaluate.py \
    --data_root "/home/GNN-NIDS/pankaj/telugu-hwr/datasets/telugu datset" \
    --test_file test.txt \
    --vocab_file output/vocabulary.txt \
    --model_path output/crnn/best_cer_model.pth \
    --model_type crnn \
    --output_dir output/evaluation/crnn \
    --cuda
```

```bash
python scripts/evaluate_confidence.py \
    --data_root "/home/GNN-NIDS/pankaj/telugu-hwr/datasets/telugu datset" \
    --val_file val.txt \
    --test_file test.txt \
    --vocab_file output/vocabulary.txt \
    --model_path output/crnn/best_cer_model.pth \
    --output_dir output/confidence_evaluation_min \
    --batch_size 32 \
    --test_samples 1000 \
    --val_samples 1000 \
    --num_samples 30 \
    --agg_method min \
    --cuda
```
