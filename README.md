# Telugu Handwriting Recognition with Confidence Calibration

This repository contains an end-to-end system for Telugu handwritten word recognition with confidence calibration. The system supports multiple model architectures (CRNN and PARSeq) and various confidence estimation techniques. It includes a web interface for interactive demonstrations.

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
│   └── crnn.py                # CRNN model
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
│   └── visualization.py       # Result visualization utilities
│
├── trainers/                  # Model trainers
│   ├── base_trainer.py        # Base trainer
│   └── crnn_trainer.py        # CRNN model trainer
│
├── scripts/                   # Training and evaluation scripts
│   ├── train_crnn.py          # Train CRNN model
│   ├── evaluate.py            # Evaluate models
│   └── evaluate_confidence.py # Confidence evaluation
│
├── webapp/                    # Web application for demonstration
│   ├── static/                # Static assets (JS, CSS, images)
│   ├── templates/             # HTML templates
│   ├── utils/                 # Webapp utilities
│   ├── app.py                 # Flask web server
│   └── model_manager.py       # Model management for webapp
│
├── run_webapp.py              # Script to run the web application
├── requirements.txt           # Project dependencies
├── requirements_webapp.txt    # Webapp-specific dependencies
└── README.md                  # Project documentation
```

## Key Features

- **Multiple Model Architectures**:
  - CRNN with CTC loss for sequence recognition

- **Advanced Confidence Calibration**:
  - Temperature Scaling for basic calibration
  - Monte Carlo Dropout for uncertainty estimation (epistemic and aleatoric)
  - Step-Dependent Temperature Scaling for position-specific calibration
  - Length-normalized confidence scores for better sequence confidence

- **Confidence Aggregation Methods**:
  - Geometric Mean (Length-normalized)
  - Product (Unnormalized)
  - Minimum (Pessimistic)

- **Comprehensive Evaluation**:
  - Character Error Rate (CER) and Word Error Rate (WER) metrics
  - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
  - Brier Score for probabilistic assessment
  - Visualization tools for calibration analysis

- **Interactive Web Application**:
  - Upload or capture handwritten text images
  - Real-time recognition with confidence visualization
  - Selection of different confidence calibration methods
  - Character-level confidence heatmaps

## Setup

1. Clone the repository:
```bash
git clone git@github.com:might-guy106/telugu-HWR-with-confidence-calibration.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install webapp-specific dependencies:
```bash
pip install -r requirements_webapp.txt
```

## Data Preparation

The system expects data in the following format:

- A text file with lines in the format: `image_path label`
- Images should be in the directory specified by `data_root`

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

## Training

### CRNN Model

```bash
python scripts/train_crnn.py \
    --data_root "/home/pankaj/Desktop/698r project/my implementations/telugu-hwr/datasets/telugu datset" \
    --train_file train.txt \
    --val_file val.txt \
    --vocab_file output/vocabulary.txt \
    --output_dir output2/crnn \
    --img_height 64 \
    --img_width 256 \
    --batch_size 32 \
    --epochs 20 \
    --learning_rate 0.001 \
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

### Confidence Calibration Evaluation

```bash
python scripts/evaluate_confidence.py \
    --data_root "/home/pankaj/Desktop/698r project/my implementations/telugu-hwr/datasets/telugu datset" \
    --val_file val.txt \
    --test_file test.txt \
    --vocab_file output/vocabulary.txt \
    --model_path output/crnn/best_cer_model.pth \
    --output_dir output/confidence_evaluation_final \
    --batch_size 16 \
    --num_samples 30 \
    --mc_dropout \
    --agg_method min \
    --cuda
```

## Results and Visualizations

After running the evaluation scripts, you can find various visualizations in the specified output directory:

- **Reliability Diagrams**: Show the relationship between confidence and accuracy
- **Temperature vs. ECE Curves**: For temperature scaling optimization
- **Position-specific Temperature Values**: For Step-Dependent Temperature Scaling
- **Calibration Comparison**: Comparing different calibration methods

## Web Application

The project includes a web application for interactive demonstrations:

```bash
python run_webapp.py
```

### Web Interface Features:

1. **Image Upload**: Upload Telugu handwritten images or use sample images
2. **Camera Capture**: Capture images directly from your device's camera
3. **Recognition Settings**: 
   - Choose different confidence estimation methods:
     - Step-Dependent Temperature Scaling
     - Temperature Scaling
     - Monte Carlo Dropout
     - Uncalibrated (raw softmax probabilities)
   - Select confidence aggregation methods:
     - Minimum
     - Geometric Mean
     - Product
4. **Results Display**:
   - Recognized Telugu text
   - Overall confidence score with color-coded indicator
   - Character-level confidence heatmap
   - Method-specific details for the selected confidence estimation approach

## Acknowledgements

- This project was developed as part of research on handwriting recognition with confidence calibration.

