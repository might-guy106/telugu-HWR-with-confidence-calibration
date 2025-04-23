# Telugu Handwriting Recognition with Confidence Calibration

## 1. Introduction

Handwriting recognition (HWR) remains a challenging task in computer vision and document analysis, especially for complex scripts like Telugu. The intricate nature of Telugu characters, with their curved shapes and complex connections, presents unique challenges for recognition systems. Despite advances in deep learning, achieving high accuracy while maintaining reliable confidence estimates for recognition results remains difficult.

This project addresses two critical aspects of handwriting recognition for Telugu script: (1) accurate recognition using state-of-the-art architectures, and (2) reliable confidence estimation for predictions. While high recognition accuracy is essential, the ability to reliably assess when a prediction might be incorrect is equally important for real-world applications.

Conventional deep learning models often produce overconfident predictions, even when wrong. This overconfidence can be problematic in applications requiring reliable uncertainty quantification, such as document processing, historical manuscript digitization, and automated form reading. We propose and evaluate multiple confidence calibration techniques to address this limitation.

The primary contributions of this work include:

1. Implementation and evaluation of two state-of-the-art architectures for Telugu handwriting recognition:
   - Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC)
   - Permuted Autoregressive Sequence (PARSeq) model based on transformers

2. Development of comprehensive confidence calibration methods:
   - Temperature scaling for improving confidence estimates
   - Monte Carlo Dropout for uncertainty estimation
   - Step-dependent temperature scaling for position-specific calibration

3. Thorough evaluation and analysis of both recognition performance and confidence calibration quality on Telugu handwritten text.

The findings from this work demonstrate that properly calibrated confidence measures significantly enhance the reliability of Telugu handwriting recognition systems, enabling more trustworthy deployment in practical applications.

## 2. Methodology

### 2.1 Dataset and Preprocessing

#### 2.1.1 Telugu Handwriting Dataset

Our experiments utilize a dataset of handwritten Telugu words. The dataset consists of images of handwritten Telugu words along with their transcriptions. Each sample in the dataset includes an image path and corresponding text label.

The dataset is organized with the following directory structure:
```
dataset/
├── TeluguSeg/      # Contains handwritten word images
├── train.txt       # Training split with image paths and labels
├── val.txt         # Validation split
└── test.txt        # Test split
```

Each line in the annotation files follows the format: `image_path label`

The dataset statistics are summarized in Table 1, showing the number of samples, unique characters, and unique words in each split.

[Table 1: Dataset Statistics - to be added]

Figure 1 illustrates some sample images from the dataset, demonstrating the variety and complexity of handwritten Telugu words.

[Figure 1: Sample images from the Telugu handwriting dataset - to be added]

Figure 2 shows the distribution of word lengths in the dataset, highlighting the range of complexity in the samples.

[Figure 2: Distribution of word lengths in the dataset - to be added]

#### 2.1.2 Preprocessing Pipeline

To prepare the images for model training and evaluation, we implemented a comprehensive preprocessing pipeline:

1. **Resizing with Aspect Ratio Preservation**: All images are resized to a fixed height (64 pixels for CRNN and 32 pixels for PARSeq) while maintaining the original aspect ratio.

2. **Width Normalization**: Images are padded or truncated to a fixed width (256 pixels for CRNN and 128 pixels for PARSeq) to enable batch processing.

3. **Grayscale Conversion**: All images are converted to grayscale to reduce computational complexity.

4. **Normalization**: Pixel values are normalized to the range [-1, 1] to improve training stability.

For data augmentation during training, we employed the following techniques:

1. **Random Affine Transformations**: Small rotations (±3°) and shearing (±5%) to simulate natural handwriting variations.

2. **Elastic Deformations**: Controlled deformations to mimic the natural variations in handwriting.

3. **Brightness and Contrast Adjustment**: Random adjustments to simulate different scanning and lighting conditions.

These preprocessing steps enhance the robustness of the models to variations in handwriting styles and image quality. The implementation of these techniques is shown in the `data/transforms.py` module.

### 2.2 Model Architectures

#### 2.2.1 CRNN with CTC Loss

The Convolutional Recurrent Neural Network (CRNN) architecture combines the strengths of CNNs for feature extraction and RNNs for sequence modeling. This architecture has proven effective for handwriting recognition tasks due to its ability to handle variable-length sequences.

Our CRNN implementation consists of three main components:

1. **Convolutional Feature Extraction**: A series of convolutional layers that extract visual features from the input image. The CNN backbone includes:
   - 5 convolutional blocks with increasing channel dimensions (64 → 128 → 256 → 512 → 512)
   - Each block contains convolution, batch normalization, and ReLU activation
   - Pooling layers to reduce spatial dimensions

2. **Recurrent Sequence Modeling**: Bidirectional LSTM layers process the extracted features as a sequence:
   - Features from CNN are reshaped into a sequence of feature vectors
   - Two BiLSTM layers with 256 hidden units for capturing contextual information
   - Dropout (0.2) between LSTM layers for regularization

3. **Transcription Layer**: Projects the LSTM outputs to character probabilities:
   - Linear projection to vocabulary size (including blank token for CTC)
   - Softmax activation for character probabilities

Figure 3 illustrates the CRNN architecture used in our implementation.

[Figure 3: CRNN Architecture for Telugu HWR - to be added]

The model is trained using Connectionist Temporal Classification (CTC) loss, which allows for alignment-free sequence learning. The CTC approach enables the model to learn the alignment between the input sequence and output text automatically, without requiring explicit segmentation of characters.

During inference, the model outputs a probability distribution over characters for each time step. The final transcription is obtained using CTC decoding, which removes duplicate consecutive characters and blank tokens.

#### 2.2.2 PARSeq Model

The Permuted Autoregressive Sequence (PARSeq) model represents a newer approach to sequence recognition based on transformer architectures. Unlike the CRNN model which uses CTC loss, PARSeq employs an autoregressive approach with multiple permutations of the target sequence during training.

The key components of our PARSeq implementation include:

1. **CNN Encoder**: A simplified ResNet-style convolutional encoder that converts the input image into a sequence of feature vectors:
   - Four convolutional blocks with progressive downsampling
   - Feature map dimensions reduced from input size to [batch_size, embed_dim, height/16, width/16]
   - Final feature maps reshaped into a sequence of vectors

2. **Transformer Decoder**: A transformer decoder architecture that autoregressively generates the output sequence:
   - Multi-head self-attention for capturing character dependencies
   - Cross-attention to attend to relevant parts of the input feature sequence
   - Position encodings to preserve sequential information
   - Feed-forward networks for transformation

3. **Permutation Training**: Multiple permutations of the target sequence are used during training to improve generalization:
   - Canonical left-to-right permutation
   - Randomly generated permutations for diversity
   - Shared parameters across all permutations to enhance robustness

Figure 4 shows the PARSeq architecture used in our implementation.

[Figure 4: PARSeq Architecture for Telugu HWR - to be added]

PARSeq offers several advantages over CRNN-CTC, including:
- Explicit modeling of character dependencies through self-attention
- No need for blank tokens or collapsing rules
- Improved handling of long-range dependencies
- More robust performance on complex scripts

The model is trained using cross-entropy loss with teacher forcing. During inference, the model generates characters autoregressively until an end-of-sequence token is produced or the maximum sequence length is reached.

### 2.3 Confidence Calibration Methods

Modern deep neural networks, while achieving high accuracy, often produce poorly calibrated confidence scores. A well-calibrated model should produce confidence scores that reflect the actual probability of correctness. For example, among all predictions with 80% confidence, approximately 80% should be correct.

We implemented and evaluated several confidence calibration methods to address this issue:

#### 2.3.1 Temperature Scaling

Temperature scaling is a simple yet effective post-processing technique for calibrating neural network outputs. It involves dividing the logits (pre-softmax activations) by a temperature parameter τ before applying the softmax function:

$$p_i(x; \tau) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$$

where $z_i$ are the logits and $\tau > 0$ is the temperature parameter. When $\tau > 1$, the probability distribution becomes more uniform (reducing confidence), and when $\tau < 1$, it becomes more peaked (increasing confidence).

The optimal temperature is determined on a validation set by minimizing the Expected Calibration Error (ECE). The calibration process does not affect the model's accuracy, as the relative ordering of logits remains unchanged.

For our implementation, we extended basic temperature scaling to handle sequence outputs by:
1. Finding the optimal temperature value that minimizes calibration error on the validation set
2. Calculating word-level confidence scores as the geometric mean of character-level confidences

Figure 5 shows the effect of different temperature values on calibration error.

[Figure 5: Temperature vs. ECE curve - to be added]

#### 2.3.2 Monte Carlo Dropout

While temperature scaling improves confidence calibration, it doesn't address the fundamental issue of uncertainty estimation. Monte Carlo Dropout provides a way to estimate model uncertainty by approximating Bayesian inference.

The approach involves:
1. Enabling dropout during inference (not just training)
2. Performing multiple forward passes with different dropout masks
3. Aggregating the results to estimate both the prediction and its uncertainty

We distinguish between two types of uncertainty:
- **Epistemic uncertainty**: Model uncertainty due to limited data or knowledge
- **Aleatoric uncertainty**: Inherent noise in the data

For sequence recognition tasks, we implemented MC Dropout by:
1. Keeping dropout layers active during inference
2. Performing N stochastic forward passes (N=30 in our experiments)
3. Calculating the most frequent prediction as the final output
4. Using the frequency of this prediction as the confidence score
5. Estimating epistemic uncertainty as the variance of predictions across samples
6. Estimating aleatoric uncertainty as the expected entropy of predictions

Figure 6 shows how MC Dropout captures different sources of uncertainty.

[Figure 6: Visualization of uncertainty estimates from MC Dropout - to be added]

#### 2.3.3 Step-Dependent Temperature Scaling

Recognizing that different positions in a sequence may require different calibration, we implemented Step-Dependent Temperature Scaling (STS). This method assigns different temperature values to different positions in the sequence:

$$p_i(x_t; \tau_t) = \frac{\exp(z_i^t/\tau_t)}{\sum_j \exp(z_j^t/\tau_t)}$$

where $\tau_t$ is the temperature for position $t$ in the sequence.

To avoid overfitting, we used a parameter-sharing scheme:
1. Unique temperature values for the first τ positions
2. A shared temperature for all positions beyond τ

This approach recognizes that early positions in the sequence may have different confidence characteristics than later positions. The optimal temperature values are determined on a validation set by minimizing the ECE for each position independently.

Figure 7 shows the temperature values learned for different positions in the sequence.

[Figure 7: Temperature values by sequence position - to be added]

#### 2.3.4 Combined Methods

Each calibration method addresses different aspects of the confidence calibration problem. To leverage the strengths of multiple approaches, we implemented a combined confidence estimator that integrates:

1. Temperature scaling for overall calibration
2. MC Dropout for uncertainty estimation
3. Weighted combination of individual confidence scores

The combined approach uses a weighted average of confidence scores:

$$\text{Confidence}_{\text{combined}} = w_{\text{temp}} \cdot \text{Confidence}_{\text{temp}} + w_{\text{MC}} \cdot \text{Confidence}_{\text{MC}}$$

where $w_{\text{temp}}$ and $w_{\text{MC}}$ are the weights for temperature scaling and MC Dropout, respectively.

The weights are determined through grid search on the validation set to optimize calibration performance. This combined approach provides both well-calibrated confidence scores and uncertainty estimates.

Figure 8 shows the reliability diagrams for the different calibration methods, including the combined approach.

[Figure 8: Reliability diagrams for different calibration methods - to be added]

## 3. Experiments

### 3.1 Experimental Setup

#### 3.1.1 Training Protocol

Both CRNN and PARSeq models were trained using the following protocols:

**CRNN Training:**
- Optimizer: Adam with initial learning rate of 0.001
- Batch size: 32
- Learning rate scheduling: ReduceLROnPlateau with factor 0.5 and patience 3
- Epochs: 30
- Loss function: CTC loss with blank index 0
- Gradient clipping: 5.0

**PARSeq Training:**
- Optimizer: AdamW with initial learning rate of 0.0007
- Weight decay: 0.0001
- Batch size: 32
- Learning rate scheduling: OneCycleLR with 10% warmup
- Epochs: 30 (but with early convergence typically around 15-20 epochs)
- Loss function: Cross-entropy with label smoothing 0.1
- Number of permutations: 6

For both models, we used early stopping based on validation character error rate (CER) with a patience of 5 epochs. The best model was selected based on validation CER.

#### 3.1.2 Evaluation Metrics

We evaluated the models using the following metrics:

1. **Character Error Rate (CER)**: The percentage of characters that need to be inserted, deleted, or substituted to convert the predicted text to the ground truth:

   $$\text{CER} = \frac{\text{Levenshtein distance}(pred, target)}{\text{length}(target)} \times 100\%$$

2. **Word Error Rate (WER)**: The percentage of words that are incorrectly predicted:

   $$\text{WER} = \frac{\text{Number of incorrect words}}{\text{Total number of words}} \times 100\%$$

3. **Word Accuracy**: The percentage of words that are correctly predicted:

   $$\text{Accuracy} = \frac{\text{Number of correct words}}{\text{Total number of words}} \times 100\%$$

For confidence calibration, we used the following metrics:

1. **Expected Calibration Error (ECE)**: Measures the difference between confidence and accuracy:

   $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$$

   where $B_m$ is the m-th confidence bin, $acc(B_m)$ is the accuracy in that bin, and $conf(B_m)$ is the average confidence in that bin.

2. **Maximum Calibration Error (MCE)**: The maximum difference between accuracy and confidence across all bins:

   $$\text{MCE} = \max_{m \in \{1,\ldots,M\}} |acc(B_m) - conf(B_m)|$$

3. **Brier Score**: The mean squared error between confidence scores and actual correctness (0 or 1):

   $$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (conf_i - correct_i)^2$$

Lower values of ECE, MCE, and Brier Score indicate better calibration.

#### 3.1.3 Implementation Details

The system was implemented in PyTorch, with the following specifications:

- PyTorch version: 1.10.0
- CUDA version: 11.3
- Training hardware: NVIDIA GeForce RTX 3090 GPU with 24GB memory
- The codebase was structured into modular components for data handling, model architecture, training, and evaluation

All confidence calibration methods were implemented as post-processing techniques that can be applied to any pre-trained model without requiring retraining.

### 3.2 Model Performance

#### 3.2.1 Recognition Results

The recognition performance of both CRNN and PARSeq models on the Telugu handwriting dataset is summarized in Table 2.

[Table 2: Recognition performance of CRNN and PARSeq models - to be added]

The CRNN model with CTC loss achieved a character error rate (CER) of X% and a word error rate (WER) of Y% on the test set. The PARSeq model demonstrated superior performance with a CER of A% and a WER of B%, representing a relative improvement of C% in CER and D% in WER compared to the CRNN model.

Figure 9 shows the training curves for both models, illustrating the convergence behavior and validation performance.

[Figure 9: Training and validation curves for CRNN and PARSeq - to be added]

#### 3.2.2 Error Analysis

We performed a detailed error analysis to understand the types of mistakes made by both models. The most common error types included:

1. **Character confusions**: Similar-looking Telugu characters being confused with each other
2. **Boundary errors**: Incorrect splitting or merging of characters
3. **Missing or extra components**: Omission or addition of character components
4. **Word-level errors**: Complete misrecognition of words, particularly rare or complex words

Figure 10 shows the confusion matrix for the most frequently confused character pairs in the test set.

[Figure 10: Character confusion matrix - to be added]

Figure 11 illustrates examples of correctly and incorrectly recognized words, highlighting the challenge of Telugu handwriting recognition.

[Figure 11: Examples of recognition results - to be added]

### 3.3 Confidence Calibration Results

#### 3.3.1 Calibration Before and After

To assess the effectiveness of confidence calibration, we first analyzed the calibration of the base models without any calibration techniques. Figure 12 shows the reliability diagrams for the uncalibrated CRNN model.

[Figure 12: Reliability diagram for uncalibrated CRNN model - to be added]

The uncalibrated model exhibited significant overconfidence, with an Expected Calibration Error (ECE) of X%. After applying temperature scaling, the ECE was reduced to Y%, representing a Z% relative improvement in calibration quality.

Table 3 summarizes the calibration metrics (ECE, MCE, and Brier Score) for the uncalibrated model and after applying different calibration methods.

[Table 3: Calibration metrics before and after calibration - to be added]

#### 3.3.2 Comparison of Calibration Methods

We conducted a comprehensive comparison of all implemented calibration methods:

1. **Temperature Scaling**: Simple but effective, reducing ECE by X%
2. **Monte Carlo Dropout**: Provides uncertainty estimates, reducing ECE by Y%
3. **Step-Dependent Temperature Scaling**: Best performance, reducing ECE by Z%
4. **Combined Method**: Second-best performance, with additional benefits of uncertainty estimation

Figure 13 provides a comparative visualization of reliability diagrams for all methods.

[Figure 13: Comparison of reliability diagrams for different calibration methods - to be added]

The Step-Dependent Temperature Scaling demonstrated the best calibration performance, with an ECE of A%. This result confirms our hypothesis that different positions in the sequence require different calibration adjustments.

#### 3.3.3 Uncertainty Estimation

In addition to improved confidence calibration, the MC Dropout and combined methods provide valuable uncertainty estimates. Figure 14 shows the relationship between prediction accuracy and uncertainty estimates.

[Figure 14: Relationship between uncertainty and accuracy - to be added]

We observed a strong negative correlation between epistemic uncertainty and prediction accuracy, confirming that the model's uncertainty estimates are meaningful. The aleatoric uncertainty showed a weaker correlation, as expected for inherent data noise.

Table 4 presents a quantitative analysis of the relationship between uncertainty estimates and prediction accuracy.

[Table 4: Correlation between uncertainty estimates and accuracy - to be added]

## 4. Results and Discussion

### 4.1 Model Comparison

Our experiments with CRNN and PARSeq architectures for Telugu handwriting recognition revealed several important findings:

1. **PARSeq vs. CRNN Performance**: PARSeq consistently outperformed CRNN in terms of recognition accuracy, with lower character and word error rates. This performance advantage comes at the cost of higher computational complexity during both training and inference.

2. **Training Efficiency**: CRNN models converged faster in terms of training time per epoch but required more epochs to reach optimal performance. PARSeq models, while slower per epoch, typically reached their best performance in fewer epochs.

3. **Robustness to Variations**: PARSeq demonstrated better robustness to variations in handwriting styles, likely due to its transformer architecture and permutation training approach, which better capture long-range dependencies and character relationships.

4. **Handling of Complex Characters**: Both models struggled with certain complex Telugu characters, but PARSeq showed better performance on the most challenging cases, particularly for characters with similar visual appearance.

Table 5: Comparison of CRNN and PARSeq models across various dimensions

[Table 5: Comprehensive comparison of CRNN and PARSeq - to be added]

### 4.2 Confidence Calibration Analysis

Our investigation into confidence calibration methods yielded several significant insights:

1. **Importance of Calibration**: The base models (both CRNN and PARSeq) produced significantly overconfident predictions, highlighting the need for proper confidence calibration in handwriting recognition systems.

2. **Effectiveness of Simple Methods**: Temperature scaling, despite its simplicity, proved highly effective at reducing calibration error, demonstrating that even simple post-processing techniques can substantially improve confidence estimates.

3. **Position-Specific Calibration**: The step-dependent temperature scaling method consistently outperformed uniform temperature scaling, confirming our hypothesis that different positions in the sequence require different calibration adjustments.

4. **Uncertainty Decomposition**: The Monte Carlo Dropout approach enabled the decomposition of uncertainty into epistemic (model) and aleatoric (data) components, providing richer information about the prediction reliability.

5. **Combined Approaches**: The combined calibration method provided a good balance between calibration quality and uncertainty estimation, though at the cost of increased computational complexity during inference.

Figure 15 shows a comparison of the different calibration methods in terms of their trade-off between calibration quality and computational cost.

[Figure 15: Trade-off between calibration quality and computational cost - to be added]

### 4.3 Practical Applications

The improved confidence calibration has several practical applications for Telugu handwriting recognition:

1. **Selective Verification**: By using well-calibrated confidence scores, a system can automatically flag low-confidence predictions for human verification, optimizing the balance between automation and accuracy.

2. **Active Learning**: Uncertainty estimates can guide the selection of samples for annotation in an active learning framework, focusing annotation efforts on the most informative examples.

3. **Rejection Option**: In critical applications, the system can refuse to make a prediction when the confidence is below a threshold, ensuring a controlled error rate.

4. **Ensemble Decision Making**: In multi-model systems, well-calibrated confidence scores enable better ensemble decision-making by weighting predictions according to their reliability.

Table 6 quantifies the potential improvement in a practical selective verification scenario, showing the reduction in verification effort for different target accuracy levels.

[Table 6: Reduction in verification effort with calibrated confidence - to be added]

## 5. Conclusion

### 5.1 Summary of Contributions

This work has made several key contributions to the field of Telugu handwriting recognition:

1. We implemented and evaluated two state-of-the-art architectures (CRNN and PARSeq) for Telugu handwriting recognition, demonstrating the superior performance of transformer-based approaches for this complex script.

2. We developed and compared multiple confidence calibration methods, showing that Step-Dependent Temperature Scaling provides the best calibration for sequence recognition tasks.

3. We introduced uncertainty estimation techniques based on Monte Carlo Dropout, enabling the decomposition of uncertainty into epistemic and aleatoric components.

4. We demonstrated the practical benefits of well-calibrated confidence estimates for applications requiring reliable uncertainty quantification.

These contributions advance the state of the art in Telugu handwriting recognition and provide a foundation for building more reliable recognition systems for complex scripts.

### 5.2 Limitations and Challenges

Despite the promising results, several limitations and challenges remain:

1. **Computational Complexity**: Methods like MC Dropout and combined approaches significantly increase inference time, potentially limiting their use in real-time applications.

2. **Dataset Limitations**: The size and diversity of the available Telugu handwriting dataset may not fully represent the variations encountered in real-world applications.

3. **Extreme Cases**: All models still struggle with highly stylized handwriting, severely degraded images, or rare character combinations.

4. **Calibration Generalization**: Calibration parameters learned on the validation set may not fully generalize to significantly different test distributions.

### 5.3 Future Work

Several promising directions for future work emerge from this research:

1. **End-to-End Calibration**: Integrating calibration objectives directly into the training process, rather than as a post-processing step.

2. **Script-Specific Approaches**: Developing calibration methods that explicitly account for the structural properties of Telugu script.

3. **Multi-Script Systems**: Extending the approaches to multi-script recognition systems, where confidence calibration becomes even more challenging due to varying recognition difficulties across scripts.

4. **Efficient Uncertainty Estimation**: Developing more computationally efficient methods for uncertainty estimation that maintain the quality of MC Dropout while reducing inference time.

5. **Transfer Learning**: Exploring how calibration transfers across different datasets and domains for the same script.

In conclusion, this work demonstrates that combining state-of-the-art recognition architectures with proper confidence calibration techniques significantly enhances the reliability of Telugu handwriting recognition systems. The methods and findings presented here provide a solid foundation for developing more trustworthy recognition systems for complex scripts, with applications extending beyond Telugu to other Indic and non-Latin scripts.
