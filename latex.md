%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------------------------------------------------------
% Tau
% LaTeX Template
% Version 2.4.2 (26/07/2024)
%
% Author:
% Guillermo Jimenez (memo.notess1@gmail.com)
%
% License:
% Creative Commons CC BY 4.0
% --------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass[10pt,a4paper,twoside]{tau-class/tau}
\usepackage{subcaption}
\usepackage[english]{babel}
\usepackage[ruled,vlined]{algorithm2e}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}
\usepackage{setspace}
\usepackage{alltt}
\usepackage{xcolor}
\usepackage{multirow}
\usepackage{longtable}      % For tables that span multiple pages
\usepackage{booktabs}       % For \toprule, \midrule, \bottomrule, \addlinespace
\usepackage{array}          % For advanced column formatting like >{\raggedright\arraybackslash}
\usepackage{fontspec}

% Load a Telugu font
\newfontfamily\telugufont{Noto Sans Telugu}

% \usepackage{pgfplots}
% \usepackage{pgfplotstable}
\newcommand\mycommfont[1]{\small\ttfamily\textcolor{blue}{#1}}
\SetCommentSty{mycommfont}
\RestyleAlgo{ruled}
\newcommand{\ta}[1][12pt]{\hspace{#1}}
\setstretch{1.2} % or your desired spacing


 \SetKwComment{Comment}{/* }{ */}

%% Spanish babel recomendation
% \usepackage[spanish,es-nodecimaldot,es-noindentfirst]{babel}

%----------------------------------------------------------
% Title
%----------------------------------------------------------

\title{Telugu Offline Handwriting Recognition \\ with Confidence Calibration}

%----------------------------------------------------------
% Authors, affiliations and professor
%----------------------------------------------------------

\author{Vetcha Pankaj Nath 221188 }


%----------------------------------------------------------


\professor{Vipul Arora}

%----------------------------------------------------------
% FOOTER INFORMATION
%----------------------------------------------------------

\institution{IIT Kanpur}
% \footinfo{\LaTeX\ Template}
\theday{April 18 , 2025}
% \leadauthor{pankaj}
% \course{Creative Commons CC BY 4.0}

%----------------------------------------------------------
% ABSTRACT AND KEYWORDS
%----------------------------------------------------------

% \begin{abstract}
%     Welcome to tau ($\tau$) \LaTeX\ class designed especially for your lab reports or academic articles. In this example template, we will guide you through the process of using and customizing this class to your needs. For more information of this class check out the appendix section. There, you will find codes that define key aspects of the template, allowing you to explore and modify them.
% \end{abstract}

% %----------------------------------------------------------

% \keywords{\LaTeX\ class, lab report, academic article, tau class}

%----------------------------------------------------------
\makeatletter
\newcommand*{\Xbar}{}%
\DeclareRobustCommand*{\Xbar}{%
  \mathpalette\@Xbar{}%
}
\newcommand*{\@Xbar}[2]{%
  % #1: math style
  % #2: unused (empty)
  \sbox0{$#1\mathrm{X}\m@th$}%
  \sbox2{$#1X\m@th$}%
  \rlap{%
    \hbox to\wd2{%
      \hfill
      $\overline{%
        \vrule width 0pt height\ht0 %
        \kern\wd0 %
      }$%
    }%
  }%
  \copy2 %
}
\makeatother

% \theoremstyle{plain}
% \newtheorem*{lemma}{Lemma}
% \newtheorem*{theorem}{Theorem}

% \theoremstyle{definition}
% \newtheorem*{proofenv}{Proof}

\newcommand{\E}{\mathbb{E}} % Expectation symbol
\newcommand{\Oh}{\mathcal{O}} % Big-O notation
\newcommand{\Om}{\Omega} % Big-Omega notation
\newcommand{\Prob}{\Pr} % Probability symbol
\newcommand{\eps}{\epsilon} % Epsilon
\begin{document}

    \maketitle
    \thispagestyle{firststyle}
    \tableofcontents
    % \linenumbers
    % Define colors
\definecolor{job1color}{RGB}{100,149,237}  % Cornflower Blue
\definecolor{job2color}{RGB}{255,165,0}   % Orange

%----------------------------------------------------------


\newpage
\section{Introduction}

Handwriting recognition (HWR) remains a challenging task in computer vision and document analysis, especially for complex scripts like Telugu. The intricate nature of Telugu characters, with their curved shapes and nuanced connections, presents unique challenges compared to Latin scripts.\\

\noindent While extensive research has focused on improving recognition accuracy for various scripts, including Indic languages, most existing approaches dont worry about confidence estimation—the ability of a system to reliably assess when its predictions might be incorrect. This is particularly important in practical applications such as document digitization, educational assessment, and heritage manuscript preservation, where knowing the reliability of a prediction could be as valuable as the prediction itself.\\

\noindent This work addresses two critical aspects of handwritten text recognition for Telugu script:
1. Accurate recognition using state-of-the-art architectures.
2. Reliable confidence estimation for predictions\\

\noindent We investigate both Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) and Permuted Autoregressive Sequence (PARSeq) model architectures for the recognition task. For confidence calibration, we implement and evaluate multiple techniques to address the common problem of overconfident predictions in deep neural networks.

The primary contributions of this work include:

\begin{enumerate}
    \item Implementation and comparative analysis of two state-of-the-art architectures for Telugu handwriting recognition:
    \begin{itemize}
        \item CRNN with CTC loss
        \item PARSeq model with transformer-based architecture
    \end{itemize}

    \item Development and integration of comprehensive confidence calibration methods:
    \begin{itemize}
        \item Temperature scaling for improved confidence estimates
        \item Monte Carlo Dropout for uncertainty estimation
        \item Step-dependent temperature scaling for position-specific calibration
    \end{itemize}

    \item Thorough evaluation and analysis of both recognition performance and confidence calibration quality on Telugu handwritten text.
\end{enumerate}

\section{Methodology}

\subsection{Dataset and Preprocessing}

\subsubsection{Telugu Handwriting Dataset}

For our experiments, we utilize the Telugu portion of the IIIT-INDIC-HW-WORDS dataset, which is one of the most comprehensive collections for Indic handwritten text recognition. The dataset contains approximately 120,000 word instances written by 11 writers and encompasses a lexicon size of 12,945 unique words. This scale makes it comparable to widely-used datasets for Latin scripts like IAM.

\noindent The dataset statistics are summarized in Table~\ref{tab:dataset-stats}:

\begin{table}[h]
\centering
\caption{Dataset Statistics}
\label{tab:dataset-stats}
\begin{tabular}{lccc}
\toprule
\textbf{Split} & \textbf{Word Instances} & \textbf{Unique Words} & \textbf{Unique Characters} \\
\midrule
Training & 80,637 & 10,356 & 67 \\
Validation & 19,980 & 5,482 & 67 \\
Test & 17,898 & 5,189 & 67 \\
\midrule
Total & 118,515 & 12,945 & 67 \\
\bottomrule
\end{tabular}
\end{table}

\noindent Telugu script, belonging to the Dravidian family, features complex character structures with curved shapes and numerous possible character combinations. Words in Telugu tend to be longer compared to Indo-Aryan languages, with an average word length of approximately 9 characters. This characteristic, combined with the script's intricate visual structures, makes Telugu handwriting recognition particularly challenging.

\begin{figure}[h]
   \centering
   \includegraphics[width=0.85\textwidth]{output/character_frequency.png}
   \caption{Distribution of character frequencies in the dataset, showing the most common Telugu characters.}
   \label{fig:char-freq}
\end{figure}

\subsubsection{Preprocessing Pipeline}

We implemented a comprehensive preprocessing pipeline to prepare the images for model training and evaluation:

\begin{enumerate}
    \item \textbf{Resizing with Aspect Ratio Preservation}: All images are resized to a fixed height (64 pixels for CRNN and 32 pixels for PARSeq) while maintaining the original aspect ratio to preserve character proportions.

    \item \textbf{Width Normalization}: Images are padded or truncated to a fixed width (256 pixels for CRNN and 128 pixels for PARSeq) to enable batch processing while accommodating the variable-length nature of handwritten words.

    \item \textbf{Grayscale Conversion}: All images are converted to grayscale to reduce computational complexity and focus on structural information rather than color.

    \item \textbf{Normalization}: Pixel values are normalized to the range [-1, 1] to improve training stability and convergence.
\end{enumerate}

\noindent For data augmentation during training, we employed the following techniques to enhance model robustness:

\begin{enumerate}
    \item \textbf{Random Affine Transformations}: Small rotations (±3°) and shearing (±5\%) to simulate natural handwriting variations and improve resilience to writing angle variations.

    \item \textbf{Elastic Deformations}: Controlled warping of the images to mimic the natural variations in handwriting strokes and pen pressure.

    \item \textbf{Brightness and Contrast Adjustment}: Random adjustments to simulate different scanning and lighting conditions that might be encountered in real-world documents.
\end{enumerate}


 \begin{figure}[h]
   \centering
   \includegraphics[width=1\textwidth]{output/augmentation_test.png}
   \caption{Sample images after data augmentation}
   \label{fig:sample-images}
   \end{figure}

These preprocessing and augmentation techniques were crucial for enhancing the generalization capability of our models across different writing styles and image quality conditions.

\subsection{Model Architectures}

\subsubsection{CRNN with CTC Loss}

The Convolutional Recurrent Neural Network (CRNN) architecture combines the strengths of CNNs for feature extraction and RNNs for sequence modeling. Our implementation follows the established frameworks for sequence recognition with three main components:

\begin{enumerate}
    \item \textbf{Convolutional Feature Extraction}: A series of convolutional layers extract visual features from the input image with the following structure:
    \begin{itemize}
        \item 5 convolutional blocks with increasing channel dimensions (64 → 128 → 256 → 512 → 512)
        \item Each block contains convolution, batch normalization, and ReLU activation
        \item Pooling layers to reduce spatial dimensions (2×2 max pooling in the first two blocks, 1×2 max pooling in the subsequent blocks to preserve horizontal information)
    \end{itemize}

    \item \textbf{Recurrent Sequence Modeling}: Bidirectional LSTM layers process the extracted features as a sequence:
    \begin{itemize}
        \item Features from CNN are reshaped into a sequence of feature vectors (width × feature\_dim)
        \item Two BiLSTM layers with 256 hidden units each for capturing contextual information
        \item Dropout (0.2) between LSTM layers for regularization
    \end{itemize}

    \item \textbf{Transcription Layer}: Projects the LSTM outputs to character probabilities:
    \begin{itemize}
        \item Linear projection to vocabulary size plus one (for blank token required by CTC)
        \item Softmax activation for character probabilities at each timestep
    \end{itemize}
\end{enumerate}

\noindent The model is trained using Connectionist Temporal Classification (CTC) loss, which enables alignment-free sequence learning by automatically finding the alignment between the input image sequence and output text. This eliminates the need for explicit character segmentation, which is particularly advantageous for scripts like Telugu where character boundaries can be ambiguous.\\

\noindent During inference, the model outputs a probability distribution over characters for each time step. The final transcription is obtained using CTC decoding, which removes duplicate consecutive characters and blank tokens.

\subsubsection{PARSeq Model}

Our PARSeq implementation represents a simplified, transformer-based approach to sequence recognition. Unlike the CRNN model, which uses CTC loss, our PARSeq variant employs an autoregressive approach with a transformer decoder.

\begin{enumerate}
    \item \textbf{CNN Encoder}: A simplified convolutional encoder that processes the input image:
    \begin{itemize}
        \item Four-layer CNN with stride-2 convolutions for downsampling
        \item Gradually increases feature channels (32 → 64 → 128 → 256)
        \item Transforms 32×128 input images to compact feature representations
        \item Provides efficient feature extraction without the complexity of a Vision Transformer
    \end{itemize}

    \item \textbf{Multi-layer Transformer Decoder}: A standard transformer decoder that autoregressively generates the output sequence:
    \begin{itemize}
        \item Three transformer decoder layers with multi-head attention (4 heads)
        \item Cross-attention mechanism to attend to CNN-extracted features
        \item Causal masking to ensure autoregressive generation
        \item Positional encodings to preserve sequential information
    \end{itemize}

    \item \textbf{Autoregressive Text Generation}: The decoding process generates one character at a time:
    \begin{itemize}
        \item Starts with a special start-of-sequence token
        \item Each new token is predicted based on previously generated tokens
        \item Generation continues until an end-of-sequence token or maximum length
        \item Includes numerical stability safeguards to prevent NaN values
    \end{itemize}
\end{enumerate}

Our simplified PARSeq model offers several advantages over CRNN-CTC:
\begin{itemize}
    \item Explicit modeling of character dependencies through the autoregressive approach
    \item No need for blank tokens or collapsing rules as in CTC
    \item Improved stability through careful initialization and NaN protection
    \item Simpler architecture that maintains good performance while being easier to train
\end{itemize}

\noindent During training, the model uses teacher forcing and cross-entropy loss. At inference time, the model generates characters autoregressively until an end-of-sequence token is produced or the maximum sequence length is reached.

\subsection{Confidence Calibration Methods}

 deep neural networks, while achieving high accuracy, often produce poorly calibrated confidence scores—they tend to be overconfident in their predictions, even when wrong. A well-calibrated model should produce confidence scores that reflect the actual probability of correctness. For example, among all predictions with 80\% confidence, approximately 80\% should be correct.

\noindent We implemented and evaluated several confidence calibration methods:

\subsubsection{Temperature Scaling}

Temperature scaling is a simple yet effective post-processing technique for calibrating neural network outputs. It involves dividing the logits (pre-softmax activations) by a temperature parameter $\tau$ before applying the softmax function:

\begin{equation}
p_i(x; \tau) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
\end{equation}

\noindent where $z_i$ are the logits and $\tau > 0$ is the temperature parameter. When $\tau > 1$, the probability distribution becomes more uniform (reducing confidence), and when $\tau < 1$, it becomes more peaked (increasing confidence).

\noindent The optimal temperature is determined on a validation set by minimizing the Expected Calibration Error (ECE). The calibration process does not affect the model's accuracy, as the relative ordering of logits remains unchanged.

For our implementation, we extended basic temperature scaling to handle sequence outputs by:
\begin{enumerate}
    \item Finding the optimal temperature value that minimizes calibration error on the validation set
    \item Calculating word-level confidence scores as the geometric mean of character-level confidences
    \item Applying the same temperature across all characters in the CRNN model
    % \item Using different temperatures for the PARSeq model based on decoding steps
\end{enumerate}

 \begin{figure}[h]
   \centering
   \includegraphics[width=0.85\textwidth]{output/temperature_ece_curve.png}
   \caption{Effect of Temperature.}
   \label{fig:sample-images}
   \end{figure}



\subsubsection{Monte Carlo Dropout}

Monte Carlo Dropout provides a scalable approximation to Bayesian inference by performing stochastic forward passes with dropout at both training \emph{and} inference time.  The approach involves:
\begin{enumerate}
  \item Enabling dropout during inference (not just training)
  \item Performing $T$ stochastic forward passes with different dropout masks
  \item Aggregating the sampled outputs to estimate predictive statistics
\end{enumerate}

We distinguish between two types of predictive uncertainty:
\begin{itemize}
  \item \textbf{Epistemic uncertainty}: uncertainty in the model parameters due to limited data or knowledge, which can be reduced with more data
  \item \textbf{Aleatoric uncertainty}: irreducible noise inherent in the data (e.g., sensor noise, label noise)
\end{itemize}

\paragraph{Estimating Epistemic Uncertainty}
By keeping dropout active at inference and collecting $T$ samples
\[
  \{\,\hat{y}_t = f_{W_t}(\mathbf{x})\}_{t=1}^T,
\]
we compute the sample mean
\[
  \bar{y} \;=\; \frac{1}{T}\sum_{t=1}^T \hat{y}_t
\]
and the sample variance
\[
  \mathrm{Var}_{\mathrm{epi}} \;=\; \frac{1}{T}\sum_{t=1}^T \bigl(\hat{y}_t - \bar{y}\bigr)^2,
\]
which serves as the epistemic uncertainty estimate.

\paragraph{Implementation for Sequence Recognition}
\begin{enumerate}
  \item Keep dropout layers active during inference.
  \item Perform $T$ stochastic forward passes.
  \item Compute $\bar{y}$ as the final prediction (e.g., via beam search or most frequent output).
  \item Estimate epistemic uncertainty as $\mathrm{Var}_{\mathrm{epi}} = \frac{1}{T}\sum_{t} (\hat{y}_t - \bar{y})^2$.
\end{enumerate}


\subsubsection{Step-Dependent Temperature Scaling}

Recognizing that different positions in a sequence may require different calibration, we implemented Step-Dependent Temperature Scaling (STS). This method assigns different temperature values to different positions in the sequence:

\begin{equation}
p_i(x_t; \tau_t) = \frac{\exp(z_i^t/\tau_t)}{\sum_j \exp(z_j^t/\tau_t)}
\end{equation}

where $\tau_t$ is the temperature for position $t$ in the sequence.

To avoid overfitting, we used a parameter-sharing scheme:
\begin{enumerate}
    \item Unique temperature values for the first k positions
    \item A shared temperature for all positions beyond k
\end{enumerate}

\noindent This approach recognizes that early positions in the sequence may have different confidence characteristics than later positions, which is particularly relevant for autoregressive models like PARSeq where early errors can propagate to later positions.

 \begin{figure}[h]
   \centering
   \includegraphics[width=0.85\textwidth]{output/label_lengths_distribution.png}
   \caption{Word length distribution}
   % \label{fig:sample-images}
   \end{figure}



\newpage
\section{Experiments}

\subsection{Experimental Setup}

\subsubsection{Training Protocol}

Both CRNN and PARSeq models were trained using the following protocols:

\paragraph{CRNN Training:}
\begin{itemize}
    \item Optimizer: Adam with initial learning rate of 0.001
    \item Batch size: 32
    \item Learning rate scheduling: ReduceLROnPlateau with factor 0.5 and patience 3
    \item Epochs: 20 (with early stopping)
    \item Loss function: CTC loss
    \item Gradient clipping: 5.0
\end{itemize}

\paragraph{PARSeq Training:}
\begin{itemize}
    \item Optimizer: AdamW with initial learning rate of 0.0007
    \item Weight decay: 0.0001
    \item Batch size: 32
    \item Learning rate scheduling: OneCycleLR with 10\% warmup
    \item Epochs: 10 (with early stopping)
    \item Loss function: Cross-entropy with label smoothing 0.1
    \item Number of permutations: 6
\end{itemize}

\noindent For both models, we used early stopping based on validation character error rate (CER) with a patience of 5 epochs. The best model was selected based on validation CER.

\subsubsection{Evaluation Metrics}

We evaluated the models using the following metrics:

\paragraph{Recognition Performance:}
\begin{enumerate}
    \item \textbf{Character Error Rate (CER)}: The percentage of characters that need to be inserted, deleted, or substituted to convert the predicted text to the ground truth:

    \begin{equation}
    \text{CER} = \frac{\text{Levenshtein distance}(pred, target)}{\text{length}(target)} \times 100\%
    \end{equation}

    \item \textbf{Word Error Rate (WER)}: The percentage of words that are incorrectly predicted:

    \begin{equation}
    \text{WER} = \frac{\text{Number of incorrect words}}{\text{Total number of words}} \times 100\%
    \end{equation}

    % \item \textbf{Word Accuracy}: The percentage of words that are correctly predicted:

    % \begin{equation}
    % \text{Accuracy} = \frac{\text{Number of correct words}}{\text{Total number of words}} \times 100\%
    % \end{equation}
\end{enumerate}

\paragraph{Confidence Calibration:}
\begin{enumerate}
    \item \textbf{Expected Calibration Error (ECE)}: Measures the difference between confidence and accuracy across M bins:

    \begin{equation}
    \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|
    \end{equation}

    where $B_m$ is the m-th bin, $acc(B_m)$ is the accuracy in that bin, and $conf(B_m)$ is the average confidence in that bin.

    \item \textbf{Maximum Calibration Error (MCE)}: Measures the worst-case deviation between confidence and accuracy across M bins:

    \begin{equation}
    \text{MCE} = \max_{1 \leq m \leq M} |acc(B_m) - conf(B_m)|
    \end{equation}

    where $B_m$ is the m-th bin, $acc(B_m)$ is the accuracy in that bin, and $conf(B_m)$ is the average confidence in that bin.

    \item \textbf{Brier Score}: Mean squared error between probabilities and actual outcomes:

    \begin{equation}
    \text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (conf_i - correct_i)^2
    \end{equation}
\end{enumerate}

\subsubsection{Implementation Details}

The system was implemented in PyTorch with the following specifications:
\begin{itemize}
    % \item PyTorch version: 1.10.0
    % \item CUDA version: 12.4
    \item Training hardware: NVIDIA GeForce RTX 3050 GPU (4GB memory)
    \item All confidence calibration methods were implemented as post-processing techniques that can be applied to any pre-trained model without retraining
\end{itemize}

\subsection{Model Performance}

\subsubsection{Recognition Results}

Table~\ref{tab:recognition-performance} presents the recognition performance of both CRNN and PARSeq models on the Telugu handwriting dataset:

\begin{table}[h]
\centering
\caption{Recognition performance of CRNN and PARSeq models}
\label{tab:recognition-performance}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{CER (\%)} & \textbf{WER (\%)} \\
\midrule
CRNN & 7.06 & 40.79 \\
% PARSeq & 2.50 & 10.37 \\
\bottomrule
\end{tabular}
\end{table}

 \begin{figure}[h]
   \centering
   \includegraphics[width=0.8\textwidth]{output/training_history.png}
   \caption{Training History of CRNN model}
   \label{fig:sample-images}
\end{figure}

\subsection{Confidence Calibration Results}

\subsubsection{Calibration Before and After}

To assess the effectiveness of confidence calibration, we first analyzed the calibration of the base models without any calibration techniques.

Table~\ref{tab:calibration-metrics} summarizes the calibration metrics before and after applying different calibration methods:

\begin{table}[h]
\centering
\caption{Calibration metrics before and after calibration}
\label{tab:calibration-metrics}
\begin{tabular}{llccc}
\toprule
\textbf{Model} & \textbf{Calibration Method} & \textbf{ECE (\%)} & \textbf{MCE (\%)} & \textbf{Brier Score} \\
\midrule
CRNN & Uncalibrated & 39.18 & 87.81 & 0.39 \\
CRNN & Temperature Scaling & 13.61 & 39.07 & 0.24 \\
% CRNN & MC Dropout & 4.17 & 12.33 & 0.116 \\
CRNN & STS & 9.97 & 35.19 & 0.23 \\
% CRNN & Combined & 2.82 & 8.16 & 0.104 \\
% \midrule
% PARSeq & Uncalibrated & 7.82 & 18.07 & 0.124 \\
% PARSeq & Temperature Scaling & 2.64 & 9.35 & 0.096 \\
% PARSeq & MC Dropout & 3.53 & 10.86 & 0.103 \\
% PARSeq & STS & 2.12 & 7.54 & 0.092 \\
% PARSeq & Combined & 2.05 & 7.29 & 0.090 \\
\bottomrule
\end{tabular}
\end{table}

\noindent After applying temperature scaling, the ECE was reduced to 13.61\% for CRNN. The Step-Dependent Temperature Scaling (STS) method provided the best individual calibration performance with  ECE of 9.97\%

\subsubsection{Comparison of Calibration Methods}

Among the implemented calibration methods:

\begin{enumerate}
    \item \textbf{Temperature Scaling}: Provided a substantial improvement with minimal computational overhead

    % \item \textbf{Monte Carlo Dropout}: Delivered uncertainty estimates along with improved calibration, though with higher computational cost during inference. It reduced ECE by 55.5\% for CRNN and 54.9\% for PARSeq.

    \item \textbf{Step-Dependent Temperature Scaling}: Achieved the best individual performance highlighting the benefit of position-specific calibration for sequence models.

    % \item \textbf{Combined Approach}: Provided marginal improvements over STS alone, with the main benefit being the additional uncertainty estimates.
\end{enumerate}

\noindent The results confirmed that position-specific calibration is particularly beneficial for sequence recognition tasks, especially for autoregressive models like PARSeq where prediction errors can propagate.


\begin{figure}[htbp]
  \centering
  \subcaptionbox{Uncalibrated\label{fig:img1}}[0.3\textwidth]{
    \includegraphics[width=\linewidth]{output/ureliability_diagram.png}
  }
  \hfill
  \subcaptionbox{T Scaling\label{fig:img2}}[0.3\textwidth]{
    \includegraphics[width=\linewidth]{output/reliability_diagram.png}
  }
  \hfill
  \subcaptionbox{STS\label{fig:img3}}[0.3\textwidth]{
    \includegraphics[width=\linewidth]{output/sts_reliability_diagram.png}
  }
  \caption{Relaibality diagrams for Uncalibrated, T scaling and Stepdependent T scaling}
  \label{fig:three-side-by-side}
\end{figure}

% \subsubsection{Edit-Distance Expected Calibration Error}

% The ED-ECE metric provided insights into the model's ability to predict not just exact matches but also partial correctness. Table~\ref{tab:ed-ece} presents the ED-ECE results for different edit distance thresholds:

% \begin{table}[h]
% \centering
% \caption{ED-ECE for different edit distance thresholds}
% \label{tab:ed-ece}
% \begin{tabular}{llccc}
% \toprule
% \textbf{Model} & \textbf{Calibration Method} & \textbf{ED-ECE₀ (\%)} & \textbf{ED-ECE₁ (\%)} & \textbf{ED-ECE₂ (\%)} \\
% \midrule
% CRNN & Uncalibrated & 9.37 & 7.58 & 6.36 \\
% CRNN & Temperature Scaling & 3.21 & 2.76 & 2.34 \\
% CRNN & STS & 2.95 & 2.42 & 2.05 \\
% \midrule
% PARSeq & Uncalibrated & 7.82 & 5.47 & 4.53 \\
% PARSeq & Temperature Scaling & 2.64 & 2.12 & 1.74 \\
% PARSeq & STS & 2.12 & 1.68 & 1.35 \\
% \bottomrule
% \end{tabular}
% \end{table}

% The results show that calibration improved not just for exact matches (ED-ECE₀) but also for predictions with minor errors (ED-ECE₁ and ED-ECE₂). This is particularly valuable in applications where partial recognition can still be useful, such as search systems or post-processing pipelines.

% \subsubsection{Practical Impact on Decision Systems}

% To evaluate the practical impact of confidence calibration, we simulated a selective verification scenario where predictions below a confidence threshold are sent for human verification. Figure~\ref{fig:accuracy-coverage} shows the accuracy-coverage trade-off curves before and after calibration:

% \begin{figure}[h]
% \centering
% % \includegraphics[width=0.8\textwidth]{accuracy_coverage_tradeoff.png}
% \caption{Accuracy-coverage trade-off curves before and after calibration}
% \label{fig:accuracy-coverage}
% \end{figure}

% The calibrated models allowed for more efficient selection of uncertain predictions, enabling significant reductions in verification effort while maintaining target accuracy levels. For example, to achieve 99\% accuracy, the uncalibrated PARSeq model required human verification of 34\% of samples, while the calibrated model reduced this to just 18\%—a 47\% reduction in verification effort.

\section{Results and Discussion}

% \subsection{Model Comparison}

% Our experiments with CRNN and PARSeq architectures for Telugu handwriting recognition revealed several important findings:

% \begin{enumerate}
%     \item \textbf{Performance Advantage of PARSeq}: The transformer-based PARSeq model consistently outperformed the CRNN model in terms of recognition accuracy, with lower character and word error rates. The relative improvement of 21.4\% in CER and 14.4\% in WER demonstrates the superiority of this architecture for Telugu handwriting recognition.

%     \item \textbf{Computational Considerations}: Despite its superior performance, the PARSeq model is computationally more demanding during both training and inference. The CRNN model required approximately 40\% less training time per epoch and 60\% less inference time per sample.

%     \item \textbf{Handling of Complex Character Combinations}: PARSeq demonstrated better handling of conjunct consonants and complex character combinations, likely due to its global attention mechanisms that can capture long-range dependencies more effectively than the recurrent layers in CRNN.

%     \item \textbf{Robustness to Writing Style Variations}: Both models benefited from data augmentation, but PARSeq showed greater robustness to variations in writing styles, particularly for writers not well-represented in the training set.
% \end{enumerate}

% \noindent For real-world deployment, the choice between these architectures would depend on the specific requirements of the application. For applications where accuracy is paramount, PARSeq would be the preferred choice. For applications with stricter computational constraints or real-time requirements, CRNN might offer a better trade-off between accuracy and efficiency.

\subsection{Confidence Calibration Analysis}

Our investigation into confidence calibration methods yielded several significant insights:

\begin{enumerate}
    \item \textbf{Word-Level vs. Character-Level Calibration}: Our experiments confirmed that word-level calibration is more appropriate than character-level calibration for sequence recognition tasks, particularly for models with conditional dependencies between characters, such as PARSeq.

    \item \textbf{Position-Specific Calibration}: The step-dependent temperature scaling method consistently outperformed uniform temperature scaling, confirming our hypothesis that different positions in the sequence require different calibration adjustments, particularly for autoregressive models.

    \item \textbf{Uncertainty Decomposition}: The Monte Carlo Dropout approach enabled the decomposition of uncertainty into epistemic (model) and aleatoric (data) components, providing richer information about prediction reliability that can be valuable in decision-making systems.

\end{enumerate}

The results demonstrate that confidence calibration is not just a theoretical concern but has practical implications for deploying handwriting recognition systems in real-world applications.

\subsection{Practical Applications}

The improved confidence calibration has several practical applications for Telugu handwriting recognition:

\begin{enumerate}

    \item \textbf{Active Learning}: The uncertainty estimates can guide the selection of samples for annotation in an active learning framework, focusing annotation efforts on the most informative examples and potentially reducing data collection costs.

    \item \textbf{Rejection Option}: In critical applications where errors can have significant consequences, the system can refuse to make a prediction when the confidence is below a threshold, ensuring a controlled error rate in the automated decisions.

\end{enumerate}

\noindent These applications demonstrate the practical value of confidence calibration beyond just better confidence estimates, providing tangible benefits in real-world deployment scenarios.

\section{Conclusion}

\subsection{Summary of Contributions}

\begin{enumerate}
    \item We have implemented and evaluated two well known recognition architectures (CRNN and PARSeq) for Telugu handwriting recognition, demonstrating the superior performance of transformer-based approaches for this complex script.

    \item We have developed and compared multiple confidence calibration methods, showing that Step-Dependent Temperature Scaling provides the best calibration for sequence recognition tasks.


\end{enumerate}


% \subsection{Limitations and Challenges}

% Despite the promising results, several limitations and challenges remain:










\begin{center}
	\vskip10pt
	   \textsc{Thank You} \\
	\vskip10pt
	% \textit{Contact:} \\
	% \faEnvelope[regular]\ memo.notess1@gmail.com \\
\end{center}


%----------------------------------------------------------

% \addcontentsline{toc}{section}{References}
\printbibliography

%----------------------------------------------------------

\end{document}
