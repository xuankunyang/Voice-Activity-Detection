# Voice Activity Detection (VAD) Project

🔗 For more details, please visit my blog [VAD](https://xuankunyang.github.io/blogs/vad/)

🎯 **Voice Activity Detection (VAD)**, or voice endpoint detection, identifies time segments in an audio signal containing speech. This is a critical preprocessing step for automatic speech recognition (ASR) and voice wake-up systems. This project lays the groundwork for my upcoming ASR project 🤭.

📈 **Workflow Overview**:
The VAD pipeline processes a speech signal as follows:
1. **Preprocessing**: Apply pre-emphasis to enhance high-frequency components.
2. **Framing**: Segment the signal into overlapping frames with frame-level labels.
3. **Windowing**: Apply window functions to mitigate boundary effects.
4. **Feature Extraction**: Extract a comprehensive set of features (e.g., short-time energy, zero-crossing rate, MFCCs, and more).
5. **Binary Classification**: Train models (DNN, Logistic Regression, Linear SVM, GMM) to classify frames as speech or non-speech.
6. **Time-Domain Restoration**: Convert frame-level predictions to time-domain speech segments.

🍻 **Project Highlights**:
I conducted extensive experiments comparing frame division methods (frame length and shift) and model performances, with rich visualizations. For details, see the report in `vad/latex/`. If you're interested in voice technologies, let's connect!


## Methodology

### 1. Preprocessing
Pre-emphasis enhances high-frequency components to reduce spectral leakage.

- **Pre-emphasis Impact**: 

  Effect of pre-emphasis with varying $\alpha$

  ![](/vad/latex/figs/counting_on_pre_emphasis.png)

### 2. Framing
The signal is divided into overlapping frames.

### 3. Windowing
A window function (e.g., Hamming) is applied to each frame.

- **Window Type Comparison**: 

  Impact of window functions (e.g., Hamming, Hanning)

  ![](/vad/latex/figs/counting_on_windows.png)

### 4. Feature Extraction
Extracted features (total dimension: 69) include:

- **Short-Time Energy** (dimension: 1):

  Measures frame energy, indicating loudness.

  Frame-level energy plots

  ![](/vad/latex/figs/energies.png)

- **Short-Time Zero-Crossing Rate** (dimension: 1):

  Counts zero crossings to distinguish voiced/unvoiced speech.

  Visualizing zero-crossing patterns

  ![](/vad/latex/figs/ZCR.png)

- **Fundamental Frequency (Pitch)** (dimension: 1):

  Estimated via autocorrelation, representing the fundamental frequency. And I have tried many smooth methods, see it in my report.

- **Spectral Centroid** (dimension: 1):

  Indicates the spectral "center of mass"

  ![](/vad/latex/figs/spectral_mean.png)

- **Sub-band Energies** (dimension: 6):

  Energy in 6 frequency sub-bands

  ![](vad\latex\figs\subband_energies.png)

- **Filter Banks (FBanks)** (dimension: 23):

  Mel-scale filter bank energies

  ![](vad\latex\figs\visualize_FBank.png)

- **Mel-Frequency Cepstral Coefficients (MFCCs)** (dimension: 12):

  Cepstral coefficients from mel filter banks

  ![](vad\latex\figs\visualize_MFCC.png)

- **Delta MFCCs** (dimension: 12):

  First-order differences of MFCCs

  ![](vad\latex\figs\visualize_Delta_MFCC.png)

- **Delta-Delta MFCCs** (dimension: 12):

  Second-order differences
  
  ![](vad\latex\figs\visualize_Delta_of_Delta_MFCC.png)

### 5. Classification Models
models I trained trained:
- **Deep Neural Network (DNN)**:
  - **Architecture**: Input (69) → 64 (ReLU, Dropout 0.5) → 32 (ReLU, Dropout 0.5) → 16 (ReLU, Dropout 0.5) → 1 (Sigmoid).
  - **Loss**: Binary Cross-Entropy.
  - **Training**: 10 epochs, Adam optimizer.

- **Logistic Regression**
  - Use **sklearn()**

- **Linear Support Vector Machine (SVM)**:
  - Use **sklearn()**

- **Gaussian Mixture Model (GMM)**:
  - Construct by myself
  - Models classes as Gaussian mixtures
  - Trained with Expectation-Maximization.
  - Its performance is not so good, I will spare some time to figure it out.

### 6. Experimental Results
Models were tested with frame lengths (320–4096) and shifts (80–2048). DNN outperformed others (frame length 4096, shift 1024):

| Model              | AUC    | EER    | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|----------|------------|-------------|----------|-------------|
| DNN                | 0.9876 | 0.0464 | 0.9603   | 0.9765    | 0.9749 | 0.9757   |
| Logistic Regression | 0.9457 | 0.1134 | 0.9432   | 0.9347    | 0.9389 | 0.9368   |
| Linear SVM | 0.8937| 0.9413| 0.1170| 0.9349| 0.9352 |0.9350|

### 7. Visualization on time domain division
I restored the framed labels back to the time domain and visualized them as follows:
![](vad\latex\figs\visualize_results_DNN.png)

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/xuankunyang/Voice-Activity-Detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Extract the features first and train a binary classifier, predict, time-domain restoration, visulize the time division. 
Considering the space limit, **I couldn't put my pre-trained models here**, but its not difficult to implement all above using this framework.
4. Explore visualizations in `vad/latex/figs/` and the report.

## Contributing
Fork the repository, create a branch, and submit pull requests. For major changes, open an issue.

## License
Licensed under the MIT License. See [LICENSE](https://mit-license.org/).

## Contact
Reach out via [email](kk-dao@sjtu.edu.cn) or GitHub issues.



# **Happy coding!** 🚀
