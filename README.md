readme = """# 📡 RF Signal Classifier

AI-powered web app that classifies radio signals (AM-DSB / FM / BPSK / QPSK) in real-time.

## 🚀 Live Demo
https://ehm9aa2v9wnpnorbkyqxvk.streamlit.app/

## 🔬 How It Works
The model extracts 9 features from each signal:
- **Time Domain:** Mean, Std, Max, Min, RMS
- **Frequency Domain:** FFT Peak, FFT Mean, FFT Std, Dominant Frequency

These features act as a fingerprint for each signal type.

## 🧠 Model
- **Algorithm:** Random Forest (200 estimators)
- **Training Data:** 2000 samples with randomized parameters
- **Accuracy:** 95%+ on unseen data
- **Dataset:** Based on RadioML concept with variable SNR (0-18 dB)

## ⚡ Features
- Generate signals with adjustable parameters
- Upload real CSV signal files for classification
- Real-time FFT visualization
- Confidence scores for each class

## 🛠️ Stack
- Python
- NumPy / SciPy
- Scikit-learn
- Streamlit
- Matplotlib

## 📊 Signal Types
| Type | Description |
|------|-------------|
| AM-DSB | Amplitude Modulation Double Sideband |
| FM | Frequency Modulation |
| BPSK | Binary Phase Shift Keying |
| QPSK | Quadrature Phase Shift Keying |

## 👨‍💻 Author
Telecommunications & Electronics Engineer
"""

with open('README.md', 'w') as f:
    f.write(readme)

from google.colab import files
files.download('README.md')
print("تم ✅")
