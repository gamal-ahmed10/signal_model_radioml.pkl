import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.fft import fft

st.set_page_config(page_title="RF Signal Classifier", page_icon="📡", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.stApp { background: #0a0a0f; color: #e0e0e0; }
.result-card {
    background: #111118;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: 4px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("signal_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

fs = 128
t = np.linspace(0, 1, fs, endpoint=False)

def generate_signal(sig_type, cf, mf, snr):
    noise_power = 10**(-snr/10)
    noise = np.random.randn(fs) * np.sqrt(noise_power)
    if sig_type == "AM-DSB":
        return np.cos(2*np.pi*mf*t) * np.cos(2*np.pi*cf*t) + noise
    elif sig_type == "FM":
        return np.cos(2*np.pi*cf*t + 5*np.sin(2*np.pi*mf*t)) + noise
    elif sig_type == "BPSK":
        bits = np.random.choice([-1,1], size=fs)
        return bits + noise
    elif sig_type == "QPSK":
        bits = np.random.choice([-1,-0.5,0.5,1], size=fs)
        return bits + noise

def extract_features(sig):
    mean_val = np.mean(sig)
    std_val = np.std(sig)
    max_val = np.max(sig)
    min_val = np.min(sig)
    rms_val = np.sqrt(np.mean(sig**2))
    fft_vals = np.abs(fft(sig))[:len(sig)//2]
    fft_peak = np.max(fft_vals)
    fft_mean = np.mean(fft_vals)
    fft_std = np.std(fft_vals)
    dominant_freq = np.argmax(fft_vals)
    return [mean_val,std_val,max_val,min_val,rms_val,fft_peak,fft_mean,fft_std,dominant_freq]

# --- UI ---
st.markdown("# 📡 RF Signal Classifier")
st.markdown("مدرب على **RadioML Dataset** — إشارات راديو حقيقية")
st.markdown("---")

tab1, tab2 = st.tabs(["🎛️ توليد إشارة", "📁 رفع ملف CSV"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        sig_type = st.selectbox("نوع الإشارة", ["AM-DSB", "FM", "BPSK", "QPSK"])
        cf = st.slider("Carrier Freq (Hz)", 10, 50, 40)
        mf = st.slider("Mod Freq (Hz)", 1, 20, 10)
        snr = st.slider("SNR (dB)", 0, 18, 10)
        classify_btn = st.button("صنف الإشارة ▶", use_container_width=True)

    sig = generate_signal(sig_type, cf, mf, snr)
    features = extract_features(sig)

    with col2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
        fig.patch.set_facecolor('#0a0a0f')
        colors = {"AM-DSB": "#4fc3f7", "FM": "#81c784", "BPSK": "#ffb74d", "QPSK": "#f06292"}
        c = colors[sig_type]

        ax1.set_facecolor('#111118')
        ax1.plot(t, sig, color=c, linewidth=1.2)
        ax1.set_title("Time Domain", color='#888', fontsize=10)
        ax1.tick_params(colors='#555')
        for spine in ax1.spines.values(): spine.set_edgecolor('#2a2a3a')

        fft_vals = np.abs(fft(sig))[:fs//2]
        freqs = np.linspace(0, fs//2, len(fft_vals))
        ax2.set_facecolor('#111118')
        ax2.plot(freqs, fft_vals, color=c, linewidth=1.2)
        ax2.set_title("Frequency Domain (FFT)", color='#888', fontsize=10)
        ax2.tick_params(colors='#555')
        for spine in ax2.spines.values(): spine.set_edgecolor('#2a2a3a')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if classify_btn:
        prediction = model.predict([features])[0]
        correct = prediction == sig_type
        color_map = {"AM-DSB": "#4fc3f7", "FM": "#81c784", "BPSK": "#ffb74d", "QPSK": "#f06292"}

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:13px;color:#666;margin-bottom:8px;letter-spacing:2px;">PREDICTED</div>
            <div class="result-label" style="color:{color_map.get(prediction,'#fff')}">{prediction}</div>
            <div style="margin-top:12px;font-size:14px;color:{'#81c784' if correct else '#f06292'}">
                {'✓ تصنيف صح!' if correct else '✗ الموديل اتخدع بالـ noise'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            proba = model.predict_proba([features])[0]
            classes = model.classes_
            st.markdown("#### Confidence")
            for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
                st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
        except:
            pass

with tab2:
    st.markdown("#### ارفع ملف CSV فيه إشارة")
    st.markdown("الملف لازم يكون فيه عمود واحد من الأرقام بيمثل الإشارة")
    
    uploaded = st.file_uploader("اختار ملف CSV", type=["csv"])
    
    if uploaded:
        import pandas as pd
        df = pd.read_csv(uploaded, header=None)
        sig_csv = df.iloc[:, 0].values[:128].astype(float)
        
        if len(sig_csv) < 10:
            st.error("الملف فيه بيانات قليلة جداً!")
        else:
            sig_csv = sig_csv[:128]
            if len(sig_csv) < 128:
                sig_csv = np.pad(sig_csv, (0, 128-len(sig_csv)))
            
            features_csv = extract_features(sig_csv)
            prediction_csv = model.predict([features_csv])[0]
            
            color_map = {"AM-DSB": "#4fc3f7", "FM": "#81c784", "BPSK": "#ffb74d", "QPSK": "#f06292"}
            
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#0a0a0f')
            ax.set_facecolor('#111118')
            ax.plot(sig_csv, color=color_map.get(prediction_csv, '#fff'), linewidth=1.2)
            ax.set_title("الإشارة اللي رفعتها", color='#888')
            ax.tick_params(colors='#555')
            for spine in ax.spines.values(): spine.set_edgecolor('#2a2a3a')
            st.pyplot(fig)
            plt.close()

            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:13px;color:#666;margin-bottom:8px;letter-spacing:2px;">PREDICTED</div>
                <div class="result-label" style="color:{color_map.get(prediction_csv,'#fff')}">{prediction_csv}</div>
            </div>
            """, unsafe_allow_html=True)

            try:
                proba = model.predict_proba([features_csv])[0]
                classes = model.classes_
                st.markdown("#### Confidence")
                for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
                    st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
            except:
                pass

st.markdown("---")
st.markdown("#### الـ Features")
feat_names = ["Mean", "Std", "Max", "Min", "RMS", "FFT Peak", "FFT Mean", "FFT Std", "Dom Freq"]
cols = st.columns(3)
for i, (name, val) in enumerate(zip(feat_names, features)):
    with cols[i % 3]:
        st.metric(name, f"{val:.3f}" if i < 8 else f"{int(val)} Hz")
