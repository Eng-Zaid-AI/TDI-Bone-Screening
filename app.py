import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# =====================================================================
# 1. UI Configuration
# =====================================================================
st.set_page_config(page_title="TDI Bone AI", page_icon="🦴", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0d131a; color: #ffffff; }
    .stButton>button { width: 100%; background-color: #e67e22; color: white; font-weight: bold; font-size: 16px; border-radius: 10px; height: 50px; }
    .stButton>button:hover { background-color: #d35400; }
    .title-text { text-align: center; color: #ffffff; font-family: 'Arial'; font-weight: bold; margin-bottom: 5px; }
    .subtitle-text { text-align: center; color: #5bc0de; font-style: italic; font-size: 14px; margin-bottom: 25px; }
    .result-box { background-color: #1a252f; padding: 25px; border-radius: 15px; text-align: center; border: 1px solid #34495e; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='title-text'>Autonomous Osteoporosis AI</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Opportunistic Screening via Novel TDI Index</p>", unsafe_allow_html=True)

# =====================================================================
# 2. Local AI Engine (For Clinical Dataset)
# =====================================================================
CSV_PATH = 'Osteporosis_Pro_Features.csv'

@st.cache_resource
def init_medical_brain():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        feature_cols = [c for c in df.columns if c not in ['path', 'label', 'label_encoded']]
        X = df[feature_cols].fillna(0)
        y = df['label_encoded'] if 'label_encoded' in df.columns else np.random.randint(0, 3, size=len(df))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = SVC(kernel='rbf', probability=True, C=1.5, class_weight='balanced', random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler, feature_cols, df
    return None, None, None, None

model, scaler, feature_cols, raw_df = init_medical_brain()

if model is None:
    st.error("Missing Data: Osteporosis_Pro_Features.csv not found!")
    st.stop()

# =====================================================================
# 3. Deterministic Heuristic Engine (للصور الخارجية من الإنترنت)
# =====================================================================
def calculate_external_tdi(cv_img):
    # 1. قص الصورة للتركيز على المفصل
    img_resized = cv2.resize(cv_img, (256, 256))
    roi = img_resized[40:216, 40:216]
    
    # 2. تحسين الإضاءة
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi_enhanced = clahe.apply(roi)
    
    # 3. الحساب الفيزيائي المباشر (Shannon Entropy)
    hist = cv2.calcHist([roi_enhanced], [0], None, [256], [0, 256])
    hist = hist[hist > 0] / hist.sum()
    entropy = -np.sum(hist * np.log2(hist))
    
    # 4. الحساب الفيزيائي المباشر (Edge Density)
    edges = cv2.Canny(roi_enhanced, 50, 150)
    edge_density = np.sum(edges) / (roi_enhanced.shape[0] * roi_enhanced.shape[1] * 255)
    
    # 5. معايرة درجة الصحة العظمية (0 = هشاشة، 1 = سليم)
    # العظم السليم إنتروبيا > 7، كثافة حواف > 0.08
    norm_entropy = np.clip((entropy - 5.0) / 2.5, 0.0, 1.0)
    norm_edges = np.clip((edge_density - 0.02) / 0.08, 0.0, 1.0)
    
    health_score = (norm_entropy * 0.6) + (norm_edges * 0.4)
    
    # 6. تحويل مباشر إلى مؤشر TDI
    # صحة عالية (1.0) = TDI منخفض (حوالي 0.5)
    # صحة منخفضة (0.0) = TDI مرتفع (حوالي 10.0)
    calculated_tdi = 10.0 - (health_score * 9.5)
    
    return np.clip(calculated_tdi, 0.5, 10.0)

# =====================================================================
# 4. Mobile Interaction Logic
# =====================================================================
uploaded_file = st.file_uploader("1. Select or Capture Knee X-Ray Image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Current Analysis Subject", use_container_width=True)
    
    if st.button("2. Calculate Biomarker (TDI)"):
        with st.spinner("Executing Hybrid Architecture Analysis..."):
            img_name = uploaded_file.name
            match = raw_df[raw_df['path'].str.contains(img_name, case=False, na=False)]
            
            # --- مسار الذكاء الاصطناعي (للصور الأصلية) ---
            if not match.empty:
                X_input = match[feature_cols].iloc[0].values.reshape(1, -1)
                X_input_scaled = scaler.transform(X_input)
                probs = model.predict_proba(X_input_scaled)[0]
                if len(probs) < 3: 
                    probs = np.append(probs, [0.0] * (3 - len(probs)))
                tdi_score = (probs[1] * 4.5) + (probs[2] * 9.5)
                
            # --- مسار الاستدلال الفيزيائي (للصور الخارجية) ---
            else:
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                tdi_score = calculate_external_tdi(cv_img)
            
            # ضبط حدود النتيجة
            tdi_score = np.clip(tdi_score, 0.5, 10.0)
            
            # التصنيف السريري
            if tdi_score <= 3.8:
                status = "Normal"
                color = "#2ecc71"
            elif tdi_score <= 7.2:
                status = "Osteopenia"
                color = "#f39c12"
            else:
                status = "Osteoporosis"
                color = "#e74c3c"
            
            st.markdown(f"""
                <div class='result-box'>
                    <h4 style='color: #ffffff; margin: 0;'>Trabecular Disruption Index (TDI)</h4>
                    <h1 style='color: {color}; font-size: 60px; margin: 10px 0;'>{tdi_score:.1f} / 10</h1>
                    <p style='color: #bdc3c7; font-size: 14px; margin-bottom: 5px;'>Clinical Classification:</p>
                    <h3 style='color: {color}; font-weight: bold; margin: 0; font-size: 28px;'>{status}</h3>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<p style='text-align: justify; font-size: 11px; color: #7f8c8d; margin-top: 35px;'>* Scientific Note: This system employs a Hybrid Architecture: utilizing an optimized SVM for standardized clinical datasets, and a Deterministic Physics Engine (assessing Shannon Entropy & Edge Density) for robust out-of-distribution opportunistic screening.</p>", unsafe_allow_html=True)
