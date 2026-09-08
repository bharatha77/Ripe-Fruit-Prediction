"""
Fruit Ripeness Prediction - Streamlit Web Application
Trained on Supported Fruits: Apple, Banana, Mango, Orange, Tomato
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Custom modules
from model_loader import (
    load_primary_model,
    predict_fruit_ripeness,
    DEFAULT_MODELS_DIR
)
from utils import (
    preprocess_image,
    calculate_ripeness_index,
    get_ripeness_recommendations,
    generate_batch_df,
    CLASS_LABELS,
    CLASS_COLORS,
    CLASS_ICONS
)

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fruit Ripeness AI Predictor",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 50%, #0d1117 100%);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .header-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Supported Fruits Badge Container */
    .fruits-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }
    
    .fruit-badge {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.88rem;
        color: #f1f5f9;
        font-weight: 600;
    }

    /* Sidebar Content Styling */
    .sidebar-section {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }

    .sidebar-section-title {
        color: #38bdf8;
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 10px 22px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.25rem;
        color: white;
        text-shadow: 0 2px 4px rgba(0,0,0,0.4);
        margin-bottom: 16px;
    }
    
    /* Metric Box */
    .metric-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Advice Card */
    .advice-card {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
    }

    .advice-title {
        font-weight: 700;
        color: #f1f5f9;
        font-size: 0.95rem;
    }

    .advice-body {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    
    /* Sidebar Tweaks */
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# CACHED MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_app_model():
    """Loads and caches the trained fruit ripeness CNN model."""
    return load_primary_model(DEFAULT_MODELS_DIR)


# ---------------------------------------------------------
# MAIN APP FUNCTION
# ---------------------------------------------------------
def main():
    # Header Banner
    st.markdown("""
    <div class="header-banner">
        <h1 class="header-title">🍎 Fruit Ripeness AI Predictor</h1>
        <p class="header-subtitle">
            Upload an image of a fruit to instantly check whether it is <b>Unripe</b>, <b>Ripe</b>, or <b>Overripe</b>.
        </p>
        <div class="fruits-badge-container">
            <span class="fruit-badge">🍎 Apple</span>
            <span class="fruit-badge">🍌 Banana</span>
            <span class="fruit-badge">🥭 Mango</span>
            <span class="fruit-badge">🍊 Orange</span>
            <span class="fruit-badge">🍅 Tomato</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # SIDEBAR: USER-FRIENDLY HELP & SUPPORTED FRUITS
    # -----------------------------------------------------
    with st.sidebar:
        st.markdown("## 🍓 Fruit Ripeness AI")
        st.caption("AI-Powered Fruit Quality Checker")

        # 1. Supported Fruits Card
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">🎯 Supported Fruits</div>
            <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                This AI model is specially trained to detect ripeness for:
                <ul style="margin-top: 6px; padding-left: 18px; color: #f8fafc;">
                    <li>🍎 <b>Apple</b></li>
                    <li>🍌 <b>Banana</b></li>
                    <li>🥭 <b>Mango</b></li>
                    <li>🍊 <b>Orange</b></li>
                    <li>🍅 <b>Tomato</b></li>
                </ul>
                <span style="font-size: 0.8rem; color: #94a3b8; font-style: italic;">
                    *Only model trained on these fruits.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Ripeness Stages Guide
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">🌈 Ripeness Stages</div>
            <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                <p style="margin-bottom: 8px;">🟡 <b>Unripe</b>: Firm, lower sugar. Needs 3-5 days to ripen.</p>
                <p style="margin-bottom: 8px;">🟢 <b>Ripe</b>: Peak flavor & sweetness. Ready to eat today!</p>
                <p style="margin-bottom: 0;">🔴 <b>Overripe</b>: Very soft. Best for baking or smoothies.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. How to Use
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">💡 How To Use</div>
            <ol style="font-size: 0.85rem; color: #cbd5e1; padding-left: 18px; margin: 0; line-height: 1.5;">
                <li>Upload a fruit photo or pick a sample.</li>
                <li>View predicted stage & probability score.</li>
                <li>Get storage & culinary recommendations.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # Load Model quietly in background
        try:
            model, model_filename = load_app_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            model = None

    if model is None:
        st.warning("Model file missing. Please ensure `models/fruit_ripe.keras` exists.")
        return

    # -----------------------------------------------------
    # NAVIGATION TABS (Only 3 clean tabs)
    # -----------------------------------------------------
    tab_single, tab_samples, tab_batch = st.tabs([
        "📁 Upload Fruit Image",
        "🍏 Sample Fruit Gallery",
        "📊 Batch Image Analysis"
    ])

    # -----------------------------------------------------
    # TAB 1: SINGLE IMAGE UPLOAD
    # -----------------------------------------------------
    with tab_single:
        col_input, col_output = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("### Upload Photo of Fruit")
            st.caption("Upload a photo of an Apple, Banana, Mango, Orange, or Tomato")
            uploaded_file = st.file_uploader(
                "Choose an image (PNG, JPG, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                key="single_uploader"
            )

            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                img_array, img_pil = preprocess_image(image_bytes)
                st.image(img_pil,caption=f"Uploaded: {uploaded_file.name}",use_column_width=True)


        with col_output:
            if uploaded_file is not None:
                st.markdown("### 🎯 Ripeness Result")
                with st.spinner("Analyzing fruit ripeness..."):
                    probs = predict_fruit_ripeness(model, img_array)
                    pred_idx = int(np.argmax(probs))
                    pred_label = CLASS_LABELS[pred_idx]
                    confidence = float(probs[pred_idx])
                    ripeness_index = calculate_ripeness_index(probs)
                    rec = get_ripeness_recommendations(pred_label, confidence)

                # Status Badge
                st.markdown(f"""
                <div class="status-badge" style="background-color: {rec['badge_color']};">
                    {rec['status_title']} ({confidence*100:.1f}% Confidence)
                </div>
                """, unsafe_allow_html=True)

                # Metrics Row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{pred_label}</div>
                        <div class="metric-label">Stage</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{confidence*100:.1f}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{ripeness_index:.0f}/100</div>
                        <div class="metric-label">Ripeness Index</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br/>", unsafe_allow_html=True)
                st.subheader("📈 Class Confidence")
                for i, class_name in enumerate(CLASS_LABELS):
                    prob = float(probs[i])
                    st.write(f"**{CLASS_ICONS[class_name]} {class_name}**: `{prob*100:.1f}%`")
                    st.progress(prob)

                st.markdown("---")
                st.subheader("💡 Storage & Usage Tips")
                st.markdown(f"""
                <div class="advice-card">
                    <div class="advice-title">🕒 Eating Window</div>
                    <div class="advice-body">{rec['eating_window']}</div>
                </div>
                <div class="advice-card" style="border-left-color: #f59e0b;">
                    <div class="advice-title">📦 Storage Advice</div>
                    <div class="advice-body">{rec['storage_advice']}</div>
                </div>
                <div class="advice-card" style="border-left-color: #10b981;">
                    <div class="advice-title">🍳 Culinary Ideas</div>
                    <div class="advice-body">{rec['culinary_uses']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("👆 Upload a fruit photo on the left to analyze its ripeness.")

    # -----------------------------------------------------
    # TAB 2: SAMPLE GALLERY
    # -----------------------------------------------------
    with tab_samples:
        st.markdown("### 🍏 Fruit Sample Gallery")
        st.write("Click on any fruit sample below to test the prediction model:")

        samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
        sample_items = [
            ("🍎 Apple Sample", os.path.join(samples_dir, "apple_sample.png")),
            ("🍌 Banana (Ripe)", os.path.join(samples_dir, "ripe_sample.png")),
            ("🥭 Mango Sample", os.path.join(samples_dir, "mango_sample.png")),
            ("🍊 Orange Sample", os.path.join(samples_dir, "orange_sample.png")),
            ("🍌 Banana (Unripe)", os.path.join(samples_dir, "unripe_sample.png")),
            ("🍌 Banana (Overripe)", os.path.join(samples_dir, "overripe_sample.png"))
        ]

        # Display grid of 3 columns
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]

        for idx, (sample_title, sample_path) in enumerate(sample_items):
            target_col = columns[idx % 3]
            with target_col:
                st.markdown(f"#### {sample_title}")
                if os.path.exists(sample_path):
                    st.image(sample_path, use_column_width=True)
                    if st.button(f"Analyze {sample_title}", key=f"btn_sample_{idx}"):
                        img_pil = Image.open(sample_path)
                        img_array, _ = preprocess_image(img_pil)
                        probs = predict_fruit_ripeness(model, img_array)
                        pred_idx = int(np.argmax(probs))
                        pred_label = CLASS_LABELS[pred_idx]
                        confidence = float(probs[pred_idx])
                        rec = get_ripeness_recommendations(pred_label, confidence)

                        st.markdown(f"""
                        <div style="background: {rec['badge_color']}; padding: 10px; border-radius: 8px; color: white; text-align: center; font-weight: bold; margin-top: 10px;">
                            {pred_label} ({confidence*100:.1f}%)
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Sample image missing.")

    # -----------------------------------------------------
    # TAB 3: BATCH PROCESSING
    # -----------------------------------------------------
    with tab_batch:
        st.markdown("### 📊 Batch Image Analysis")
        st.write("Upload multiple fruit images at once to process in bulk and export CSV reports.")

        batch_files = st.file_uploader(
            "Select multiple fruit images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="batch_uploader"
        )

        if batch_files:
            st.write(f"Processing **{len(batch_files)}** images...")
            batch_results = []

            progress_bar = st.progress(0)
            for idx, b_file in enumerate(batch_files):
                img_bytes = b_file.read()
                img_array, _ = preprocess_image(img_bytes)
                probs = predict_fruit_ripeness(model, img_array)
                pred_idx = int(np.argmax(probs))
                pred_label = CLASS_LABELS[pred_idx]
                confidence = float(probs[pred_idx])
                ripeness_index = calculate_ripeness_index(probs)

                batch_results.append({
                    "filename": b_file.name,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "probs": probs,
                    "ripeness_index": ripeness_index
                })
                progress_bar.progress((idx + 1) / len(batch_files))

            df_batch = generate_batch_df(batch_results)
            st.markdown("#### Batch Summary Table")
            st.dataframe(df_batch, use_container_width=True)

            # Export CSV
            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Ripeness Report (CSV)",
                data=csv_data,
                file_name="fruit_ripeness_batch_report.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
