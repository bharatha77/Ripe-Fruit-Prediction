# 🍎 Ripe Fruit Prediction AI Streamlit Web App

An AI-powered fruit ripeness classifier built with TensorFlow & Streamlit. Instantly predict whether a fruit is **Unripe** 🟡, **Ripe** 🟢, or **Overripe** 🔴 with confidence scores, ripeness index meters, and smart storage advice.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
<!--![License](https://img.shields.io/badge/License-MIT-green)-->

---

## 🎯 Supported Fruits

The underlying Deep Learning CNN model is specially trained to detect ripeness stage for:

- 🍎 **Apple**
- 🍌 **Banana**
- 🥭 **Mango**
- 🍊 **Orange**
- 🍅 **Tomato**

---

## 🌟 Key Features

- 📁 **Single Image Upload**: Upload fruit photos (`PNG`, `JPG`, `JPEG`, `WEBP`) for instant evaluation.
- 🍏 **Fruit Sample Gallery**: 1-click test with pre-loaded sample images for Apple, Banana, Mango, and Orange.
- 📊 **Batch Processing**: Process multiple fruit photos simultaneously and download CSV analytical reports.
- 💡 **Storage & Culinary Tips**: Contextual recommendations based on predicted ripeness stage.
- ⚙️ **Cloud Deployment Ready**: Optimized for Streamlit Community Cloud with automatic Keras 3 deserialization compatibility.

---

## 📁 Repository Structure

```
Ripe-Fruit-Prediction/
├── app.py                  # Streamlit Web Application interface
├── model_loader.py         # Primary model loader & TF 2.20.0 compatibility layer
├── utils.py                # Image preprocessing, index calculator & advice engine
├── requirements.txt        # Dependencies list
├── README.md               # Project documentation
├── models/
│   └── fruit_ripe.keras    # Active trained CNN Keras model (Single Model)
├── sample_images/          # Demo sample fruit images
└── .streamlit/
    └── config.toml         # Custom Streamlit dark/emerald theme configuration
```

---

## 💻 Local Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/bharatha77/Ripe-Fruit-Prediction.git
   cd Ripe-Fruit-Prediction
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---
<!--
## ☁️ Deployment on Streamlit Community Cloud

1. Fork or push this repository to GitHub: `https://github.com/bharatha77/Ripe-Fruit-Prediction.git`
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**, select `bharatha77/Ripe-Fruit-Prediction`, set branch to `main`, and main file path to `app.py`.
4. Click **Deploy!** 🚀-->
