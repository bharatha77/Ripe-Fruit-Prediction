"""
Utility functions for Fruit Ripeness AI
Handles image preprocessing, class definitions, ripeness index computation,
recommendations generation, and batch reporting.
"""

import io
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------
# CLASS DEFINITIONS & MAPPING
# Note: Keras flow_from_directory sorts folders alphabetically:
# Index 0: 'Overipe' / 'Overripe'
# Index 1: 'Ripe'
# Index 2: 'Unripe'
# ---------------------------------------------------------
CLASS_LABELS = ["Overripe", "Ripe", "Unripe"]

CLASS_COLORS = {
    "Overripe": "#ef4444",  # Red
    "Ripe": "#10b981",      # Emerald Green
    "Unripe": "#f59e0b"     # Amber / Yellow
}

CLASS_ICONS = {
    "Overripe": "🔴",
    "Ripe": "🟢",
    "Unripe": "🟡"
}


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(image_input, target_size=(224, 224)):
    """
    Preprocesses an input image for model prediction.
    Ensures RGB color format, resizing to target dimensions,
    and normalization to [0.0, 1.0].
    
    Args:
        image_input: bytes, bytearray, io.BytesIO, or PIL.Image.Image
        target_size: tuple of (width, height), default (224, 224)
        
    Returns:
        tuple: (img_array, img_pil)
            - img_array: numpy tensor with shape (1, 224, 224, 3) in [0.0, 1.0]
            - img_pil: original PIL Image in RGB format
    """
    if isinstance(image_input, (bytes, bytearray)):
        img_pil = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, io.BytesIO):
        img_pil = Image.open(image_input)
    elif isinstance(image_input, str):
        img_pil = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img_pil = image_input
    else:
        # Fallback for file-like objects
        img_pil = Image.open(image_input)

    # Ensure RGB color space (strips alpha channel, converts grayscale or CMYK)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    # High quality resize
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    resized_img = img_pil.resize(target_size, resample=resample_filter)

    # Convert to float32 array normalized to [0, 1]
    img_array = np.array(resized_img, dtype=np.float32) / 255.0

    # Add batch dimension: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, img_pil


# ---------------------------------------------------------
# RIPENESS METRICS & RECOMMENDATIONS
# ---------------------------------------------------------
def calculate_ripeness_index(probs):
    """
    Calculates a continuous ripeness index from 0 to 100 based on class probabilities.
    
    Classes:
        probs[0] = Overripe (high ripeness, score ~95)
        probs[1] = Ripe (ideal ripeness, score ~70)
        probs[2] = Unripe (early ripeness, score ~15)
        
    Returns:
        float: Ripeness score between 0 and 100
    """
    p_overripe = float(probs[0])
    p_ripe = float(probs[1])
    p_unripe = float(probs[2])

    index = (p_unripe * 15.0) + (p_ripe * 70.0) + (p_overripe * 98.0)
    return float(np.clip(index, 0.0, 100.0))


def get_ripeness_recommendations(pred_label, confidence):
    """
    Returns culinary and storage guidance based on predicted ripeness.
    """
    conf_pct = f"{confidence * 100:.1f}%"

    if pred_label == "Unripe":
        return {
            "status_title": f"🟡 Unripe ({conf_pct})",
            "badge_color": CLASS_COLORS["Unripe"],
            "eating_window": "Wait 3 to 5 days before consuming.",
            "storage_advice": "Store at room temperature in a well-ventilated area away from direct sunlight. To speed up ripening, place in a paper bag with an apple or banana.",
            "culinary_uses": "Great for pickling, tart salads, curries, or baking where firm texture is desired."
        }
    elif pred_label == "Ripe":
        return {
            "status_title": f"🟢 Perfectly Ripe ({conf_pct})",
            "badge_color": CLASS_COLORS["Ripe"],
            "eating_window": "Best consumed within 1 to 2 days for optimal taste and nutrition.",
            "storage_advice": "Store in the refrigerator crisper drawer to maintain peak freshness and prevent overripening.",
            "culinary_uses": "Ideal for fresh eating, fruit bowls, desserts, juices, and gourmet garnishes."
        }
    else:  # Overripe
        return {
            "status_title": f"🔴 Overripe ({conf_pct})",
            "badge_color": CLASS_COLORS["Overripe"],
            "eating_window": "Use immediately within 24 hours.",
            "storage_advice": "Peel and slice, then store in an airtight container in the freezer for long-term smoothie use.",
            "culinary_uses": "Perfect for baking (banana bread, muffins), sweet jams, fruit purees, smoothies, or natural sweetening."
        }


# ---------------------------------------------------------
# BATCH ANALYSIS HELPERS
# ---------------------------------------------------------
def generate_batch_df(batch_results):
    """
    Converts list of batch prediction results into a structured DataFrame.
    """
    rows = []
    for item in batch_results:
        probs = item.get("probs", [0.0, 0.0, 0.0])
        rows.append({
            "File Name": item.get("filename", "Unknown"),
            "Predicted Stage": item.get("predicted_label", "Unknown"),
            "Confidence": f"{item.get('confidence', 0.0) * 100:.1f}%",
            "Ripeness Index (/100)": f"{item.get('ripeness_index', 0.0):.0f}",
            "Overripe Prob": f"{probs[0] * 100:.1f}%",
            "Ripe Prob": f"{probs[1] * 100:.1f}%",
            "Unripe Prob": f"{probs[2] * 100:.1f}%"
        })
    return pd.DataFrame(rows)
