"""
Model Loader for Fruit Ripeness Prediction
Handles loading trained models across multiple formats (.keras, .h5, weights-only)
and provides safe inference functions.
"""

import os
import numpy as np

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def build_base_architecture():
    """
    Rebuilds the exact CNN architecture defined during training.
    Used as an ultra-reliable fallback when loading weights directly.
    """
    from keras.models import Sequential
    from keras.layers import (
        Input,
        Conv2D,
        BatchNormalization,
        MaxPooling2D,
        GlobalAveragePooling2D,
        Dense,
        Dropout
    )

    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(3, activation='softmax')
    ])
    return model


def load_primary_model(models_dir=None):
    """
    Loads the trained model from the models directory.
    Attempts formats in order of stability:
    1. fruit_ripe_fixed.keras (re-serialized clean Keras 3 format)
    2. fruit_ripe.weights.h5 (rebuilding architecture + loading weights)
    3. fruit_ripe.keras (standard load)
    4. fruit_ripe.h5 (legacy HDF5)
    
    Returns:
        tuple: (model, model_filename)
    """
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR

    import keras

    # 1. Try improved transfer learning model if available
    improved_path = os.path.join(models_dir, "fruit_ripe_improved.keras")
    if os.path.exists(improved_path):
        try:
            model = keras.models.load_model(improved_path)
            return model, "fruit_ripe_improved.keras (MobileNetV2)"
        except Exception:
            pass

    # 2. Try clean fixed keras file
    fixed_path = os.path.join(models_dir, "fruit_ripe_fixed.keras")
    if os.path.exists(fixed_path):
        try:
            model = keras.models.load_model(fixed_path)
            return model, "fruit_ripe_fixed.keras"
        except Exception:
            pass

    # 2. Try loading weights into rebuilt architecture
    weights_path = os.path.join(models_dir, "fruit_ripe.weights.h5")
    if os.path.exists(weights_path):
        try:
            model = build_base_architecture()
            model.load_weights(weights_path)
            return model, "fruit_ripe.weights.h5 (Reconstructed)"
        except Exception:
            pass

    # 3. Try standard fruit_ripe.keras
    keras_path = os.path.join(models_dir, "fruit_ripe.keras")
    if os.path.exists(keras_path):
        try:
            model = keras.models.load_model(keras_path)
            return model, "fruit_ripe.keras"
        except Exception:
            pass

    # 4. Try legacy fruit_ripe.h5
    h5_path = os.path.join(models_dir, "fruit_ripe.h5")
    if os.path.exists(h5_path):
        try:
            model = keras.models.load_model(h5_path)
            return model, "fruit_ripe.h5"
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not load any valid fruit ripeness model from {models_dir}. "
        "Please ensure fruit_ripe_fixed.keras or fruit_ripe.weights.h5 exists."
    )


# Training class weights used in mini-project.ipynb:
# Overripe: 1.904455, Ripe: 0.809491, Unripe: 0.806730
CLASS_WEIGHT_PRIORS = np.array([1.904455, 0.809491, 0.806730], dtype=np.float32)


def predict_fruit_ripeness(model, img_array, calibrate_imbalance=True):
    """
    Runs inference on preprocessed image array.
    
    Args:
        model: Loaded Keras model
        img_array: Preprocessed image tensor with shape (1, 224, 224, 3) or (224, 224, 3)
        calibrate_imbalance: Whether to de-bias the predictions using inverse training class weights
        
    Returns:
        numpy.ndarray: 1D array of 3 probabilities [P(Overripe), P(Ripe), P(Unripe)]
    """
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    # Ensure float32
    img_array = img_array.astype(np.float32)

    raw_preds = model.predict(img_array, verbose=0)
    probs = np.squeeze(raw_preds)

    # Ensure valid probability distribution (sum to 1.0)
    if probs.ndim == 0:
        probs = np.array([probs])

    # De-bias artificially inflated Overripe logits from training class_weight
    if calibrate_imbalance and len(probs) == 3:
        probs = probs / CLASS_WEIGHT_PRIORS

    probs_sum = np.sum(probs)
    if probs_sum > 0:
        probs = probs / probs_sum

    return probs

