# src/machine_learning/predictive_analysis.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from src.data_management import load_pkl_file

# Prefer tflite-runtime on Heroku; fallback to TensorFlow Lite locally (macOS)
try:
    from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter  # uses TF if tflite-runtime isn't available

# Path to the deployed TFLite model
MODEL_PATH = "outputs/step_2/tflite/step_2_bs32_k3_do0.3_act-elu_opt-adamax_seed27.tflite"


def plot_predictions_probabilities(pred_proba, pred_class, key=None):
    """
    Plot prediction probability results (unique key avoids DuplicateElementId).
    """
    prob_per_class = pd.DataFrame(
        data=[0, 0],
        index=['Healthy', 'Mildew'],
        columns=['Probability']
    )
    prob_per_class.loc[pred_class, 'Probability'] = float(pred_proba)
    # complement for the other class
    other_class = 'Healthy' if pred_class == 'Mildew' else 'Mildew'
    prob_per_class.loc[other_class, 'Probability'] = 1 - float(pred_proba)
    prob_per_class['Diagnostic'] = prob_per_class.index
    prob_per_class = prob_per_class.round(3)

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Probability',
        range_y=[0, 1],
        width=600,
        height=300,
        template='seaborn'
    )
    st.plotly_chart(fig, key=key or f"prob_chart_{hash(pred_class) ^ hash(round(float(pred_proba),3))}")


def resize_input_image(img: Image.Image):
    """
    Resize image to match the model input shape and normalize pixel values to [0,1].
    Returns array shaped (1, H, W, 3) as float32.
    """
    image_shape = load_pkl_file(file_path='outputs/v1/image_shape.pkl')  # (H, W, C)
    img_resized = img.resize((int(image_shape[1]), int(image_shape[0])), Image.LANCZOS).convert("RGB")
    arr = np.asarray(img_resized).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def _prepare_input_for_interpreter(x: np.ndarray, input_details: dict) -> np.ndarray:
    """
    Adapt normalized float32 input x in [0,1] to the dtype expected by the TFLite model.
    Supports float32, uint8, int8 models.
    """
    dtype = input_details['dtype']
    scale, zero_point = (0.0, 0) if 'quantization' not in input_details else input_details['quantization']

    if dtype == np.float32:
        return x.astype(np.float32)

    if dtype == np.uint8:
        # Map [0,1] -> [0,255]
        return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)

    if dtype == np.int8:
        # If quant params exist, use them; otherwise map [0,1] to int8 range with zero_point 0
        if scale and scale > 0:
            q = np.round(x / scale + zero_point)
        else:
            q = np.round(x * 255.0 - 128.0)  # rough fallback
        return np.clip(q, -128, 127).astype(np.int8)

    # Fallback: cast to expected dtype
    return x.astype(dtype)


def _postprocess_output(y: np.ndarray, output_details: dict) -> float:
    """
    Return float probability in [0,1] from model output tensor.
    Handles float32, uint8, int8 outputs.
    Assumes single sigmoid output shaped (1, 1) or (1,).
    """
    y = np.array(y).reshape(-1)[0]
    dtype = output_details['dtype']
    scale, zero_point = (0.0, 0) if 'quantization' not in output_details else output_details['quantization']

    if dtype == np.float32:
        return float(y)

    if dtype == np.uint8:
        # Dequantize: real = (q - z) * scale
        if scale and scale > 0:
            return float((y - zero_point) * scale)
        return float(y / 255.0)

    if dtype == np.int8:
        if scale and scale > 0:
            return float((y - zero_point) * scale)
        # Rough fallback from int8 to [0,1]
        return float((y + 128.0) / 255.0)

    # Fallback: best effort cast
    return float(y)


def load_model_and_predict(my_image: np.ndarray):
    """
    Load and perform prediction using the TensorFlow Lite model.
    my_image should be (1,H,W,3) float32 in [0,1] as returned by resize_input_image.
    """
    # Load TFLite interpreter
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Prepare input for model dtype
    x = _prepare_input_for_interpreter(my_image, input_details)

    # Set input tensor and run inference
    interpreter.set_tensor(input_details['index'], x)
    interpreter.invoke()

    # Get raw output and convert to probability
    y_raw = interpreter.get_tensor(output_details['index'])
    pred_proba = _postprocess_output(y_raw, output_details)

    # Map probability to class
    target_map = {0: 'Healthy', 1: 'Mildew'}
    pred_class = target_map[int(pred_proba > 0.5)]
    # Ensure we report the probability of the predicted class
    if pred_class == 'Healthy':
        pred_proba = 1 - pred_proba

    st.write(
        f"The predictive analysis indicates the sample leaf is "
        f"**{pred_class.lower()}** with mildew."
    )

    return float(pred_proba), pred_class
