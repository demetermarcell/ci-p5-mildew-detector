import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file


def plot_predictions_probabilities(pred_proba, pred_class, key=None):
    """
    Plot prediction probability results (unique key avoids DuplicateElementId).
    """
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    prob_per_class = pd.DataFrame(
        data=[0, 0],
        index={'Healthy': 0, 'Mildew': 1}.keys(),
        columns=['Probability']
    )
    prob_per_class.loc[pred_class] = pred_proba
    for x in prob_per_class.index.to_list():
        if x not in pred_class:
            prob_per_class.loc[x] = 1 - pred_proba
    prob_per_class = prob_per_class.round(3)
    prob_per_class['Diagnostic'] = prob_per_class.index

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y=prob_per_class['Probability'],
        range_y=[0, 1],
        width=600, height=300, template='seaborn'
    )
    st.plotly_chart(fig, key=key or f"prob_chart_{hash(pred_class) ^ hash(round(float(pred_proba),3))}")

def resize_input_image(img):
    """
    Reshape image to average image size
    """
    image_shape = load_pkl_file(file_path='outputs/v1/image_shape.pkl')
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    my_image = np.expand_dims(img_resized, axis=0)/255

    return my_image


def load_model_and_predict(my_image):
    """
    Load and perform ML prediction over live images
    """

    model = load_model('outputs/step_2/models/step_2_bs32_k3_do0.3_act-elu_opt-adamax_seed27.keras')

    pred_proba = model.predict(my_image)[0, 0]

    target_map = {v: k for k, v in {'Healthy': 0, 'Mildew': 1}.items()}
    pred_class = target_map[pred_proba > 0.5]
    if pred_class == target_map[0]:
        pred_proba = 1 - pred_proba

    st.write(
        f"The predictive analysis indicates the sample leaf is "
        f"**{pred_class.lower()}** with mildew.")

    return pred_proba, pred_class