import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities,
)

def page_mildew_detector_body():
    st.info(
        "* The client is interested in telling whether a given leaf is infected with mildew or not."
    )
    st.write(
        "* You can download a set of infected and healthy leaves for live prediction. "
        "You can download the images from "
        "[here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    images_buffer = st.file_uploader(
        "Upload cherry leaf image samples. You may select more than one.",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if images_buffer:
        df_report = pd.DataFrame([])

        for i, image in enumerate(images_buffer):
            img_pil = Image.open(image).convert("RGB")
            st.info(f"Cherry Leaf Sample: **{image.name}**")

            img_array = np.array(img_pil)
            st.image(
                img_pil,
                caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height",
            )

            # Preprocess and predict
            resized_img = resize_input_image(img=img_pil)
            pred_proba, pred_class = load_model_and_predict(resized_img)

            # Unique key per chart → filename + index + short hash
            unique_key = f"prob_{i}_{abs(hash(image.name)) % 10_000_000}"
            plot_predictions_probabilities(pred_proba, pred_class, key=unique_key)

            # Collect into report
            df_report = df_report._append(
                {"Name": image.name, "Result": pred_class},
                ignore_index=True,
            )

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            # keep compatibility with your existing helper returning markdown
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)
