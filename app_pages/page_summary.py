import streamlit as st

def page_summary_body():
    st.title("🍒 Quick Project Summary")

    st.info("""
### Project Summary
Cherry Powdery Mildew Detector is a machine learning–based web application designed to help identify powdery mildew infection on cherry leaves using image recognition.
""")

    st.info("""
### Powdery Mildew
Powdery mildew is a fungal disease that affects a wide variety of plants, including cherry trees. It is caused by *Podosphaera clandestina* and appears as white, powdery patches on the surface of leaves, stems, and buds.
These patches result from the growth of fungal hyphae and chains of spores (conidia) on the plant’s surface. Infected leaves may curl, discolor, or fall prematurely, leading to reduced photosynthesis, lower fruit yield, and compromised quality.

The disease spreads rapidly under warm, dry conditions with high humidity and poor air circulation — making early and accurate detection essential to prevent large-scale crop loss.
""")

    st.info("""
### Business Case
Farmy & Foods, a major agricultural producer, has reported increasing cases of powdery mildew infecting its cherry plantations. The current manual inspection process—where trained staff spend up to 30 minutes per tree visually checking leaves—is slow, subjective, and unscalable across thousands of trees. The company aims to modernize this process through machine learning and image analysis, enabling faster, more accurate detection of infected leaves. The project’s goal is to develop a binary classification model and an interactive Streamlit dashboard that automates this detection, reduces inspection time, and ensures that only high-quality, disease-free produce reaches the market.
""")

    st.success("""
### Project Dataset
The available dataset contains images of 2104 healthy leaves and 2104 infected cherry leaves.
""")

    st.write("---")
