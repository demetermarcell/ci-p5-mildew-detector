import streamlit as st
import os

def _img(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing image: `{path}`")

def page_project_hypothesis_body():
    st.title("🔬 Project Hypotheses & Validation")

    # --- Hypotheses -----------------------------------------------------------------
    st.info(
        "### 1) Visual Difference Hypothesis\n"
        "Images of cherry leaves infected with powdery mildew will show distinct visual characteristics—"
        "such as white, powdery textures and reduced leaf coloration—compared to healthy leaves."
    )

    st.info(
        "### 2) Predictive Capability Hypothesis\n"
        "A binary image classification model trained on labeled cherry leaf images can accurately distinguish "
        "between healthy and infected leaves, achieving ≥97% accuracy on a held-out test set."
    )

    st.write("---")

    # --- Validation: Visual Differences ---------------------------------------------
    st.success("### 1) Validation of Visual Difference Hypothesis")
    st.write(
        "A computational pipeline loaded, resized, and normalized leaf images for consistent pixel-level "
        "comparison across samples. Mean and average representations were generated for both classes "
        "to highlight overall texture and color patterns."
    )
    st.write(
        "Processing **30 images/label** took **0.6 s**; increasing to **100** and **200** images raised runtimes to "
        "**8.0 s** and **38.4 s**, respectively. The results confirm a clear visual distinction: infected samples "
        "exhibit more surface irregularities and discoloration. However, as the number of averaged images increases, "
        "the contrast between mean representations visually diminishes due to pixel-level smoothing."
    )
    st.write(
        "These visualizations align with prior research and demonstrate that the differences can be **quantified "
        "computationally**, not just observed by eye."
    )

    st.caption("Difference between average healthy and average infected leaves")
    _img("outputs/v1/avg_diff.png", caption="Average difference (healthy vs infected)")

    st.write("#### Average and Variability (v1, Sample size 30)")
    col1, col2 = st.columns(2)
    with col1:
        _img("outputs/v1/avg_var_healthy.png", caption="v1 — Healthy: average & variability")
    with col2:
        _img("outputs/v1/avg_var_powdery_mildew.png", caption="v1 — Infected: average & variability")

    st.write("#### Average and Variability (v2, Sample size 200)")
    col3, col4 = st.columns(2)
    with col3:
        _img("outputs/v2/avg_var_healthy.png", caption="v2 — Healthy: average & variability")
    with col4:
        _img("outputs/v2/avg_var_powdery_mildew.png", caption="v2 — Infected: average & variability")

    st.write("#### Average and Variability (v3, Sample size 100)")
    col5, col6 = st.columns(2)
    with col5:
        _img("outputs/v3/avg_var_healthy.png", caption="v3 — Healthy: average & variability")
    with col6:
        _img("outputs/v3/avg_var_powdery_mildew.png", caption="v3 — Infected: average & variability")

    st.write("---")

    # --- Validation: Predictive Capability ------------------------------------------
    st.success("### 2) Validation of Predictive Capability Hypothesis")
    st.write(
        "We used a stratified train/validation/test split with strict folder isolation (no augmentation leakage), "
        "early stopping, and best-checkpoint selection on validation F1. The final model was evaluated once on the "
        "untouched test set: it achieved **100% accuracy**, with **Precision = 1.00**, **Recall = 1.00**, "
        "**F1 = 1.00**, and a confusion matrix showing **zero misclassifications** across both classes. "
        "These results meet—and exceed—the ≥97% accuracy criterion, confirming the model reliably separates "
        "healthy from infected leaves."
    )

