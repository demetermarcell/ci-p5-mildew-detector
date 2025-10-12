import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation():
    return load_pkl_file('outputs/step_2/reports/eval_step_2_bs32_k3_do0.3_act-elu_opt-adamax_seed27.pkl')
