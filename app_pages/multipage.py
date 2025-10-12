import streamlit as st

class MultiPage:
    """Class for managing multiple Streamlit pages"""
    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

    def add_page(self, title, func):
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.sidebar.title(self.app_name)
        page = st.sidebar.radio("Navigation", self.pages, format_func=lambda p: p["title"])
        page["function"]()