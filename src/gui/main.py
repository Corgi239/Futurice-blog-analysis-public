import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Project 🅱️log")

st.sidebar.success("Choose your functionality :D")

st.markdown(
    """
    For the course CS-C3250 Data Science Project, Team Futurice bas built the Futurice Blog Analysis Tool

    **👈 Choose your functionality from the sidebar** to see a demo.
    
    ### Pages
    - Visual
    - Keyword Popularity
    - Dynamic Topic Modelling
    - Blog Doctor
"""
)