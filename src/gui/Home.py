import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Futurice blog analysis demo")

st.sidebar.success("Choose your functionality")

st.markdown(
    """
    For the course CS-C3250 Data Science Project, Team Futurice bas built the Futurice Blog Analysis Tool

    **👈 Choose your functionality from the sidebar** to see a demo.
    
    ### Pages

    * General Dashboard
        * Explore features of existing blogs with customizable filters
        * Spot interesting trends and feature interactions
    * Blog Doctor
        * Analyze the features of your next blog
        * Check how your next blog lines up against existing popular blogs
    * Dynamic Topic Modelling
        * Discover topic structures within Futurice's blog collection
    * Keyword Popularity
        * View an interactive summary of how the popularity of different keywords changed over time

"""
)