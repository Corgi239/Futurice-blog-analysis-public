import streamlit as st
import streamlit.components.v1 as components

header = st.container()
with header:
    st.title('Keyword popularity visualization')
    st.write('This page displays the keywords popularity by a chosen period')
    st.markdown('*This is a work in progress, some features might not work correctly...*')
instructions = st.container()
with instructions:
    st.markdown('## How to use')
    st.markdown(
        """
        1. Select the period from the drop box to see the graph.
        2. Choose play to see the graph evolve automatically or pause to see the trend at a certain time point.
        3. Use the slider to manually control the time.
        """
    )


with st.container():
    period = st.selectbox(
        'Period length',
        ('1-month period', '3-month period', '6-month period', '12-month period')
    )
    period_dict = {'1-month period':1, '3-month period':3, '6-month period':6, '12-month period':12}
    # @st.cache
    def fetch_plot():
        return open("reports/figures/interactive/{:d}months_bigrams.html".format(period_dict[period]), 'r')
    html = fetch_plot()
    source_code = html.read() 
    components.html(source_code, height=700)