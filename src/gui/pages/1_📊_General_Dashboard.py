import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import altair as alt
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from altair import datum
from bokeh.palettes import Category20
# import trend_graph



GOOGLE_ANALYTICS_START_DATE = datetime.date(2019,9,13)

# Method for importing data from a local CSV
@st.cache
def fetch_data():
    df = pd.read_csv(
        'data/final/futurice_blog_data.csv', 
        sep='\t', 
        parse_dates=['time'],
        date_parser=lambda col: pd.to_datetime(col))
    df['year'] = df.time.dt.year.apply(np.round).astype('Int64').astype(str)
    return df

# Page configuration
st.set_page_config(
    layout='wide'
)

######## FILTERING MENU ########
data = fetch_data()

# Configuration of plottable features
numeric_features = data.select_dtypes(include=['number', 'datetime']).columns
categorical_features = ['category', 'year']

# Define display names for features
feature_labels = {
        'pageviews' : 'Page views',
        'unique_pageviews' : 'Unique page views',
        'avg_time' : 'Average view time',
        'bounce_rate' : 'Bounce rate',
        'exit%' : 'Exit percent',
        'time' : 'Release date',
        'category' : 'Blog category (topic)',
        'mmr_lift' : 'Relevance score'
    }

st.header('Blog Data Exploration')

category_checkboxes = {}
with st.expander("Filters") as filtering:
    (data_filters, settings) = st.columns(2, gap='large')
    with data_filters:
        # Reduce line distancing
        st.markdown("""
            <style>
            [data-testid=stVerticalBlock]{
                gap: 0.5rem;
            }
            </style>
            """, unsafe_allow_html=True)
        st.markdown('##### Select the categories of posts to be displayed')

        # Create a checkbox for every category as well as "(de)select" all buttons
        cats = sorted(data.category.dropna().unique())
        def select_all():
            for cat in cats:
                st.session_state[f'checkbox_{cat}'] = True
        def deselect_all():
            for cat in cats:
                st.session_state[f'checkbox_{cat}'] = False
        for cat in cats:
            checkbox = st.checkbox(label=cat, key=f'checkbox_{cat}', value=True)
            category_checkboxes[cat] = checkbox
        with st.container() as buttons:
            sel = st.button('Select all', on_click=select_all)
            desel = st.button('Deselect all', on_click=deselect_all)

        # A few empty lines for spacing
        for i in range(3):
            st.write("\n")

        with st.container() as time_interval_selection:
            min_date = (data['time'].dropna().min())
            max_date = (data['time'].dropna().max())
            st.write('##### Posting time interval')
            date_slider = st.slider(
                label='',
                min_value=min_date.to_pydatetime().date(), 
                max_value=max_date.to_pydatetime().date(),
                value=[GOOGLE_ANALYTICS_START_DATE, max_date.to_pydatetime().date()]
            )
            if date_slider[0] < GOOGLE_ANALYTICS_START_DATE:
                st.error("Google analytics data for blogs released before 13-09-2019 might not be accurate")
    with settings:
        warning_labels = ['pageviews', 'unique_pageviews', 'avg_time', 'bounce_rate', 'exit%']
        st.markdown('##### Select the features for the axes')
        xaxis = st.selectbox(
            label='X-axis: ', 
            options=numeric_features, 
            index=0, 
            format_func=lambda x:feature_labels.get(x, x.capitalize())
        )
        if (xaxis in warning_labels):
            st.warning("Authentic Google analytics data can not be presented in a public demo. Analytics have been replaced with randomized data.")
        yaxis = st.selectbox(
            label='Y-axis: ', 
            options=numeric_features, 
            index=3, 
            format_func=lambda x:feature_labels.get(x, x.capitalize())
        )
        if (yaxis in warning_labels):
            st.warning("Authentic Google analytics data can not be presented in a public demo. Analytics have been replaced with randomized data.")
        hist_axis = st.selectbox(
            label='Selection categorical axis: ', 
            options=categorical_features, 
            index=0, 
            format_func=lambda x:feature_labels.get(x, x.capitalize())
        )
        for i in range(3):
                st.write("\n")
        show_fit = st.checkbox(label='Show fit', value=True)
        fit_mode = st.selectbox(
            label='Fit mode',
            options=['Single fit', 'Categorical fit'],
            disabled=not show_fit
        )
        fit_smoothness = st.slider(
            label='Fit smoothness',
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            disabled=not show_fit
        )

        
######## PLOTS #########

with st.container() as interactive_plot:   
  
    # Determine which categories are selected
    selected_categories = set()
    for k, v in category_checkboxes.items():
        if v: selected_categories.add(k)
    
    # Apply filters to data
    all_data = data.dropna()
    data = all_data.loc[(data.time >= pd.Timestamp(date_slider[0])) & (data.time <= pd.Timestamp(date_slider[1]))]
    data = data.loc[data.category.isin(selected_categories)]

    # Define interactive selections
    brush = alt.selection(type='interval')
    mouseover = alt.selection_single(on='mouseover', nearest=False, empty='none', clear='mouseout')
    
    # Define persistent colors for categories
    color_scales = {}
    for feature in categorical_features:
        all_categories = sorted(all_data[feature].dropna().unique())
        present_categories = list(sorted(data[feature].unique()))
        palette = Category20[len(all_categories)]
        selected_colors = [color for (cat, color) in zip(all_categories, palette) if cat in present_categories]
        color_scale = alt.Scale(domain=present_categories, range=selected_colors)
        color_scales[feature] = color_scale

    # Define display parameters for axes
    feature_axis_params = {
        'pageviews' : {
            'shorthand' : 'pageviews:Q',
            'scale' : alt.Scale(type='symlog'),
            'title' : 'Page views'
        },
        'unique_pageviews' : {
            'shorthand' : 'unique_pageviews:Q',
            'scale' : alt.Scale(type='log'),
            'title' : 'Unique page views'
        },
        'avg_time' : {
            'shorthand' : 'avg_time:Q',
            'scale' : alt.Scale(type='symlog', constant=5),
            'title' : 'Average view time [sec]'
        },
        'bounce_rate' : {
            'shorthand' : 'bounce_rate:Q',
            'scale' : alt.Scale(type='linear'),
            'title' : 'Bounce rate'
        },
        'exit%' : {
            'shorthand' : 'exit%:Q',
            'scale' : alt.Scale(type='linear'),
            'title' : 'Exit percent'
        },
        'time' : {
            'shorthand' : 'time:T',
            'timeUnit' : 'yearmonthdate',
            'axis' : alt.Axis(format='%b-%Y'),
            'scale' : alt.Scale(type='time'),
            'title' : 'Release date'
        },
        'category' : {
            'if_true' : alt.Color(f'category:N', scale=color_scales['category']),
            'if_false': alt.value('lightgray')
        }
    }

    # Define default axis parameters
    def default_numerical_axis_params(axis_name):
        return {
            'shorthand' : f'{axis_name}:Q',
            'scale' : alt.Scale(type='linear'),
            'title' : axis_name.capitalize()
        }
    def default_categorical_axis_params(axis_name):
        return {
            'if_true' : alt.Color(f'{axis_name}:N', scale=color_scales[axis_name]),
            'if_false': alt.value('lightgray')
        }

    # Create the engagement chart
    base = alt.Chart(data)
    engagement = base.mark_point(opacity=0.9).transform_calculate(
        href = 'https://futurice.com/' + alt.datum.url,
        trunc_blog_title = alt.expr.truncate(alt.datum.title, 90)
    ).encode(
        x=alt.X(**feature_axis_params.get(xaxis, default_numerical_axis_params(xaxis))),
        y=alt.Y(**feature_axis_params.get(yaxis, default_numerical_axis_params(yaxis))),
        color=alt.condition(
            predicate=brush, 
            **feature_axis_params.get(hist_axis, default_categorical_axis_params(hist_axis))
        ),
        size=alt.condition(mouseover, alt.value(30), alt.value(5)),
        tooltip=[
            alt.Tooltip('trunc_blog_title:N', title='blog title'),
            alt.Tooltip('category:N'),
            alt.Tooltip('time:T', title='release date'),
            alt.Tooltip('pageviews', format='.0f'),
            alt.Tooltip('avg_time', format='.0f', title='average view time'),
            alt.Tooltip('exit%', format='.1%', title='exit percent'),
            alt.Tooltip('bounce_rate', format='.2f', title='bounce_rate')
        ],
        href = 'href:N'
    ).add_selection(
        brush,
        mouseover
    ).properties(
        height = 500,
        width = 900
    )

    # Define parameters for different fit options
    match fit_mode:
        case 'Single fit':
            grouping = []
            colouring = alt.value('grey')
        case 'Categorical fit':
            grouping = [hist_axis]
            colouring = f'{hist_axis}:N'
        case _:
            grouping = []
            colouring = alt.value('grey')

    fit = alt.Chart(data).transform_loess(
        f'{xaxis}', 
        f'{yaxis}',
        bandwidth = fit_smoothness,
        groupby=grouping
    ).mark_line().encode(
        x=alt.X(**feature_axis_params.get(xaxis, default_numerical_axis_params(xaxis))),
        y=alt.Y(**feature_axis_params.get(yaxis, default_numerical_axis_params(yaxis))),
        # color = alt.value('grey'),
        color = colouring,
        strokeWidth = alt.value(2),
        opacity = alt.value(0.5)
    )

    engagement_fit = engagement
    if show_fit:
        engagement_fit = engagement_fit + fit

    # Create histogram based on selection
    bars = alt.Chart(data).mark_bar().encode(
        y=alt.Y(f'{hist_axis}:N'),
        color=f'{hist_axis}:N',
        x=f'count({hist_axis}):Q'
    ).transform_filter(
        brush
    ).properties(
        width = 900
    )

    # Configure blogs to open in new tabs
    chart = engagement_fit & bars
    chart.usermeta = {
        "embedOptions": {
            'loader': {'target': '_blank'}
        }
    }

    # Display the chart
    chart
    
