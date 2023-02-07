import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit.components.v1 as components
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
import altair as alt
from altair import datum
import re
import textstat

# Needed for mmr_lift_score
from pytrends.request import TrendReq 
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity 
from datetime import date, timedelta

# Page configuration
st.set_page_config(
    layout='wide'
)
@st.cache
def download_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')
    nltk.download(["vader_lexicon"])
download_nltk()
@st.cache
def fetch_data():
    df = pd.read_csv(
        'data/final/futurice_blog_data.csv', 
        sep='\t', 
        parse_dates=['time'],
        date_parser=lambda col: pd.to_datetime(col))
    df['year'] = df.time.dt.year.apply(np.round).astype('Int64').astype(str)
    return df
data = fetch_data()

# Helper function to calculate the text length
def text_length(text):
    sentences = sent_tokenize(text)

    sentences = [re.sub(pattern="\d+[.]",repl="", string=sent.strip()) for sent in sentences]
    sentences = [re.sub(pattern="[^a-zA-Z0-9\s]",repl="", string=sent) for sent in sentences]

    ## Filter out the strings that only contains a white space
    no_white_space = [ sent.strip().replace('\r', '.').replace('\n', '.').split('.') for sent in sentences if sent != "" ]
    no_white_space = [sentence.strip() for sentences in no_white_space for sentence in sentences if sentence != '']

    splitted = [ [char for char in sent.split(" ") if char != ""] for sent in no_white_space]
    return np.sum([len(chunk) for chunk in splitted])

# Helper function to calculate average sentence length
def avg_sent_length(text):
    sentences = sent_tokenize(text)

    sentences = [re.sub(pattern="\d+[.]",repl="", string=sent.strip()) for sent in sentences]
    sentences = [re.sub(pattern="[^a-zA-Z0-9\s]",repl="", string=sent) for sent in sentences]

    ## Filter out the strings that only contains a white space
    no_white_space = [ sent.strip().replace('\r', '.').replace('\n', '.').split('.') for sent in sentences if sent != "" ]
    no_white_space = [sentence.strip() for sentences in no_white_space for sentence in sentences if sentence != '']

    splitted = [ [char for char in sent.split(" ") if char != ""] for sent in no_white_space]
    return np.mean([len(chunk) for chunk in splitted])

def flesch(text):
    return round(textstat.flesch_reading_ease(text), 1)

def dale_chall(text):
    return round(textstat.dale_chall_readability_score(text), 1)

# Return the chosen semantic score for the input text
# possible score_mod options are ["neg", "neu", "pos", "compound"]
def semantic_scores(text, score_mod):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)[score_mod]
    


def analyze(blogtext):
    out = {}
    # Format: out[<Feature title>] = (
    #   <value>, 
    #   <column_name>,
    #   <feature description>, 
    #   [optional] <additional markdown message>
    # )
    out['word count'] = (
        text_length(blogtext), 
        'text_length',
        'length of the blog text, measured in words'
    )
    out['average sentence length'] = (
        round(avg_sent_length(blogtext), 2), 
        'average_sentence_length',
        'average length of sentences in the blog text, measured in words'
    )
    out['Flesch readability score'] = (
        flesch(blogtext),
        'flesch',
        'Flesch readability score estimates the ease of reading the text. The score is based on average syllable lengths of words in the text, as well as the average sentence length.  ',
        r""" 
        | Score        | Education level (US) | Notes                                                                   |  
        | ------------ | -------------------- | ----------------------------------------------------------------------- |  
        | 100.00–90.00 | 5th grade            | Very easy to read. Easily understood by an average 11-year-old student. |  
        | 90.0–80.0    | 6th grade            | Easy to read. Conversational English for consumers.                     |  
        | 80.0–70.0    | 7th grade            | Fairly easy to read.                                                    |  
        | 70.0–60.0    | 8th & 9th grade      | Plain English. Easily understood by 13- to 15-year-old students.        |  
        | 60.0–50.0    | 10th to 12th grade   | Fairly difficult to read.                                               |  
        | 50.0–30.0    | College	          | Difficult to read.                                                      |  
        | 30.0–10.0    | College graduate	  | Very difficult to read. Best understood by university graduates.        |  
        | 10.0–0.0	   | Professional	      | Extremely difficult to read. Best understood by university graduates.   |  
        """
    )
    out['Dale-Chall readability score'] = (
        dale_chall(blogtext),
        'dale_chall',
        'Dale-Chall readability score estimates the ease of reading the text. The score is based on the number of difficult words in the text and average sentence length.  ',
        r"""
        | Score | Education level |
        | ----- | --------------- |
        | 4.9 and Below	| Grade 4 and Below |
        | 5.0 to 5.9	| Grades 5 - 6 |
        | 6.0 to 6.9	| Grades 7 - 8 |
        | 7.0 to 7.9	| Grades 9 - 10 |
        | 8.0 to 8.9	| Grades 11 - 12 |
        | 9.0 to 9.9	| Grades 13 - 15 (College) |
        | 10 and Above	| Grades 16 and Above (College Graduate) |
        """
    )
    out['Positive semantic score'] = (
        semantic_scores(blogtext, "pos"),
        'semantic pos score',
        '''
        Positive semantic score measures the percentage of words that are labeled to be positive.
        '''
    )
    out['Neutral semantic score'] = (
        semantic_scores(blogtext, "neu"),
        'semantic neu score',
        '''
        Neutral semantic score measures the percentage of words that are labeled to be neutral.
        '''
    )
    out['Negative semantic score'] = (
        semantic_scores(blogtext, "neg"),
        'semantic neg score',
        '''
        Negative semantic score measures the percentage of words that are labeled to be negative.
        '''
    )
    return out

relevant_features = {
    'avg_time' : ['text_length', 'average_sentence_length', 'dale_chall'],
    'bounce_rate' : ['text_length', 'average_sentence_length', 'dale_chall', 'semantic pos score', 'mmr_lift'],
    'exit%' : ['text_length', 'semantic pos score', 'mmr_lift', 'average_sentence_length'],
    'pageviews' : ['text_length']
}
high_good = {
    'pageviews' : True,
    'avg_time' : True,
    'bounce_rate' : False,
    'exit%': False
}

input, output = st.columns(2, gap='large')
input, output = st.columns([2,3], gap='large')

with input:
    def icon(good, relevant):
        if not relevant:
            return '🔵'
        elif good:
            return '🟢' 
        else:
         return '🟠'
    form = st.form(key="input_form")
    with form:
        target = st.selectbox(
            label='Optimizing for', 
            options=['avg_time', 'bounce_rate', 'exit%', 'pageviews']
        )
        if target == 'pageviews':
            st.error('Target region estimations are not reliable for pageviews')
        blogtext = st.text_area(label="Your blog text")
        submitted = st.form_submit_button("Analyze")
        if submitted:
            with output:
                st.markdown("🟢 - *all good*, 🟠 - *worth a second look*, 🔵 - *not relevant*")
                st.info("Feature importance based on regression analysis")
                expanders = {}
                analysis = analyze(blogtext)
                for key, value in analysis.items():
                    predictor = value[1]
                    preds = data[predictor]
                    targets = data[target]
                    if high_good[target]:
                        percent = 90
                        good_target = np.percentile(targets, percent)
                        successful_preds = data[data[target] > good_target][predictor]
                    else:
                        percent = 10
                        good_target = np.percentile(targets, percent)
                        successful_preds = data[data[target] < good_target][predictor]
                    int_low = np.percentile(successful_preds, 20)
                    int_hi = np.percentile(successful_preds, 80)
                    cond_low = int_low <= value[0]
                    cond_hi = value[0] <= int_hi
                    good = cond_low and cond_hi
                    relevant = value[1] in relevant_features[target]
                    expanders[key] = st.expander(f'**{key[0].upper() + key[1:]}**: *{value[0]}* {icon(good, relevant)}')
                    with expanders[key]:
                        st.markdown(f'*Feature description:* {value[2]}')
                        if len(value) == 4:
                            st.markdown(value[3]) 
                            st.write("")
                        if not cond_low:
                            st.markdown(f'*Assessment:* the value of {key} is low compared to well-performing blogs')
                        elif not cond_hi:
                            st.markdown(f'*Assessment:* the value of {key} is high compared to well-performing blogs')
                        else:
                            st.markdown(f'*Assessment:* the value of {key} is in line with those of well-performing blogs')
                        
                        # horrible hack for adding a legend
                        label_column = pd.DataFrame({'hist_label':data.index.size * [r'distribution across top 10% blogs']})
                        hist = alt.Chart(pd.concat([data, label_column], axis=1)).mark_bar(opacity=0.8).transform_filter(
                            datum[target] >= good_target
                        ).encode(
                            alt.X(predictor, bin=True, axis=alt.Axis(title=f'{predictor}')),
                            y=alt.Y('count()'),
                            color=alt.Color('hist_label', title=r' ')
                        )
                        cutoff = pd.DataFrame({
                            'start':[int_low], 
                            'stop':[int_hi], 
                            'value' : [value[0]],
                            'area_label':[r'20-80 percentile region of top 10% blogs'],
                            'line_label':['your blog']
                        })
                        area = alt.Chart(
                            cutoff.reset_index()
                        ).mark_rect(
                            opacity=0.4,
                            color='lightgreen',
                            clip=True
                        ).encode(
                            x='start',
                            x2='stop',
                            y=alt.value(0),  # pixels from top
                            y2=alt.value(200),  # pixels from top
                            opacity=alt.Opacity('area_label', title=' ', scale=alt.Scale(range=[0.4, 0.4]))
                        ) + alt.Chart(cutoff).mark_rule(color='lightgreen', opacity=0.6, size=1.5).encode(
                            x='start'
                        ) + alt.Chart(cutoff).mark_rule(color='lightgreen', opacity=0.6, size=1.5).encode(
                            x='stop'
                        )
                        line = alt.Chart(pd.DataFrame(cutoff)).mark_rule(color='red', size=1).encode(
                            x='value',
                            strokeWidth=alt.Size('line_label', title=' ', scale=alt.Scale(range=[2, 2]))
                        )
                        st.altair_chart(
                            (hist + area + line).configure_legend(labelLimit=0).properties(
                                height = 200,
                                width = 600
                            )
                        )