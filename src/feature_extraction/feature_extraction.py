import re
import textstat
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import chain
import ssl
from nltk.sentiment import SentimentIntensityAnalyzer
from pytrends.request import TrendReq 
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity 

def main():
    # Download NLTK corpuses
    print("Downloading NLTK corpuses")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')    
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download(["vader_lexicon"])
    print("Completed")

    # Fetch the data
    df = pd.read_csv("data/interim/blogs_with_analytics.csv", sep="\t")
    df.dropna(subset=["text"], inplace=True)

    # Basic text stats
    print("Extracting basic text statistics...")
    df = extract_basic_text_stats(df)
    print("Completed")

    # Sentiment scores
    print("Extracting sentiment scores...")
    df = extract_sentiment_scores(df)
    print("Completed")

    # Lift scores
    print("Calculating lift scores...")
    # df = extract_lift_scores(df)
    print("Completed")

    # Save the final result
    df.to_csv("data/final/futurice_blog_data.csv", sep='\t', index=False)
    print("Finished")


def extract_basic_text_stats(df):
    """
        Extracts basic statistics from blog text. 
        Extracted features include:
            * Text length
            * Average stopwords per sentence
            * Dale-Chall readability score
            * Flesch readability score
    """
    # Sentence lengths
    texts = df["text"].astype(str)
    sents_df = [sent_tokenize(sent) for sent in texts]
    sents_df = [ [re.sub(pattern="\d+[.]",repl="", string=sent.strip()) for sent in sent_df] for sent_df in sents_df ]
    sents_df = [ [re.sub(pattern="[^a-zA-Z0-9\s]",repl="", string=sent) for sent in sent_df] for sent_df in sents_df ]
    res_df = [ [ sent.strip().replace('\r', '.').replace('\n', '.').split('.') for sent in sent_df if sent != "" ] for sent_df in sents_df ]
    res_df = [ [sentence.strip() for sentences in bunch for sentence in sentences if sentence != ''] for bunch in res_df ]
    splitted_df = [ [ [char for char in sent.split(" ") if char != ""] for sent in res] for res in res_df ]
    avg_df = [ np.mean([len(chunk) for chunk in spliting]) for spliting in splitted_df ]
    sum_df = [ np.sum([len(chunk) for chunk in spliting]) for spliting in splitted_df ]
    df['text_length'] = np.array(sum_df)

    # Readability scores
    dale_chall = np.full(df.shape[0], -1, float)
    flesch = np.full(df.shape[0], -1, float)
    for i, text in enumerate(df.text):
        dale_chall[i] = textstat.dale_chall_readability_score(text)
        flesch[i] = textstat.flesch_reading_ease(text)
    df["dale_chall"] = dale_chall
    df["flesch"] = flesch

    # Average stopwords
    texts = df["text"].astype(str)
    sents_length_df = np.array([len(sent_tokenize(sent)) for sent in texts])
    stopwords_df = np.array([word_tokenize(text) for text in texts])
    stopwords_df = np.array([len([w for w in tokens if w in stopwords.words('english')]) for tokens in stopwords_df])
    df["average_stopword"] = np.divide(stopwords_df, sents_length_df)
    return df

def extract_sentiment_scores(df):
    """Extracts sentiment scores from blog texts"""
    def get_semantic_scores(text):
        return sia.polarity_scores(text)

    # Function that adds semantic scores to one specified index (index) in data frame (df) to columns (column_names = [neg, neu, pos, compound])
    def input_semantic_scores_to_df(df, index, column_names):
        scores = get_semantic_scores(df.iloc[index]["text"])
        semantic_labels = ["neg", "neu", "pos", "compound"]

        for i in range(len(scores)):
            df.at[index, column_names[i]] = scores[semantic_labels[i]]

    sia = SentimentIntensityAnalyzer()
    for index in range(len(df)):
        input_semantic_scores_to_df(df, index, ["semantic neg score", "semantic neu score", "semantic pos score", "semantic compound score"])
    return df

def extract_lift_scores(df):
    """
        Computes a lift score for each blog. 
        The score is informed by popularity of the blog's prominent keywords in Google searches. 
    """

    # Fetches the missing keywords from google trends
    def get_google_trends_to_csv(keyword, df, df_index, df_name = "data/interim/trend_single_score.csv"):
        comp_time = "2010-01-01 2022-11-19"
        pytrends = TrendReq(hl='en-US', tz=-120, timeout=(10,25), retries = 4, backoff_factor=10)
        pytrends.build_payload([keyword], timeframe=comp_time, geo = "")
        loc_df = pytrends.interest_over_time()

        if(len(loc_df) < 1):
            loc_df = pytrends.interest_over_time()
            if(len(loc_df) < 1):
                with open("data/interim/lift_score_unpopular_google_searches", "a") as file:
                    file.write(str(df_index) + ": " + keyword +"\n")
                    return df, False
        loc_df = loc_df.drop(columns="isPartial")
        df = pd.concat([df, loc_df], axis=1)
        df.to_csv(df_name)
        return df, True

    #Calculates the lift score = has the word been more or less trendy this month than on average within the last year
    def get_lift(keyword, df, year, month, df_index):
        if keyword not in df.columns:
            print("    adding " + keyword + " to data base.")
            df, succeeded = get_google_trends_to_csv(keyword, df, df_index)
            if not succeeded:
                print("    adding failed.")
                return df, 1
        end_index = int(np.where((df.index.year ==  year) & (df.index.month == month))[0])
        start_index = int(np.where((df.index.year ==  year-1) & (df.index.month == month))[0])
        month_score = df.iloc[end_index][keyword]
        mean = df.iloc[start_index:end_index][keyword].mean()
        return df, month_score/(max(mean, 1))

    # Maximal Marginal Relevance
    # Returns top_n best keywords
    def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

        # Extract similarity within words, and between words and the document
        word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
        word_similarity = cosine_similarity(word_embeddings)

        # Initialize candidates and already choose best keyword/keyphras
        keywords_idx = [np.argmax(word_doc_similarity)]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]
        for _ in range(top_n - 1):
            # Extract similarities within candidates and
            # between candidates and selected keywords/phrases
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

            # Calculate MMR
            mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update keywords & candidates
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)
        return [words[idx] for idx in keywords_idx]
    
    # Preprocesses text and call mmr
    # Returns top_n keywords
    def get_mmr_keywords(doc, top_n=5):
        n_gram_range = (1,1)
        count = CountVectorizer(ngram_range=n_gram_range, stop_words="english").fit([doc])
        candidates = count.get_feature_names_out()
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        doc_embedding = model.encode([doc])
        candidate_embeddings = model.encode(candidates)
        if len(candidates) <= top_n:
            return candidates
        return mmr(doc_embedding, candidate_embeddings, candidates, top_n=top_n, diversity=0.2)
    
    # Main loop
    try:
        keyword_df = pd.read_csv("data/interim/trend_single_score.csv", parse_dates=["date"], infer_datetime_format=True, index_col=["date"])
    except:
        print("Could not open the keyword popularity database. Initializing a new one.")
        pytrends = TrendReq(hl='en-US', tz=-120, timeout=(10,25), retries = 4, backoff_factor=10)
        keyword = "Google"
        comp_time = "2010-01-01 2022-11-19"
        pytrends.build_payload([keyword], timeframe=comp_time, geo = "")
        loc_df = pytrends.interest_over_time()
        loc_df = loc_df.drop(columns="isPartial")
        keyword_df = loc_df

    for index in range(len(df)):
        keywords = get_mmr_keywords(df.iloc[index]["text"])
        print(str(index) + ": " + str(keywords))
        lift_sum = 0
        for word in keywords: 
            keyword_df, lift = get_lift(word, keyword_df, int(df.iloc[index]["time"].year), int(df.iloc[index]["time"].month), index)
            lift_sum += lift
            df.at[index, "mmr_lift"] = lift_sum
    return df

if __name__ == '__main__':
    main()