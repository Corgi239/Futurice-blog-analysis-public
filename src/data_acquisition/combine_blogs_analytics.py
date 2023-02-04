"""
This script combines the scraped blog data and the processed Google Analytics data into one unified dataframe.
The output is written to data/final/blogs_with_analytics.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime

def main():
    combine_data('data/interim/blogs_with_analytics.csv')

def combine_data(output_path):
    # Read blog data and analytics data
    blogs = pd.read_csv('data/raw/scraped_blog_data.csv', 
        sep=',', 
        engine='python', 
        parse_dates=['time'], 
        date_parser=lambda col: pd.to_datetime(col, utc=True)
    )
    analytics = pd.read_csv('data/interim/google_analytics.csv')

    # Clean up category titles
    blogs['category'] = blogs['category'].str.replace('&amp;', 'and')

    # Convert time strings to datetime format
    blogs['time'] = blogs['time'].dt.date

    blogs['url'] = 'blog' + blogs['url'].str.split('blog', expand=True)[1].str.rstrip('/')

    # Combine the two tables
    combined = pd.merge(blogs, analytics, how='inner', on='url')

    # Drop entires that do not have urls, pageviews, or text
    combined = combined.dropna(subset=['url', 'pageviews', 'text'])

    # Write the resulting merged table to file
    combined.to_csv(output_path, sep='\t', index=False)

if __name__ == '__main__':
    main()