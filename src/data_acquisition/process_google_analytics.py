"""
This script processes the provided xlsx files containing Google Analytics data and saves the results into a csv file.

Extracted sections include the following attributes for each blog:

    * URL
    * pageviews
    * unique pageviews
    * average time on page
    * bounce rate
    * exit percent

NOTE: In the publicly available version of the project, we are not able to use the confidential Google Analytics data provided by Futurice. Thus, the data will be replaced by randomly generated samples.
"""

import pandas as pd
import numpy as np

def main():
    urls = pd.read_csv('data/raw/scraped_blog_data.csv')['url']
    generate_random_data(urls, 'data/interim/google_analytics.csv')


def generate_random_data(urls, output_filepath):
    """Generates random data to replace the private Google Analytics data"""

    r = np.random.default_rng(seed=42)
    n = len(urls)
    df = pd.DataFrame()
    df['url'] = urls
    df['url'] = df['url'].str.split('futurice.com/', expand=True)[1].str.rstrip('/')
    df['pageviews'] = np.ceil(r.lognormal(mean=5, sigma=1.5, size=n)).astype('int')
    df['unique_pageviews'] = np.ceil(df.pageviews - r.uniform(low=0, high=df.pageviews * 0.1)).astype('int')
    df['avg_time'] = r.lognormal(mean=5, sigma=0.8, size=n)
    df['bounce_rate'] = r.uniform(low=0, high=1, size=n)
    df['exit%'] = r.uniform(low=0, high=1, size=n)
    df.to_csv(output_filepath, index=False)

def process_xlsx_data(output_filepath):
    """
    Reads and aggregates data from the xlsx database, writes the results to a csv file.
    This method is not used in the public version of the project.
    """

    def excel_to_df(filepath):
        """Reads an xlsx file into a formatted pandas dataframe"""
        # Import data from excel sheet
        df = pd.read_excel(filepath, sheet_name='Dataset1', engine="openpyxl")

        # Clean up column names
        df.rename(columns={
            'Page path level 2':'url',
            'Pageviews': 'pageviews',
            'Unique Pageviews': 'unique_pageviews',
            'Avg. Time on Page': 'avg_time',
            'Bounce Rate': 'bounce_rate',
            '% Exit': 'exit%'
            }, inplace=True)

        # Clean up blog URLs to match formatting of previous data
        df['url'] = 'blog' + df['url'].str.split('?', expand=True)[0].str.rstrip('/')
        return df

    # Import all three excel sheets
    df1 = excel_to_df('../data/raw/Google_analytics_1.xlsx')
    print(f'size of df1: {df1.size}')
    df2 = excel_to_df('../data/raw/Google_analytics_2.xlsx')
    print(f'size of df2: {df2.size}')
    df3 = excel_to_df('../data/raw/Google_analytics_3.xlsx')
    print(f'size of df3: {df3.size}')

    # Stitch the dataframes together
    df = pd.concat([df1, df2, df3], axis=0).reset_index()

    # Aggregate data along blog URLs
    wm = lambda x: np.average(x, weights=df.loc[x.index, "pageviews"])
    df = df.groupby('url').agg(
        {
            'pageviews': 'sum',
            'unique_pageviews': 'sum',
            'avg_time': wm,
            'bounce_rate' : wm,
            'exit%': wm
        }
    )

    #Write into a CSV file
    df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    main()