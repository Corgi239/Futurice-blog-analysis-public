"""
Crawls the blog section of futurice.com website, fetches all available blog urls, downloads the contents of the blogs, parses the contents into relevant sections, and saves the results to data/raw/blog_text.csv.

Scraped sections include:

    * blog URL
    * title
    * publishing date
    * blog category
    * description (teaser)
    * introduction paragraph
    * main text
    * author
    * author job title

NOTE: Requires you to install Firefox's Geckodriver onto your system, see online documentation.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def main():
    
    # Setting browser to Firefox and initializing as headless (without the browser opening up in a window)
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    options = FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)

    short_urls = scrape_blog_urls()
    urls_list = ["https://futurice.com" + short_url for short_url in short_urls]

    # Slice the list into 100 url chunks
    chunks = [urls_list[x:x+100] for x in range(0, len(urls_list), 100)]

    # List of urls that aren't compatible with scraper
    list_of_incompatible_blogs = []

    # Initialise DataFrame with column names
    df = pd.DataFrame(columns=["url","title","date","category","description","body","introduction","author","author_job_title"])

    # Start iterating through the blogs
    for chunk in chunks:

        # Initialise a list which will contain our blog objects (dictionaries)
        list_of_blog_objects = []

        #iterate through the urls in each chunk
        for url in chunk:

            # Get the blog
            driver.get(url)

            # Title of the blog
            title = driver.title

            # Date the blog was published. In Datetime format
            try:
                date = driver.find_element(By.CLASS_NAME, "sc-adf7a739-0.dbnxnV").get_attribute("datetime")
            except:
                date = "This blog is incompatible"
                list_of_incompatible_blogs.append(url)
                continue

            # Category the blog is under
            category = driver.find_element(By.CLASS_NAME, "sc-adf7a739-1.lkSQky").text

            # Description of the blog for SEO, different from the blurb at the top of the blog
            description = driver.find_element(By.XPATH, "//meta[@name='description']").get_attribute("content")

            # The main body of the blog
            try:
                body = driver.find_element(By.CLASS_NAME, "sc-62bcb39b-0.kwENye.bodySection_body__wrapper__dyPBE").text
            except:
                body = "This blog is incompatible"
                list_of_incompatible_blogs.append(url)
                continue

            # The introduction blurb
            try:
                introduction = driver.find_element(By.CLASS_NAME, "introduction_intro__text__wT0nc").text
            except:
                # In case there is no introduction, we use the first 30 words of the body
                try:
                    introduction = body.split()[:30]
                    result_str = " ".join(body)
                # In case there is no body
                except:
                    introduction = "This blog is incompatible"

            
            # The authors of the blog and their job titles

            # Getting a list of Selenium elements that match the class name,
            # Mapping a function that converts those elements to strings
            author = driver.find_elements(By.CLASS_NAME, "sc-b0268d1f-5.hePoge")
            author = list(map(lambda x: x.text, author))
            
            # Same as above
            author_job_title = driver.find_elements(By.CLASS_NAME, "sc-b0268d1f-6.kOgHwO")
            author_job_title = list(map(lambda x: x.text, author_job_title))

            # Checking for number of authors. 
            # If 1, assigns to variable
            # If >1, concatenates with ':' in the middle and assigns to variable
            # If 0, defaults to "No ..." 
            if (len(author) == 1):
                author = author[0]
                author_job_title = author_job_title[0]
            elif (len(author) > 1):
                author = ':'.join(author)
                author_job_title = ':'.join(author_job_title)
            else:
                author = "No Author"
                author_job_title = "No Author Position"
            
            # Empty dictionary representing each blog 
            blog_object = {}

            # Assigning key-value pairs to the dictionary
            blog_object["url"] = url
            blog_object["title"] = title
            blog_object["date"] = date
            blog_object["category"] = category
            blog_object["description"] = description
            blog_object["body"] = body
            blog_object["introduction"] = introduction
            blog_object["author"] = author
            blog_object["author_job_title"] = author_job_title

            # Appending blog object to list
            list_of_blog_objects.append(blog_object)
        
        # Convert the list of blog objects into a DataFrame    
        temporary_df_holder = pd.DataFrame(list_of_blog_objects)    

        # Concatenate the latest DataFrame with existing DataFrame, ignore the index of the new one 
        # { Every temporary DataFrame will have index [1:50], if we don't specify ignore parameter, 
        # pandas will try to preserve old and new indices and throw an error }

        df = pd.concat([df, temporary_df_holder], ignore_index=True)    
        
        # Free up memory held by both current chunk of blogs (in list) and temporary DataFrame 
        del list_of_blog_objects
        del temporary_df_holder
        
        # Wait 15 seconds between chunks to not trip the bot detection tools
        time.sleep(15)

    # Save DataFrame as CSV
    df.to_csv("data/raw/scraped_blog_data.csv")

    # Save urls of incompatible blogs as CSV
    pd.DataFrame(list_of_incompatible_blogs, columns = ["urls"]).to_csv("data/raw/incompatible_blog_urls.csv")

def scrape_blog_urls(verbose = False):
    """Helper method for scraping all available blog urls, starting from base page "futurice.com/blog"""

    def scrape_one_base_page_for_urls(page_num):
        """Scrapes all blog links form the "page_num":th page in the "https://futurice.com/blog?page=" and returns the urls found as a list of strings (or a empty list)"""
        urls = []
        base_url = "https://futurice.com/blog?page="
        
        r = requests.get(base_url + str(page_num))

        # Check if was able to access the internet page
        if r.status_code//100 != 2:
            print("ERROR WHILE READING WEBPAGE")
            return List()
        
        # Parse the text
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Add all the links returned by the bs4 to the list
        for item in soup.body.main.find_all("a"):
            if item.get("href"):
                urls.append(item.get("href"))
            
        return urls

    scrape_more = True # A flag, that is turned false, when detected that the index:th page is last page
    index = 1          # iteration index
    url_count= 0
    urls = []

    while scrape_more:
        urls.extend(scrape_one_base_page_for_urls(index))
        if len(urls) == url_count:
            scrape_more = False
        if verbose: {print(str(index) + ": " + str(len(urls)))}
        index += 1
        url_count = len(urls)

    if verbose: {print("Finished")}
    return urls

if __name__ == '__main__':
    main()