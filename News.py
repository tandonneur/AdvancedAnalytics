#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:58:00 2019

@author: EJones
"""
import pandas as pd
import sys
import re
import requests  # install using conda instal requests
import newspaper # install using pip install newspaper3k
from newspaper import Article

# newsapi requires tiny\segmenter:  pip install tinysegmenter==0.3
# Install newsapi using:  pip install newsapi-python
# from newsapi import NewsApiClient # Needed for using API Feed
from time import time
from datetime import date

class News:
    def newspaper_stories(words, urls=None, display=True):
        if urls == None:
            news_urls = {'huffington': 'http://huffingtonpost.com', 
                 'reuters': 'http://www.reuters.com', 
                 'cbs-news': 'http://www.cbsnews.com',
                 'usa-today': 'http://usatoday.com',
                 'cnn': 'http://cnn.com',
                 'npr': 'http://www.npr.org',
                 'abc-news': 'http://abcnews.com',
                 'us-news': 'http://www.usnews.com',
                 'msn':  'http://msn.com',
                 'pbs': 'http://www.pbs.org',
                 'nbc-news':  'http://www.nbcnews.com',
                 'fox': 'http://www.foxnews.com'}
        else:
            news_urls = urls
            
        df_articles = pd.DataFrame(columns=['agency', 'url', 'story'])
        n_articles  = {}
        today = str(date.today())
        for agency, url in news_urls.items():
            paper = newspaper.build(url, memoize_articles=False, \
                                   fetch_images=False, request_timeout=20)
            if display:
                print("\n", paper.size(), "Articles available from " +\
                      agency.upper()+" on "+today+" :")
            n_articles_selected = 0
            article_collection = []
            for word in words:
                word = word.lower()
                for article in paper.articles:
                    # Exclude articles that are in a language other then en
                    # or contains mostly video or pictures
                    if article.url.find('.video/')>=0 or \
                       article.url.find('/video') >=0 or \
                       article.url.find('/picture') >=0 or \
                       article.url.find('.pictures/')>=0 or \
                       article.url.find('/photo') >=0 or \
                       article.url.find('.photos/')>=0 or \
                       article.url.find('.mx/' )>=0 or \
                       article.url.find('/mx.' )>=0 or \
                       article.url.find('.fr/' )>=0 or \
                       article.url.find('/fr.' )>=0 or \
                       article.url.find('.de/' )>=0 or \
                       article.url.find('/de.' )>=0 or \
                       article.url.find('.it/' )>=0 or \
                       article.url.find('/it.' )>=0 or \
                       article.url.find('.gr/' )>=0 or \
                       article.url.find('/gr.' )>=0 or \
                       article.url.find('.se/' )>=0 or \
                       article.url.find('/se.' )>=0 or \
                       article.url.find('.es/' )>=0 or \
                       article.url.find('/es.' )>=0 :
                         continue
                    if agency=='usa-today':
                        if article.url.find('tunein.com') <0:
                               article_collection.append(article.url)
                        continue
                    if agency=='huffington':
                        if article.url.find('.com') >=0:
                               article_collection.append(article.url)
                        continue
                    if agency=='cbs-news':
                        if  article.url.find('.com') >=0 :
                                # secure-fly are duplicates of http
                                if article.url.find('secure-fly')>=0:
                                     continue
                                article_collection.append(article.url)
                        continue
                    article_collection.append(article.url)
            if display:
                print(len(article_collection), "Articles selected for download")
            j = 0
            for article_url in article_collection:
                j += 1
                article = Article(article_url)
                article.download()
                m = article_url.find(".com")
                m_org = article_url.find(".org")
                if m_org>m:
                    m = m_org
                m += 5
                k = len(article_url) - m
                if k > 70:
                    k=70
                if display:
                    print(j, " ", article_url[m:k+m])
                n = 0
                # Allow for a maximum of 5 download failures
                stop_sec=1 # Initial max wait time in seconds
                while n<3:
                    try:
                        article.parse()
                        n = 99
                    except:
                        n += 1
                        # Initiate download again before new parse attempt
                        article.download()
                        # Timeout for 5 seconds waiting for download
                        t0 = time()
                        tlapse = 0
                        print("Waiting", stop_sec,"sec")
                        while tlapse<stop_sec:
                            tlapse = time()-t0
                        # Double wait time if needed for next exception
                        stop_sec = stop_sec+1
                if n != 99:
                    # download failed
                    continue
                story          = article.text.lower()
                url_lower_case = article.url.lower()
                for word in words:
                    flag = 0
                    if url_lower_case.find(word)>0:
                        flag = 1
                        break
                    if story.find(word)>0:
                        flag = 1
                        break
                if flag == 0:
                    continue
                df_story    = pd.DataFrame([[agency, article_url, story]], \
                                   columns=['agency', 'url', 'story'])
                df_articles = df_articles.append(df_story)
                n_articles_selected += 1
            n_articles[agency] = [n_articles_selected, len(article_collection)]
            if display:
                ratio = str(n_articles_selected)+"/"+\
                                    str(len(article_collection))
                ratio = ratio + " articles selected from "+url.upper()
                print(ratio)
        if display:
            print("")
            for agency in news_urls:
                ratio = str(n_articles[agency][0])+"/"+str(n_articles[agency][1])
                ratio = ratio + " articles selected from "+agency.upper()
                print(ratio)
            print("\nTotal Articles Selected on "+today+":", df_articles.shape[0])
        return df_articles
    
    def clean_html(html):
        # First we remove inline JavaScript/CSS:
        pg = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
        # Then we remove html comments. This has to be done before removing regular
        # tags since comments can contain '>' characters.
        pg = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", pg)
        # Next we can remove the remaining tags:
        pg = re.sub(r"(?s)<.*?>", " ", pg)
        # Finally, we deal with whitespace
        pg = re.sub(r"&nbsp;", " ", pg)
        pg = re.sub(r"&rsquo;", "'", pg)
        pg = re.sub(r"&#x27;", "'", pg)
        pg = re.sub(r"&ldquo;", '"', pg)
        pg = re.sub(r"&rdquo;", '"', pg)
        pg = re.sub(r"&quot;", '"', pg)
        pg = re.sub(r"&amp;", '&', pg)
        pg = re.sub(r"\n", " ", pg)
        pg = re.sub(r"\t", " ", pg)
        pg = re.sub(r"/>", " ", pg)
        pg = re.sub(r'/">', " ", pg)
        k = 1
        m = len(pg)
        while k>0:
            pg = re.sub(r"  ", " ", pg)
            k = m - len(pg)
            m = len(pg)
        return pg.strip()

    def newsapi_get_urls(search_words, key=None, urls=None):
        if urls==None:
            agency_urls = {
                'huffington': 'http://huffingtonpost.com',
                'reuters': 'http://www.reuters.com',
                'cbs-news': 'http://www.cbsnews.com',
                'usa-today': 'http://usatoday.com',
                'cnn': 'http://cnn.com',
                'npr': 'http://www.npr.org',
                'wsj': 'http://wsj.com',
                'fox': 'http://www.foxnews.com',
                'abc': 'http://abc.com',
                'abc-news': 'http://abcnews.com',
                'abcgonews': 'http://abcnews.go.com',
                'nyt': 'http://nytimes.com',
                'washington-post': 'http://washingtonpost.com',
                'us-news': 'http://www.usnews.com',
                'msn':  'http://msn.com',
                'pbs': 'http://www.pbs.org',
                'nbc-news':  'http://www.nbcnews.com',
                'enquirer': 'http://www.nationalenquirer.com',
                'la-times': 'http://www.latimes.com'
                }
        else:
            agency_urls = urls
        if len(search_words)==0 or agency_urls==None:
            return None
        print("Searching agencies for pages containing:", search_words)
        # This is my API key, each user must request their own
        # API key from https://newsapi.org/account
        try:
            api = NewsApiClient(api_key=key)
        except:
            raise RuntimeError("***Call to request_pages invalid.\n"+\
                               " api key was not accepted.")
            sys.exit()
            
        api_urls  = []
        # Iterate over agencies and search words to pull more url's
        # Limited to 1,000 requests/day - Likely to be exceeded 
        for agency in agency_urls:
            domain = agency_urls[agency].replace("http://", "")
            print(agency, domain)
            for word in search_words:
                # Get articles with q= in them, Limits to 20 URLs
                try:
                    articles = api.get_everything(q=word, language='en',\
                                        sources=agency, domains=domain)
                except:
                    print("--->Unable to pull news from:", agency, "for", word)
                    continue
                # Pull the URL from these articles (limited to 20)
                d = articles['articles']
                for i in range(len(d)):
                    url = d[i]['url']
                    api_urls.append([agency, word, url])
        df_urls  = pd.DataFrame(api_urls, columns=['agency', 'word', 'url'])
        n_total  = len(df_urls)
        # Remove duplicates
        df_urls  = df_urls.drop_duplicates('url')
        n_unique = len(df_urls)
        print("\nFound a total of", n_total, " URLs, of which", n_unique,\
              " were unique.")
        return df_urls
    
    def request_pages(df_urls):
        try:
            if df_urls.shape[0]==0:
                return None
        except:
            raise RuntimeError("***Call to request_pages invalid.")
            sys.exit()
            
        web_pages = []
        for i in range(len(df_urls)):
            u   = df_urls.iloc[i]
            url = u[2]
            k = len(url)
            short_url = url[0:k]
            short_url = short_url.replace("https://", "")
            short_url = short_url.replace("http://", "")
            k = len(short_url)
            if k>70:
                k=70
            short_url = short_url[0:k]
            n = 0
            # Allow for a maximum of 3 download attempts
            stop_sec=3 # Max wait time per attempt
            while n<2:
                try:
                    r = requests.get(url, timeout=(stop_sec))
                    if r.status_code == 408:
                        print("-->HTML ERROR 408", short_url)
                        raise ValueError()
                    if r.status_code == 200:
                        print(short_url)
                    else:
                        print("-->Web page: "+short_url+" status code:", \
                                  r.status_code)
                    n=99
                    continue # Skip this page
                except:
                    n += 1
                    # Timeout waiting for download
                    t0 = time()
                    tlapse = 0
                    print("Waiting", stop_sec, "sec")
                    while tlapse<stop_sec:
                        tlapse = time()-t0
            if n != 99:
                # download failed skip this page
                continue
            # Page obtained successfully
            html_page = r.text
            page_text = News.clean_html(html_page)
            web_pages.append([url, page_text])
        df_www  = pd.DataFrame(web_pages, columns=['url', 'text'])
        n_total  = len(df_www)
        print("Attempted to download", len(df_urls), "web pages.", \
              " Obtained", n_total, ".")
        return df_www
   