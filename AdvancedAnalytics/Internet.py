"""

@author: Edward R Jones
@version 1.14
@copyright 2020 - Edward R Jones, all rights reserved.
"""

import sys
import warnings
import pandas as pd

import re
import requests  # install using conda install requests

from time import time
from datetime import date

try:
    import newspaper # install using conda install newspaper3k
    from newspaper import Article
except:
    warnings.warn("AdvancedAnalytics.Scrape.newspaper_stories "+\
                  "missing NEWSPAPER3K package")
try:
# newsapi requires tiny\segmenter:  pip install tinysegmenter==0.3
# Install newsapi using:  pip install newsapi-python
    from newsapi import NewsApiClient # Needed for using API Feed
except:
    warnings.warn("AdvancedAnalytics.Scrape.newsapi_get_urls "+\
                  "missing NEWSAPI package")
   
class scrape(object):
        
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

    def newsapi_get_urls(search_words, key, urls=None):
        if urls==None:
            agency_urls = {
                'abc-news': 'https://abcnews.go.com',
                'al-jazeera-english': 'http://www.aljazeera.com',
                'ars-technica': 'http://arstechnica.com',
                'associated-press': 'https://apnews.com/',
                'axios': 'https://www.axios.com',
                'bleacher-report': 'http://www.bleacherreport.com',
                'bloomberg': 'http://www.bloomberg.com',
                'breitbart-news': 'http://www.breitbart.com',
                'business-insider': 'http://www.businessinsider.com',
                'buzzfeed': 'https://www.buzzfeed.com',
                'cbs-news': 'http://www.cbsnews.com',
                'cnbc': 'http://www.cnbc.com',
                'cnn': 'http://us.cnn.com',
                'crypto-coins-news': 'https://www.ccn.com',
                'engadget': 'https://www.engadget.com',
                'entertainment-weekly': 'http://www.ew.com',
                'espn': 'http://espn.go.com',
                'espn-cric-info': 'http://www.espncricinfo.com/',
                'fortune': 'http://fortune.com',
                'fox-news': 'http://www.foxnews.com',
                'fox-sports': 'http://www.foxsports.com',
                'google-news': 'https://news.google.com',
                'hacker-news': 'https://news.ycombinator.com',
                'ign': 'http://www.ign.com',
                'mashable': 'http://mashable.com',
                'medical-news-today': 'http://www.medicalnewstoday.com',
                'msnbc': 'http://www.msnbc.com',
                'mtv-news': 'http://www.mtv.com/news',
                'national-geographic': 'http://news.nationalgeographic.com',
                'national-review': 'https://www.nationalreview.com/',
                'nbc-news': 'http://www.nbcnews.com',
                'new-scientist': 'https://www.newscientist.com/section/news',
                'newsweek': 'http://www.newsweek.com',
                'new-york-magazine': 'http://nymag.com',
                'next-big-future': 'https://www.nextbigfuture.com',
                'nfl-news': 'http://www.nfl.com/news',
                'nhl-news': 'https://www.nhl.com/news',
                'politico': 'https://www.politico.com',
                'polygon': 'http://www.polygon.com',
                'recode': 'http://www.recode.net',
                'reddit-r-all': 'https://www.reddit.com/r/all',
                'reuters': 'http://www.reuters.com',
                'techcrunch': 'https://techcrunch.com',
                'techradar': 'http://www.techradar.com',
                'the-american-conservative': 'http://www.theamericanconservative.com/',
                'the-hill': 'http://thehill.com',
                'the-huffington-post': 'http://www.huffingtonpost.com',
                'the-new-york-times': 'http://www.nytimes.com',
                'the-next-web': 'http://thenextweb.com',
                'the-verge': 'http://www.theverge.com',
                'the-wall-street-journal': 'http://www.wsj.com',
                'the-washington-post': 'https://www.washingtonpost.com',
                'the-washington-times': 'https://www.washingtontimes.com/',
                'time': 'http://time.com',
                'usa-today': 'http://www.usatoday.com/news',
                'vice-news': 'https://news.vice.com',
                'wired': 'https://www.wired.com'
                    }
        else:
            agency_urls = urls
        if len(search_words)==0 or agency_urls==None:
            return None
        print("Searching agencies for pages containing:", search_words)
       
        # Get your NEWSAPI key from https://newsapi.org/account
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
            page_text = scrape.clean_html(html_page)
            web_pages.append([url, page_text])
        df_www  = pd.DataFrame(web_pages, columns=['url', 'text'])
        n_total  = len(df_www)
        print("Attempted to download", len(df_urls), "web pages.", \
              " Obtained", n_total, ".")
        return df_www
    
class Metrics:
    # Function for calculating loss and confusion matrix
    def binary_loss(y, y_predict, fn_cost, fp_cost, display=True):
        loss     = [0, 0]       #False Neg Cost, False Pos Cost
        conf_mat = [[0, 0], [0, 0]] #tn, fp, fn, tp
        for j in range(len(y)):
            if y[j]==0:
                if y_predict[j]==0:
                    conf_mat[0][0] += 1 #True Negative
                else:
                    conf_mat[0][1] += 1 #False Positive
                    loss[1] += fp_cost[j]
            else:
                if y_predict[j]==1:
                    conf_mat[1][1] += 1 #True Positive
                else:
                    conf_mat[1][0] += 1 #False Negative
                    loss[0] += fn_cost[j]
        if display:
            fn_loss = loss[0]
            fp_loss = loss[1]
            total_loss = fn_loss + fp_loss
            misc    = conf_mat[0][1] + conf_mat[1][0]
            misc    = misc/len(y)
            print("{:.<23s}{:10.4f}".format("Misclassification Rate", misc))
            print("{:.<23s}{:10.0f}".format("False Negative Loss", fn_loss))
            print("{:.<23s}{:10.0f}".format("False Positive Loss", fp_loss))
            print("{:.<23s}{:10.0f}".format("Total Loss", total_loss))
        return loss, conf_mat
