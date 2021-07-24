"""

@author: Edward R Jones
@version 1.34
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
        
    def newspaper_stories(words, search_type='or', search_level=0, urls=None, 
                      display=True, memorize=False, language='en'):
        config = newspaper.Config()
        config.memoize_articles = memorize
        config.language         = language
        config.fetch_images     = False
        config.request_timeout  = 20
        config.MIN_WORD_COUNT   = 300
        config.MIN_SENT_COUNT   = 10
        if urls == None or urls =='top_news':
            news_urls = {
            'huffington': 'http://huffingtonpost.com', 
            'reuters':    'http://www.reuters.com', 
            'cbs-news':   'http://www.cbsnews.com',
            'usa-today':  'http://usatoday.com',
            'cnn':        'http://cnn.com',
            'npr':        'http://www.npr.org',
            'abc-news':   'http://abcnews.com',
            'us-news':    'http://www.usnews.com',
            'msn':        'http://msn.com',
            'pbs':        'http://www.pbs.org',
            'nbc-news':   'http://www.nbcnews.com',
            'msnbc':      'http://www.msnbc.com',
            'fox':        'http://www.foxnews.com'}
        elif urls=='all_us_news':
            news_urls = {
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
            'american-conservative': 'http://www.theamericanconservative.com/',
            'hill': 'http://thehill.com',
            'huffington-post': 'http://www.huffingtonpost.com',
            'next-web': 'http://thenextweb.com',
            'verge': 'http://www.theverge.com',
            'wall-street-journal': 'http://www.wsj.com',
            'washington-post': 'https://www.washingtonpost.com',
            'washington-times': 'https://www.washingtontimes.com/',
            'time': 'http://time.com',
            'usa-today': 'http://www.usatoday.com/news',
            'vice-news': 'https://news.vice.com',
            'wired': 'https://www.wired.com'}
        elif urls == "texas_universities":
            news_urls = {
            'A&M':               'http://www.tamu.edu',
            'A&M-Commerce':      'http://www.tamuc.edu',
            'A&M-Corpus':        'http://www.tamucc.edu',
            'A&M-Kingsville':    'http://www.tamuk.edu',
            'A&M-Galveston':     'http://www.tamug.edu',
            'A&M-PrairieView':   'http://www.pvamu.edu',
            'A&M-International': 'http://www.tamiu.edu',
            'A&M-WestTexas':     'http://www.wtamu.edu',
            'Baylor':            'http://www.baylor.edu',
            'Rice':              'http://www.rice.edu',
            'SFAustin':          'http://www.sfasu.edu',
            'SMU':               'http://www.smu.edu',
            'SulRoss':           'http://www.sulross.edu',
            'TexasState':        'http://www.txstate.edu',
            'Texas_Tech':        'http://www.ttu.edu',
            'UDallas':           'http://www.udallas.edu',
            'UHouston':          'http://www.uh.edu',
            'UTexas':            'http://www.utexas.edu', 
            'UT_Dallas':         'http://www.utdallas.edu',
            'UT_ElPaso':         'http://www.utep.edu',
            'UT_Houston':        'http://www.uth.edu',
            'UT_NorthTexas':     'http://www.unt.edu',
            'UT_SanAntonio':     'http://www.utsa.edu'}
        elif urls == 'popular':
            news_urls = {}
            agency_urls = newspaper.popular_urls()
            for i in range(len(agency_urls)):
                val = agency_urls[i]
                url = agency_urls[i].replace("http://", "")
                url = url.replace("www.", "")
                url = url.replace("blog.", "")
                url = url.replace("blogs.", "")
                url = url.replace(".com", "")
                url = url.replace(".net", "")
                url = url.replace(".au", "")
                url = url.replace(".org", "")
                url = url.replace(".co.uk", "")
                url = url.replace("the", "")
                url = url.replace(".", "-")
                url = url.replace('usa', 'usa-')
                if url=='berkeley-edu':
                    continue
                if url=='beta-na-leagueoflegends':
                    continue
                if url=='bottomline-as-ucsb-edu':
                    continue
                news_urls[url] = val
        else:
            news_urls = urls
        
        print("\nSearch Level {:<d}:".format(search_level), end="")
        if search_level==0:
            print(" Screening URLs for search words")
            print("   URLs must contain one or more of:", end="")
        else:
            print(" No URL Screening")
            print("   Deep Search for Articles containing: ", 
                  end="")
        i=0
        for word in words:
            i += 1
            if i < len(words):
                if search_type == 'or':
                    print(word+" or ", end="")
                else:
                    print(word+" & ", end="")
            else:
                print(word)
                    
        df_articles = pd.DataFrame(columns=['agency', 'url', 'length', 
                                            'keywords', 'title', 'summary', 
                                            'text'])
        n_articles  = {}
        today       = str(date.today())
        for agency, url in news_urls.items():
            paper   = newspaper.build(url, config=config)
            if display:
                print("\n{:>6d} Articles available from {:<s} on {:<10s}:".
                      format(paper.size(), agency.upper(), today))
            article_collection = []
            for article in paper.articles:
                url_lower = article.url.lower()
                # Exclude articles that are in a language other then en
                # or contains mostly video or pictures
                # search_level 0 only downloads articles with at least
                # one of the key words in its URL
                # search_level 1 download all articles that appear to be
                # appear to be in English and are not mainly photos or
                # videos.
                # With either search level, if an article is downloaded
                # it is scanned to see if it contains the search words
                # It is also compared to other articles to verify that
                # it is not a duplicate of another article.
                
                 # Special Filters for some Agencies
                if agency=='cbs-news':
                    if  url_lower.find('.com') >=0 :
                        # secure-fly are duplicates of http
                        if article.url.find('secure-fly')>=0:
                            continue
                if agency=='usa-today':
                    if url_lower.find('tunein.com') >= 0:
                        continue
                if agency=='huffington':
                    # Ignore huffington if it's not .com
                    if url_lower.find('.com') < 0:
                        continue
                    
                # Filter Articles that are primarily video, film or not en
                if url_lower.find('.video/')   >=0 or \
                   url_lower.find('/video')    >=0 or \
                   url_lower.find('/picture')  >=0 or \
                   url_lower.find('.pictures/')>=0 or \
                   url_lower.find('/photo')    >=0 or \
                   url_lower.find('.photos/')  >=0 or \
                   url_lower.find('espanol')   >=0 or \
                   url_lower.find('.mx/' )     >=0 or \
                   url_lower.find('/mx.' )     >=0 or \
                   url_lower.find('.fr/' )     >=0 or \
                   url_lower.find('/fr.' )     >=0 or \
                   url_lower.find('.de/' )     >=0 or \
                   url_lower.find('/de.' )     >=0 or \
                   url_lower.find('.it/' )     >=0 or \
                   url_lower.find('/it.' )     >=0 or \
                   url_lower.find('.gr/' )     >=0 or \
                   url_lower.find('/gr.' )     >=0 or \
                   url_lower.find('.se/' )     >=0 or \
                   url_lower.find('/se.' )     >=0 or \
                   url_lower.find('.es/' )     >=0 or \
                   url_lower.find('/es.' )     >=0 or \
                   url_lower.find('?button')   >=0 or \
                   url_lower.find('calendar.') >=0 or \
                   url_lower.find('calendar/') >=0 or \
                   url_lower.find('/event/')   >=0 or \
                   url_lower.find('engr.utexas') >=0 or \
                   url_lower.find('sites.smu.')  >=0:
                       continue

                # Filter if search_level == 0, URL quick search
                if search_level == 0:
                    # Verify url contains at least one of the key words
                    found_it = False
                    for word in words:
                        j = url_lower.find(word)
                        if j>= 0:
                            found_it = True
                            break
                    if found_it:
                        # Article contains words and passes filters
                        # Save this article for full review
                        article_collection.append(article.url)
                else:
                    #  No URL screening, Save for full review
                    article_collection.append(article.url)
            n_to_review = len(article_collection)
            if display:
                print("{:>6d} Selected for download".format(n_to_review))
            
            for article_url in article_collection:
                article = Article(article_url, config=config)
                try:
                    article.download()
                except: 
                    if display:
                        print("Cannot download:", article_url[0:79])
                    continue
                n = 0
                # Limit download failures
                stop_sec=1 # Initial max wait time in seconds
                while n<2:
                    try:
                        article.parse()
                        n = 99
                    except:
                        n += 1
                        # Initiate download again before new parse attempt
                        article.download()
                        # Timeout for 5 seconds waiting for download
                        t0     = time()
                        tlapse = 0
                        while tlapse<stop_sec:
                            tlapse = time()-t0
                        # Double wait time if needed for next exception
                        stop_sec = stop_sec+1
                if n != 99:
                    if display:
                        print("Cannot download:", article_url[0:79])
                    n_to_review -= 1
                    continue
                article.nlp()
                keywords = article.keywords
                title    = article.title
                summary  = article.summary
                text     = article.text
                text_lower_case = text.lower()
                if search_type == 'or':
                    found_it = False
                    # Verify the url contains at least one of the key words
                    for word in words:
                        j = text_lower_case.find(word)
                        if j>= 0:
                            found_it = True
                            break
                else: 
                    # search type 'and'
                    found_it = True
                    for word in words:
                        j = text_lower_case.find(word)
                        if j < 0:
                            found_it = False
                            break
                if found_it:
                    # Article contains words and passes filters
                    # Save this article for later full review
                    length   = len(text)
                    df_story = pd.DataFrame([[agency, article_url, length, 
                                              keywords, title, summary, 
                                              text]], 
                                columns=['agency', 'url', 'length', 'keywords',
                                             'title', 'summary', 'text'])
                    # Check for an identical already in the file
                    if df_articles.shape[0]==0:
                        df_articles  = df_articles.append(df_story)
                    else:
                        # Verify this story is not already in df_articles
                        same_story = False
                        for i in range(df_articles.shape[0]):
                            if text==df_articles['text'].iloc[i]:
                                same_story   = True
                                n_to_review -= 1
                                continue
                        if not(same_story):
                            df_articles  = df_articles.append(df_story)
                else:
                    n_to_review -= 1
                    
                print("=", end='')
            n_articles[agency] = [n_to_review, len(article_collection)]
        if display:
            print("\n\nArticles Selected by Agency:")
            for agency in news_urls:
                ratio = str(n_articles[agency][0]) + "/" + \
                        str(n_articles[agency][1])
                ratio = ratio
                print("{:>10s} Articles from {:<s}".
                      format(ratio, agency.upper()))
            print("\nArticles Collected on "+today+":", 
                      df_articles.shape[0],'from', 
                      df_articles['agency'].nunique(), "Agencies.")
            print("\nSize    Agency    Title")
            print("*{:->78s}*".format("-"))
            for i in range(df_articles.shape[0]):
                k = len(df_articles['title'].iloc[i])
                if k > 63:
                    for j in range(25):
                        k = 63-j
                        if df_articles['title'].iloc[i][k] == " ":
                            break
                
                    print("{:>5d} {:<10s} {:<63s}".
                          format(df_articles['length'].iloc[i], 
                                 df_articles['agency'].iloc[i],
                                 df_articles['title' ].iloc[i][0:k]))
                    if len(df_articles['title'].iloc[i])>63:
                        print("                {:<60s}".
                              format(df_articles['title'].iloc[i][k:120]))
                else:
                    print("{:>5d} {:<10s} {:<s}".
                          format(df_articles['length'].iloc[i], 
                                 df_articles['agency'].iloc[i],
                                 df_articles['title' ].iloc[i]))
                print("")
            print("*{:->78s}*".format("-"))
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

    def newsapi_get_urls(apikey, search_words, urls=None):
        try:
            api = NewsApiClient(api_key=apikey)
        except:
            raise RuntimeError("APIKEY Invalid")
        if len(search_words)==0 or search_words==None:
            raise RuntimeError("No Search Words")
        print("Searching agencies for pages containing:", search_words)
        # This is my API key, each user must request their own
        # API key from https://newsapi.org/account
        api = NewsApiClient(api_key=apikey)
        api_urls  = []
        # Note that newsapi only draws articles from registered sources
        # These require a particular key/value combination in news_urls
        # Even if the url is correct, if the key is not what is registered
        # the search will be rejected for that agency
        if urls == None or urls == 'top_news':
               news_urls = { 
                    'al-jazeera-english':      'http://www.aljazeera.com',
                    'the-huffington-post':     'http://www.huffingtonpost.com',
                    'bloomberg':               'http://www.bloomberg.com',
                    'reuters':                 'http://www.reuters.com', 
                    'cbs-news':                'http://www.cbsnews.com',
                    'usa-today':               'http://www.usatoday.com/news',
                    'cnn':                     'http://us.cnn.com',
                    'abc-news':                'https://abcnews.go.com',
                    'msnbc':                   'http://www.msnbc.com',
                    'nbc-news':                'http://www.nbcnews.com',
                    'the-wall-street-journal': 'http://www.wsj.com',
                    'fox-news':                'http://www.foxnews.com',
                    'associated-press':        'https://apnews.com/'}
        elif urls=='all_us_news':
                news_urls = {}
                sources = api.get_sources()
                n_sources = len(sources['sources'])
                for i in range(n_sources):
                    cay  = sources['sources'][i]['id']
                    val  = sources['sources'][i]['url']
                    lang = sources['sources'][i]['language']
                    ctry = sources['sources'][i]['country']
                    if lang == 'en' and ctry == 'us':
                        news_urls[cay] = val
        else:
                news_urls = urls
        # Iterate over agencies and search words to pull more url's
        # Limited to 300 requests/day - Likely to be exceeded 
        for agency in news_urls:
            domain = news_urls[agency].replace("http://" , "")
            domain = news_urls[agency].replace("https://", "")
            print("{:.<30s} {:<50s}".format(agency, domain))
            for word in search_words:
                # Get articles with q= in them, Limits to 20 URLs
                try:
                    articles = api.get_everything(q=word, language='en',
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
        print("\nFound a total of", n_total, " URLs, of which", n_unique,
              " were unique.")
        return df_urls
    
    def request_pages(df_urls):
        web_pages = []
        for i in range(len(df_urls)):
            u   = df_urls.iloc[i]
            url = u[2]
            short_url = url[0:50]
            short_url = short_url.replace("https//", "")
            short_url = short_url.replace("http//", "")
            n = 0
            # Allow for a maximum of 2 download failures
            stop_sec=1 # Initial max wait time in seconds
            while n<2:
                try:
                    r = requests.get(url, timeout=(stop_sec))
                    if r.status_code == 404:
                        print("-->HTML ERROR 404", short_url)
                        raise ValueError()
                    if r.status_code == 200:
                        print("Obtained: "+short_url)
                    else:
                        print("-->Web page: "+short_url+" status code:", \
                                  r.status_code)
                    n=99
                    continue # Skip this page
                except:
                    if r.status_code == 404:
                        n=99
                        continue
                    n += 1
                    # Timeout waiting for download
                    t0 = time()
                    tlapse = 0
                    print("Waiting", stop_sec, "sec")
                    while tlapse<stop_sec:
                        tlapse = time()-t0
                    # Double wait time if needed for next exception
                    stop_sec = stop_sec*2
            if n != 99:
                # download failed skip this page
                continue
            # Page obtained successfully
            html_page = r.text
            page_text = scrape.clean_html(html_page)
            web_pages.append([url, page_text])
        df_www  = pd.DataFrame(web_pages, columns=['url', 'text'])
        n_total  = len(df_urls)
        # Remove duplicates
        df_www  = df_www.drop_duplicates('url')
        n_unique = len(df_urls)
        print("Found a total of", n_total, " web pages, of which", n_unique,\
              " were unique.")
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
