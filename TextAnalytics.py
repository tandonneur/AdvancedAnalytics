#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:49:22 2019

@author: EJones
"""

import sys
import numpy  as np

import matplotlib.pyplot as plt
import random
import string
# Install nltk using conda install nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

from wordcloud import WordCloud 
# Install using conda install wordcloud
# from wordcloud import STOPWORDS


class TextAnalytics(object):  
    def __init__(self, synonyms=None, stop_words=None, pos=True, stem=True):
        if synonyms!=None and type(synonyms) != dict:
            raise RuntimeError("\n"+\
                "***Invalid TextAnalytics Initialization.\n"+\
                "***synonyms are invalid, expecting 'Dictionary'.")
            sys.exit()
        if stop_words!=None and type(stop_words) != list:
            raise RuntimeError("\n"+\
                "***Invalid TextAnalytics Initialization.\n"+\
                "***stop words are invalid, expecting 'List'.")
            sys.exit()
        if pos!=True and pos!=False:
            raise RuntimeError("\n"+\
                "***Invalid TextAnalytics Initialization.\n"+\
                "***POS is not True or False")
            sys.exit()
        if stem!=True and stem!=False:
            raise RuntimeError("\n"+\
                "***Invalid TextAnalytics Initialization.\n"+\
                "***STEM is not True or False")
            sys.exit()
        if synonyms==None:
            self.synonyms_ = {}
        else:
            self.synonyms_ = synonyms
        if stop_words==None:
            self.stop_words_ = []
        else:
            self.stop_words_ = stop_words
        self.pos_  = pos
        self.stem_ = stem
        
    def preprocessor(s): 
        # Preprocess String s
        s = s.lower()
        # Replace not contraction with not
        s = s.replace("'nt", "n't")
        s = s.replace("can't", "can not")
        s = s.replace("cannot", "can not")
        s = s.replace("won't", "will not")
        s = s.replace("did't", "did not")
        s = s.replace("couldn't", "could not")
        s = s.replace("shouldn't", "should not")
        s = s.replace("wouldn't", "would not")
        s = s.replace("n't", " not")
        punc = string.punctuation
        for i in range(len(punc)):
            s = s.replace(punc[i], ' ')
        return s
    
    # Customized NLP Processing
    def analyzer(self, s):
        # Synonym List - Map Keys to Values
        syns = { \
                  'wont':'would not', \
                  'cant':'can not', 'cannot':'can not', \
                  'couldnt':'could not', \
                  'shouldnt':'should not', \
                  'wouldnt':'would not'}
        syns.update(self.synonyms_)
        
        # Preprocess String s
        s = TextAnalytics.preprocessor(s)
    
        # Tokenize 
        tokens = word_tokenize(s)
        #tokens = [word.replace(',','') for word in tokens ]
        tokens = [word for word in tokens if ('*' not in word) and \
                  ("''" != word) and ("``" != word) and \
                  (word!='description') and (word !='dtype') \
                  and (word != 'object') and (word!="'s")]
        
        # Map synonyms
        for i in range(len(tokens)):
            if tokens[i] in syns:
                tokens[i] = syns[tokens[i]]
       
        # Remove stop words
        punctuation = list(string.punctuation)+['..', '...']
        pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
        others   = ["'d", "co", "ed", "put", "say", "get", "can", "become",\
                    "los", "sta", "la", "use", "iii", "else", "could", \
                    "would", "come", "take"]
        stop = stopwords.words('english') + \
                punctuation + pronouns + others + self.stop_words_
        filtered_terms = [word for word in tokens if (word not in stop) and \
                      (len(word)>1) and (not word.replace('.','',1).isnumeric()) \
                      and (not word.replace("'",'',2).isnumeric())]
        
        # Lemmatization & Stemming - Stemming with WordNet POS
        # Since lemmatization requires POS need to set POS
        if self.pos_ == True or self.stem_ == True:
            tagged_tokens = pos_tag(filtered_terms, lang='eng')
        else:
            tagged_tokens = filtered_terms
        # Stemming with for terms without WordNet POS
        if self.stem_ == True:
            stemmer = SnowballStemmer("english")
            wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
            wnl = WordNetLemmatizer()
            stemmed_tokens = []
            for tagged_token in tagged_tokens:
                term = tagged_token[0]
                pos  = tagged_token[1]
                pos  = pos[0]
                try:
                    pos   = wn_tags[pos]
                    z = wnl.lemmatize(term, pos=pos)
                    if z not in stop:
                        stemmed_tokens.append(z)
                except:
                    z = stemmer.stem(term)
                    if z not in stop:
                        stemmed_tokens.append(z)
        else:
            stemmed_tokens = tagged_tokens
        return stemmed_tokens
    
    def score_topics(self, v, tf_matrix):
        # ***** SCORE DOCUMENTS ***** Score = TF x V
        ntopics  = v.shape[0]            # Number of topic clusters
        ndocs    = tf_matrix.shape[0]    # Number of documents
        
        # doc_scores is returned as a list of lists
        # The number of lists is ndocs
        # Each list has ntopics+1 values, where the first is 
        # the cluster number.  The others are the document's 
        # scores for each cluster.
        doc_scores   = [[0]*(ntopics+1)] * ndocs
        # topic_counts is a list of the number of documents
        # for each cluster
        topic_counts =  [0]*ntopics
        
        for d in range(ndocs):
            idx       = 0
            max_score = -1e+64
            # Calculate Review Score
            k = tf_matrix[d].nonzero()
            nwords    = len(k[0])
            doc_score = [0]*(ntopics+1)
            # get scores for rth doc, ith topic
            totalscore = 0
            for i in range(ntopics):
                score  = 0
                for j in range(nwords):
                    l = k[1][j]
                    if tf_matrix[d,l] != 0:
                            score += tf_matrix[d,l] * v[i][l]
                doc_score[i+1] = score
                abscore        = abs(score)
                totalscore    += abscore
                if abscore > max_score:
                    max_score  = abscore
                    idx        = i
            # Save review's highest scores
            # Normalize topic score to sum to 1 (probabilities)
            doc_score[1:] = np.abs(doc_score[1:])/totalscore
            doc_score [0] = idx
            doc_scores[d] = doc_score
            topic_counts[idx] += 1
        # Display the number of documents for each cluster
        print('{:<6s}{:>8s}{:>8s}'.format("TOPIC", "REVIEWS", "PERCENT"))
        for i in range(ntopics):
            print('{:>3d}{:>10d}{:>8.1%}'.format((i+1), topic_counts[i], \
                  topic_counts[i]/ndocs))
        return doc_scores # ndocs x (ntopics+1)

    def display_topics(self, lda, terms, n_terms=15, \
                       word_cloud=False, mask=None):
        for topic_idx, topic in enumerate(lda):
            message  = "Topic #%d: " %(topic_idx+1)
            print(message)
            abs_topic = abs(topic)
            if type(terms[0])==tuple:
                topic_terms_sorted = \
                    [[terms[i][0], topic[i]] \
                         for i in abs_topic.argsort()[:-n_terms - 1:-1]]
            else:
                topic_terms_sorted = \
                    [[terms[i], topic[i]] \
                         for i in abs_topic.argsort()[:-n_terms - 1:-1]]
                
            k = 5
            n = int(n_terms/k)
            m = n_terms - k*n
            for j in range(n):
                l = k*j
                message = ''
                for i in range(k):
                    if topic_terms_sorted[i+l][1]>0:
                        word = "+"+topic_terms_sorted[i+l][0]
                    else:
                        word = "-"+topic_terms_sorted[i+l][0]
                    message += '{:<15s}'.format(word)
                print(message)
            if m> 0:
                l = k*n
                message = ''
                for i in range(m):
                    if topic_terms_sorted[i+l][1]>0:
                        word = "+"+topic_terms_sorted[i+l][0]
                    else:
                        word = "-"+topic_terms_sorted[i+l][0]
                    message += '{:<15s}'.format(word)
                print(message)
            print("")
            if word_cloud:
                topic_cloud = {}
                for i in range(n_terms):
                    topic_cloud[topic_terms_sorted[i][0]] = \
                                topic_terms_sorted[i][1]
                # Show Word Cloud based dictionary with term Frequencies
                TextAnalytics.word_cloud_dic(topic_cloud, mask=mask, \
                                             max_words=n_terms)
        return
    
    def shades_of_grey(word, font_size, position, orientation, \
                       random_state=None, **kwargs):
        return "hsl(0, 0%%, %d%%)" % random.randint(60,1000)
    
    def word_cloud_string(s, mask=None, bg_color="maroon", \
                          stopwords=None, max_words=30):
        wcloud = WordCloud(background_color=bg_color,   \
               mask=mask, max_words=max_words, stopwords=stopwords, \
               max_font_size=40,  prefer_horizontal=0.9,  \
               min_font_size=10, relative_scaling=0.5,    \
               width=400, height=200, scale=1, margin=10, random_state=12345)
        # Show Word Cloud based term Frequencies (unweighted)
        wcloud.generate(s)
        plt.imshow( \
        wcloud.recolor(color_func=TextAnalytics.shades_of_grey, \
                       random_state=12345), interpolation="bilinear")
        plt.axis("off")
        plt.figure()
        plt.show()
        return
    
    def word_cloud_dic(td, mask=None, bg_color="maroon", max_words=30):
        wcloud = WordCloud(background_color=bg_color,   \
               mask=mask, max_words=max_words, \
               max_font_size=40,  prefer_horizontal=0.9,  \
               min_font_size=10, relative_scaling=0.5,    \
               width=400, height=200, scale=1, margin=10, \
               random_state=12345)
        # Show Word Cloud based term Frequencies (unweighted)
        wcloud.generate_from_frequencies(td)
        plt.imshow( \
                   wcloud.recolor(color_func=TextAnalytics.shades_of_grey, \
                            random_state=12345), interpolation="bilinear")
        plt.axis("off")
        plt.figure()
        plt.show()
        return
    
    
    # Converts a Term-Frequency matrix into a dictionary
    # tf is a sparse term-frequency matrix
    # terms is a list of term names (strings)
    # Returns dictionary where the terms are keys and value frequencies
    def term_dic(tf, terms, scores=None):
        td   = {}
        for i in range(tf.shape[0]):
            # Iterate over the terms with nonzero scores
            term_list = tf[i].nonzero()[1]
            if len(term_list)>0:
                if scores==None:
                    for t in np.nditer(term_list):
                        if td.get(terms[t]) == None:
                            td[terms[t]] = tf[i,t]
                        else:
                            td[terms[t]] += tf[i,t]
                else:
                    for t in np.nditer(term_list):
                        score = scores.get(terms[t])
                        if score != None:
                            # Found Sentiment Word
                            score_weight = abs(scores[terms[t]])
                            if td.get(terms[t]) == None:
                                td[terms[t]] = tf[i,t]  * score_weight
                            else:
                                td[terms[t]] += tf[i,t] * score_weight
        return td
