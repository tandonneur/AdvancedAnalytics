#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes in AdvancedAnalytics 
ReplaceImputeEncode
linreg
logreg
DecisionTree
NeuralNetwork
TextAnalytics
Sentiment
News
Calculate

@author: Edward R Jones
@version 1.0
@copyright 2018 - Edward R Jones, all rights reserved.
"""

import sys
import warnings
import numpy  as np
import pandas as pd
from math import sqrt, log, pi
from sklearn import preprocessing
from sklearn.impute  import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report 
from copy import deepcopy #Used to create sentiment word dictionary

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

#from wordcloud import WordCloud 
# Install using conda install wordcloud
# from wordcloud import STOPWORDS

import re
import requests  # install using conda instal requests
#import newspaper # install using pip install newspaper3k
#from newspaper import Article

# newsapi requires tiny\segmenter:  pip install tinysegmenter==0.3
# Install newsapi using:  pip install newsapi-python
# from newsapi import NewsApiClient # Needed for using API Feed
from time import time
from datetime import date

class DM:
    Interval = 'I'
    Binary   = 'B'
    Nominal  = 'N'
    ID       = 'Z'
    Text     = 'T'
    Ignore   = 'Z'
    interval = 'I'
    binary   = 'B'
    nominal  = 'N'
    text     = 'T'
    ignore   = 'Z'

"""
Class ReplaceImputeEncode

@parameters:
    *** __init__() ***
    data_map - The metadata dictionary.  If not passed, a metadata
        dictionary is created from the data. Each column in the data must
        be described in the metadata.  The dictionary keys correspond to
        the names for each column.  The value for each key is a list of
        three objects:
            1.  A character indicating data type - I=interval, B=binary, 
                N=nominal, and all other indicators are ignored.
            2.  A tuple containing the lower and upper bounds for integer
                attributes, or a list of allowed categories for binary
                and nominal attributes.  These can be integers or strings.
                
    nominal_encoding - Can be 'one-hot', 'SAS' or None.
    
    interval_scale   - Can be 'std', 'robust' or None.
    
    drop             - True of False.  True drops the last nominal encoded
                       columns.  False keeps all nominal encoded columns.
    
    display          - True or False.  True displays the number of missing 
                        and outliers found in the data.
                        
    *** fit() & fit_transform () ***
    df  - a pandas DataFrame containing the data description
                        by the metadata found in attributes_map (required)
    attributes_map - See above description.
    
@Cautions:
    The incoming DataFrame, df, and the attributes_map are deepcopied to
    ensure that changes to the DataFrame are only held within the class
    object self.copy_df.  The attributes_map is deepcopied into
    self.features_map.  The categorical values are change to numeric and
    the number of missing and number of outliers are also changed from zero.
"""

class ReplaceImputeEncode(object):
    
    def __init__(self, data_map=None, nominal_encoding=None, \
                 interval_scale=None, drop=False, display=False, \
                 no_impute=None):
        if interval_scale=='None' or interval_scale=='none':
            self.interval_scale=None
        else:
            self.interval_scale=interval_scale
        self.go_flag = False
        self.features_map = data_map
        self.drop    = drop
        self.display = display
        self.interval_scale = interval_scale
        self.no_impute = no_impute
        if nominal_encoding=='None' or nominal_encoding=='none':
            self.nominal_encoding = None
        else:
            self.nominal_encoding = nominal_encoding
        #nominal_encoding can be 'SAS' or 'one-hot'
        if nominal_encoding != 'SAS' and nominal_encoding != 'one-hot' \
            and nominal_encoding != None:
            raise RuntimeError("***Call to ReplaceImputeEncode invalid. "+\
                 "***   nominal_encoding="+nominal_encoding+" is invalid."+\
                 "***   must use 'one-hot' or 'SAS'")
            sys.exit()
        if interval_scale != 'std' and interval_scale != 'robust' \
            and interval_scale != None:
            raise RuntimeError("***Call to ReplaceImputeEncode invalid. "+\
                     "***   interval_scale="+interval_scale+" is invalid."+\
                     "***   must use None, 'std' or 'robust'")
            sys.exit()
        if nominal_encoding=='SAS' and drop==False:
            raise RuntimeError("***Call to ReplaceImputeEncode invalid. "+\
                  "***nominal_encoding='SAS' requested with drop=False "+\
                  "***'SAS' encoding requires drop=True")
            sys.exit()
        if data_map==None:
            print("Attributes Map is required.")
            print("Please pass map using data_map attribute.")
            print("If one is not available, try creating one using "+ \
                  "call to draft_features_map(df)")
            return
        #self.features_map = deepcopy(data_map)
        self.features_map = data_map
        self.interval_attributes = []
        self.nominal_attributes  = []
        self.binary_attributes   = []
        self.onehot_attributes   = []
        self.hot_drop_list       = []
        self.missing_counts      = {}
        self.outlier_counts      = {}
        for feature,v in self.features_map.items():
            # Initialize data map missing and outlier counters to zero
            self.missing_counts[feature] = 0
            self.outlier_counts[feature] = 0
            self.regex = re.compile("[BINTZ]", re.IGNORECASE)
            t = str(v[0])
            t = t.upper()
            if not self.regex.match(t):
                raise RuntimeError( \
                  "***Data Map in call to ReplaceImputeEncode invalid. "+\
                  "***Data Type for '"+ feature + "' is not a string. "+\
                  "***Should be 'B', 'I', 'N', 'T', or 'Z'")
            if t=='I' or t=='0':
                self.interval_attributes.append(feature)
            else:
                if t=='B' or t=='1':
                    self.binary_attributes.append(feature)
                else:
                    if t!='B' and t!='N' and t!='1' and t!='2': 
                        # Ignore, don't touch this attribute
                        continue
                    # Attribute must be Nominal
                    self.nominal_attributes.append(feature)
                    # Setup column names for encoding, all but the last one
                    n_cat = len(v[1])
                    if self.drop == True:
                        n_cat -= 1
                    for i in range(n_cat):
                        if type(v[1][i])==int:
                            my_str = feature+str(v[1][i])
                        else:
                            my_str = feature+("%i" %i)+":"+str(v[1][i])[0:4]
                        self.onehot_attributes.append(my_str)

        self.n_interval = len(self.interval_attributes)
        self.n_binary   = len(self.binary_attributes)
        self.n_nominal  = len(self.nominal_attributes)
        self.n_onehot   = len(self.onehot_attributes)
        self.cat        = self.n_binary + self.n_nominal
        self.col = []
        for i in range(self.n_interval):
            self.col.append(self.interval_attributes[i])
        for i in range(self.n_binary):
            self.col.append(self.binary_attributes[i])
        if self.nominal_encoding==None:
            for i in range(self.n_nominal):
                self.col.append(self.nominal_attributes[i])
        else:
            for i in range(self.n_onehot):
                self.col.append(self.onehot_attributes[i])

        self.go_flag = True
        
    def fit(self, df, data_map=None):
        self.df_copy = deepcopy(df)
        #self.df_copy = df
        if data_map==None and self.features_map==None:
            warnings.warn("  Call to ReplaceImputeEncode missing required"+\
              " Data Map.\n  Using draft_features_map() to construct "+\
              "Data Map.\n"+\
              "  Inspect the generated Data Map for Consistency.")
            self.features_map = \
                    self.draft_features_map(self.df_copy, display_map=True)
        else: 
            if self.features_map == None:
                self.features_map = data_map
                #self.features_map = deepcopy(data_map)

        self.interval_attributes = []
        self.nominal_attributes  = []
        self.binary_attributes   = []
        self.onehot_attributes   = []
        self.hot_drop_list       = []
        for feature,v in self.features_map.items():
            t=str(v[0])
            t=t.upper()
            if not (self.regex.match(t)):
                raise RuntimeError( \
                  "***Data Map in call to ReplaceImputeEncode invalid.\n"+\
                  "***   Data Type for '"+ feature + "' is not a string.\n"+\
                  "***   Should be 'B', 'I', 'N', 'T', or 'Z'")
            if t=='I' or t=='0':
                self.interval_attributes.append(feature)
            else:
                if t=='B' or t=='1':
                    self.binary_attributes.append(feature)
                else:
                    if t=='N' or t=='2':
                        self.nominal_attributes.append(feature)
                        for i in range(len(v[1])):
                            if type(v[1][i])==int:
                                my_str = feature+str(v[1][i])
                            else:
                                my_str = feature+("%i" %i)+":"+ \
                                                str(v[1][i])[0:4]
                            self.onehot_attributes.append(my_str)
                        if self.drop==True:
                            self.hot_drop_list.append(my_str)
                    else:
                        if t=='T' or t=='Z':
                            continue
                        else:
                        # Data Map Invalid
                            raise RuntimeError( \
                  "***Data Map in call to ReplaceImputeEncode invalid.\n"+\
                  "***   Data Type for '"+ feature + "' invalid")
                        sys.exit()
        self.n_interval = len(self.interval_attributes)
        self.n_binary   = len(self.binary_attributes)
        self.n_nominal  = len(self.nominal_attributes)
        self.n_onehot   = len(self.onehot_attributes)
        self.cat        = self.n_binary + self.n_nominal
        self.n_obs      = df.shape[0]
        self.n_ignored  = df.shape[1] - \
                         self.n_interval-self.n_binary-self.n_nominal
        self.col = []
        for i in range(self.n_interval):
            self.col.append(self.interval_attributes[i])
        for i in range(self.n_binary):
            self.col.append(self.binary_attributes[i])
        if self.nominal_encoding==None:
            for i in range(self.n_nominal):
                self.col.append(self.nominal_attributes[i])
        else:
            for i in range(self.n_onehot):
                self.col.append(self.onehot_attributes[i])

        if self.display:
            print("\n********** Data Preprocessing ***********")
            print("Features Dictionary Contains:\n%i Interval," \
                  %self.n_interval, "\n%i Binary," %self.n_binary,\
                  "\n%i Nominal, and" %self.n_nominal, \
                  "\n%i Excluded Attribute(s).\n" %self.n_ignored)
            print("Data contains %i observations & %i columns.\n" %df.shape)
        self.initial_missing = df.isnull().sum()
        self.feature_names = np.array(df.columns.values)
        for feature in self.feature_names:
            if self.initial_missing[feature]>(self.n_obs/2):
                warnings.warn(feature+":has more than 50\% missing.")
        # Initialize number missing in attribute_map
        for feature,v in self.features_map.items():
            self.missing_counts[feature] = self.initial_missing[feature]
            #v[2][0] = self.initial_missing[feature]

        # Scan for outliers among interval attributes
        nan_map = df.isnull()
        for index in df.iterrows():
            i = index[0]
        # Check for outliers in interval attributes
            for feature, v in self.features_map.items():
                if nan_map.loc[i,feature]==True:
                    continue
                if v[0]=='i' or v[0]=='I': # Interval Attribute
                    if type(v[1]) != tuple:
                       raise RuntimeError("\n" +\
                          "***Call to ReplaceImputeEncode invalid.\n"+\
                          "***   Attribute Map has invalid description " +\
                          "for " +feature)
                       sys.exit()
                    l_limit = v[1][0]
                    u_limit = v[1][1]
                    if df.loc[i,feature]>u_limit or df.loc[i,feature]<l_limit:
                        self.outlier_counts[feature] += 1
                        self.df_copy.loc[i,feature] = None
                else: 
                    if v[0]!='b' and v[0]!='n' and v[0]!='B' and v[0]!='N': 
                        # don't touch this attribute
                        continue
                    # Categorical Attribute
                    in_cat = False
                    for cat in v[1]:
                        if df.loc[i,feature]==cat:
                            in_cat=True
                    if in_cat==False:
                        self.df_copy.loc[i,feature] = None
                        self.outlier_counts[feature] += 1
        if self.display:
            print("\nAttribute Counts")
            max_label = 0
            for k, v in self.features_map.items():
                if len(k) > max_label:
                    max_label = len(k)
            max_label += 2
            label_format = ("{:.<%i" %(max_label+5))+"s}{:>8s}{:>10s}" 
            print(label_format.format('', 'Missing', 'Outliers'))
            label_format = ("{:.<%i" %max_label)+"s}{:10d}{:10d}"
            for k,v in self.features_map.items():
                print(label_format.format(k, self.missing_counts[k], \
                                          self.outlier_counts[k]))
                #print(label_format.format(k, v[2][0], v[2][1]))
            
    def draft_data_map(self, df, display_map=True):
        feature_names = np.array(df.columns.values)
        draft_features_map = {}
        print("\nGenerating DATA_MAP for use in ReplaceImputeEncode."+\
                      "\nRecommend inspecting generated map.")
        for feature in feature_names:
            n = df[feature].value_counts()
            if type(df[feature].iloc[0]) != str:
                min_ = round(df[feature].min()-0.5,4)
                max_ = round(df[feature].max()+0.5,4)
            if len(n) > 5 and type(df[feature].iloc[0]) !=str:
                # Attribute is Interval
                draft_features_map[feature]=['I',(min_, max_)]
                continue
            if len(n) == 2: 
                # Attribute is Binary
                val1 = df[feature].unique()[0]
                val2 = df[feature].unique()[1]
                draft_features_map[feature]=['B',(val1, val2)]
            else:
                # Attribure is Nominal or Ordinal
                val = []
                a   = df[feature].unique()
                if len(a) < 30:
                    for i in range(len(a)):
                        if type(a[i]) == str:
                            val.append(a[i])
                    categories = tuple(val)
                    draft_features_map[feature]=['N',categories]
                else:
                    # Set attribute to text field
                    draft_features_map[feature]=['T',('')]
        if display_map:
            # print the features map
            print("************* FEATURE MAP **************\n")
            last_feature = feature_names[-1]
            print("attribute_map = {")
            for feature,v in draft_features_map.items():
                if feature==last_feature:
                    print("\t'"+feature+"':", v, "}\n")
                else:
                    print("\t'"+feature+"':", v, ",")
                    
            for feature, v in draft_features_map.items():
                if v[0] == 'i' or v[0]=='I':
                    print(feature+": Assumed Interval Attribute")
                    print("    with lower bound ", v[1][0], "and", \
                          "upper bound.", v[1][1])
                if v[0] == 'b' or v[0]=='B':
                    print(feature+": Assumed Binary Attribute.")
                if v[0] == 'n' or v[0]=='N':
                    print(feature+": Assumed Nominal Attribute with", \
                          len(v[1]), "Categories.")
                print("")
        return draft_features_map
    
    def display_data_map(self):
        # print the features map
        print("************* DATA MAP **************\n")
        for feature,v in self.features_map.items():
            print("'"+feature+"':", v, "\n")
            
    def impute(self):
        self.impute_interval()
        self.impute_binary()
        self.impute_nominal()
        self.imputed_data()
            
    def impute_interval(self):
        if (self.n_interval==0):
            self.imputed_interval_data = np.empty((self.n_obs, 0))
            return
        # Put the interval data from the dataframe into a numpy array
        #depricated= self.df_copy.as_matrix(columns=self.interval_attributes)
        interval_data= self.df_copy[self.interval_attributes].values
        # Create the Imputer for the Interval Data
        #self.interval_imputer = preprocessing.Imputer(strategy='mean')
        self.interval_imputer = SimpleImputer(strategy='mean')
        # Impute the missing values in the Interval data
        self.imputed_interval_data = \
            self.interval_imputer.fit_transform(interval_data)
    def impute_binary(self):
        if (self.n_binary==0):
            self.imputed_binary_data = np.empty((self.n_obs, 0))
            return
        # Put the nominal and binary data from the dataframe into a numpy array
        #cat_df = df[self.binary_attributes]
        cat_df = pd.DataFrame(columns=self.binary_attributes)
        for feature in self.binary_attributes:
            cat_df[feature]= self.df_copy[feature].astype('category').cat.codes
            cat_df.loc[cat_df[feature]==-1, feature] = None
        #Depricated cat_array = cat_df.as_matrix()
        cat_array = cat_df.values
        # Create Imputer for Categorical Data
        #cat_imputer = preprocessing.Imputer(strategy='most_frequent')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        # Impute the missing values in the Categorical Data
        self.imputed_binary_data = \
            cat_imputer.fit_transform(cat_array)
            
    def impute_nominal(self):
        if (self.n_nominal==0):
            self.imputed_nominal_data = np.empty((self.n_obs, 0))
            return
        # Put the nominal and binary data from the dataframe into a numpy array
        cat_df  = pd.DataFrame(columns=self.nominal_attributes)
        for feature in self.nominal_attributes:
            cat_df[feature]= self.df_copy[feature].astype('category').cat.codes
            cat_df.loc[cat_df[feature]==-1, feature] = None
        #Depricated cat_array = cat_df.as_matrix()
        #print(cat_array[0])
        cat_array = cat_df.values
        # Create Imputer for Categorical Data
        #cat_imputer = preprocessing.Imputer(strategy='most_frequent')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        # Impute the missing values in the Categorical Data
        self.imputed_nominal_data = \
            cat_imputer.fit_transform(cat_array)
            
    def imputed_data(self):
        # Bring Interval and Categorial Data Together into a dataframe
        # The Imputed Data
        # Col is not the same as self.col.  col contains the main attribute
        # names, self.col contains the one-hot names
        col = self.interval_attributes + self.binary_attributes + \
                self.nominal_attributes
        # if no_impute is given, replace these attributes with their 
        # original, missing values
        if self.no_impute != None:
            idx = []
            for i in range(len(self.no_impute)):
                idx.append(-1)
            for i in range(len(col)):
                for j in range(len(self.no_impute)):
                    if col[i] == self.no_impute[j]:
                        idx[j] = i
                        break
            for j in range(len(self.no_impute)):
                k = idx[j]
                if k < 0:
                    warnings.warn("  \nArgument "+self.no_impute[j]+ \
                                  " in 'no_impute' is invalid.\n")
                    break
                if k<self.n_interval:
                    for i in range(self.n_obs):
                        self.imputed_interval_data[i,k] = \
                             self.df_copy[self.no_impute[j]][i]
                else:
                    if k < self.n_interval + self.n_binary:
                        k = k - self.n_interval
                        for i in range(self.n_obs):
                            self.imputed_binary_data[i,k] = \
                                 self.df_copy[self.no_impute[j]][i]
                    else:
                        k = k - self.n_interval - self.n_binary
                        for i in range(self.n_obs):
                            self.imputed_nominal_data[i,k] = \
                                 self.df_copy[self.no_impute[j]][i]
       
        self.data_imputed= \
                np.hstack((self.imputed_interval_data,\
                           self.imputed_binary_data, \
                           self.imputed_nominal_data))

        self.imputed_data_df = \
                pd.DataFrame(self.data_imputed, columns=col)
            
    def scale_encode(self):
        self.standardize_interval()
        self.encode_binary()
        self.encode_nominal()
        self.encoded_data()
            
    def standardize_interval(self):
        if (self.n_interval==0 or self.interval_scale==None):
            self.scaled_interval_data = self.imputed_interval_data
            return
        # Standardize Interval Data using Z-Scores
        if self.interval_scale=='std':
            scaler = preprocessing.StandardScaler() 
            scaler.fit(self.imputed_interval_data)
            self.scaled_interval_data = \
                scaler.transform(self.imputed_interval_data)
        # Standardize Interval Data using median and IQR
        if self.interval_scale=='robust':
            scaler = preprocessing.RobustScaler() 
            scaler.fit(self.imputed_interval_data)
            self.scaled_interval_data = \
                scaler.transform(self.imputed_interval_data)
            
    def encode_binary(self):
        # Uses 1 and -1 encoding for binary instead of 0, 1
        # SAS uses the 1, -1 convention
        if self.n_binary == 0 or self.nominal_encoding == None:
            return
        if self.nominal_encoding == 'SAS':
            if self.n_binary == 0:
                return
            for j in range(self.n_binary):
                for i in range(self.n_obs):
                    if self.imputed_binary_data[i,j] == 1:
                        self.imputed_binary_data[i,j] = -1
                    else:
                        self.imputed_binary_data[i,j] = 1
              
    def encode_nominal(self):
        if (self.n_nominal==0 or self.nominal_encoding==None):
            return
        # Create an instance of the OneHotEncoder & Selecting Attributes
        # Attributes must all be non-negative integers
        # Missing values may show up as -1 values, which will cause an error
        onehot = preprocessing.OneHotEncoder(categories='auto')
        self.hot_array = \
                onehot.fit_transform(self.imputed_nominal_data).toarray()
        n_features = []
        nominal_categories = 0
        for i in range(self.n_nominal):
            feature = self.nominal_attributes[i]
            v = self.features_map[feature]
            n_features.append(len(v[1]))
            nominal_categories += len(v[1])
        if nominal_categories != self.hot_array.shape[1]:
            raise RuntimeError('  Call to ReplaceImputeEncode Invalid '+ \
               '  Number of one-hot columns is', self.hot_array.shape[1], \
               'but nominal categories is ', nominal_categories, \
               '  Data map might contain nominal attributes with '+ \
               'invalid categories or some not found in the data.')
            sys.exit()
            
        # SAS Encoding subtracts the last one-hot vector from the others, for
        # each nominal attribute.
        if self.nominal_encoding == 'SAS':
            self.sas_encoded = \
                np.zeros((self.n_obs, (self.n_onehot-self.n_nominal)))
            ilast = -1
            idx1  = 0
            idx2  = 0
            for l in range(self.n_nominal):
                m = n_features[l]
                ilast = ilast + m
                for j in range(m-1):
                        for i in range(self.n_obs):
                            last = self.hot_array[i,ilast]
                            self.sas_encoded[i,idx1] = \
                                            self.hot_array[i,idx2] - last
                        idx1 += 1
                        idx2 += 1
                idx2 += 1
            
    def encoded_data(self):
        # Bring encoded and scaled data together into a dataframe
        # The Imputed and Encoded Data
        if self.n_nominal==0:
            if self.interval_scale==None:
                self.data_encoded = np.hstack((self.imputed_interval_data, \
                                       self.imputed_binary_data))
            else:
                self.data_encoded = np.hstack((self.scaled_interval_data, \
                                       self.imputed_binary_data))
            if self.drop==True:
                for i in range(self.n_nominal):
                    self.col.remove(self.hot_drop_list[i])
                    
        if self.n_nominal>0 and self.nominal_encoding==None:
            if self.interval_scale==None:
                self.data_encoded = np.hstack((self.imputed_interval_data, \
                                       self.imputed_binary_data, \
                                       self.imputed_nominal_data))
            else:
                self.data_encoded = np.hstack((self.scaled_interval_data, \
                                       self.imputed_binary_data, \
                                       self.imputed_nominal_data))
                
        if self.n_nominal>0 and self.nominal_encoding == 'SAS':
            if self.interval_scale==None:
                self.data_encoded = np.hstack((self.imputed_interval_data, \
                                       self.imputed_binary_data, \
                                       self.sas_encoded))
            else:
                self.data_encoded = np.hstack((self.scaled_interval_data, \
                                       self.imputed_binary_data, \
                                       self.sas_encoded))
            if self.drop==True:
                for i in range(self.n_nominal):
                    self.col.remove(self.hot_drop_list[i])
                    
        if self.n_nominal>0 and self.nominal_encoding == 'one-hot':
            if self.interval_scale==None:
                self.data_encoded = np.hstack((self.imputed_interval_data, \
                                       self.imputed_binary_data, \
                                       self.hot_array))
            else:
                self.data_encoded = np.hstack((self.scaled_interval_data, \
                                       self.imputed_binary_data, \
                                       self.hot_array))
        # data_encoded array ready for conversion to dataframe
        self.encoded_data_df = \
                pd.DataFrame(self.data_encoded, columns=self.col)

        if self.nominal_encoding == 'one-hot' and self.drop==True:
            self.encoded_data_df = \
                self.encoded_data_df.drop(self.hot_drop_list, axis=1)
            for i in range(self.n_nominal):
                self.col.remove(self.hot_drop_list[i])
                
    def transform(self):
        self.impute()
        self.scale_encode()
        return self.encoded_data_df
        
    def fit_transform(self, df, data_map=None):
        self.fit(df, data_map)
        self.transform()
        return self.encoded_data_df

class linreg(object):
    
    def display_coef(lr, X, y, col):
        if col==None:
            raise RuntimeError("  Call to display_coef is Invalid.\n"+\
                  "  List of Coefficient Names is Empty.\n")
            sys.exit()
        print("\nCoefficients")
        if len(col)!=X.shape[1]:
            raise RuntimeError("  Call to display_coef is Invalid.\n"+\
                  "  Number of Coefficient Names is not equal to the"+\
                  " Number of Columns in X")
            sys.exit()
        print("\nCoefficients")
        max_label = len('Intercept')+2
        for i in range(len(col)):
            if len(col[i]) > max_label:
                max_label = len(col[i])
        label_format = ("{:.<%i" %max_label)+"s}{:15.4f}"
        print(label_format.format('Intercept', lr.intercept_))
        for i in range(X.shape[1]):
            print(label_format.format(col[i], lr.coef_[i]))
    
    def display_metrics(lr, X, y, w=None):
        predictions = lr.predict(X)
        n  = X.shape[0]
        p  = X.shape[1] # Notations uses Sheather's convention
        k  = p+2 # need to count the estimated variance and intercept
        print("\nModel Metrics")
        print("{:.<23s}{:15d}".format('Observations', n))
        print("{:.<23s}{:15d}".format('Coefficients', p+1))
        print("{:.<23s}{:15d}".format('DF Error', X.shape[0]-X.shape[1]-1))
        if type(w)==np.ndarray:
            R2 = r2_score(y, predictions, sample_weight=w)
            n = w.sum()
        else:
            R2 = r2_score(y, predictions)
        print("{:.<23s}{:15.4f}".format('R-Squared', R2))
        adjr2 = 1.0-R2 
        adjr2 = ((n-1)/(n-p-1))*adjr2
        adjr2 = 1.0 - adjr2
        print("{:.<23s}{:15.4f}".format('Adj. R-Squared', adjr2))
        if type(w)==np.ndarray:
            MAE = mean_absolute_error(y,predictions, sample_weight=w)
        else:
            MAE = mean_absolute_error(y,predictions)
        print("{:.<23s}{:15.4f}".format('Mean Absolute Error', MAE))
        MAE = median_absolute_error(y,predictions)
        print("{:.<23s}{:15.4f}".format('Median Absolute Error', MAE))
        if type(w)==np.ndarray:
            ASE = mean_squared_error(y,predictions, sample_weight=w)
        else:
            ASE = mean_squared_error(y,predictions)
        print("{:.<23s}{:15.4f}".format('Avg Squared Error', ASE))
        print("{:.<23s}{:15.4f}".format('Square Root ASE', sqrt(ASE)))
        if ASE<1e-20:
            twoLL = -np.inf
            LL    = twoLL
        else:
            twoLL = n*(log(2*pi) + 1.0 + log(ASE))
            LL    = twoLL/(-2.0)
        print("{:.<23s}{:15.4f}".format('Log(Likelihood)', LL))
        AIC  = twoLL + 2*k
        print("{:.<23s}{:15.4f}".format('AIC            ', AIC))
        if (n-k-1)>0:
            AICc = AIC + 2*k*(k+1)/(n-k-1)
        else:
            AICc = AIC + 2*k*(k+1)
            
        print("{:.<23s}{:15.4f}".format('AICc           ', AICc))
        BIC  = twoLL + log(n)*k
        print("{:.<23s}{:15.4f}".format('BIC            ', BIC))
        
    def return_metrics(lr, X, y, w=None):
        metrics = [0, 0, 0, 0]
        predictions = lr.predict(X)
        n  = X.shape[0]
        p  = X.shape[1] # Notations uses Sheather's convention
        k  = p+2 # need to count the estimated variance and intercept
        if type(w)==np.ndarray:
            R2 = r2_score(y, predictions, sample_weight=w)
            n = w.sum()
        else:
            R2 = r2_score(y, predictions)
        adjr2 = 1.0-R2 
        adjr2 = ((n-1)/(n-p-1))*adjr2
        adjr2 = 1.0 - adjr2
        metrics[0] = adjr2
        if type(w)==np.ndarray:
            ASE = mean_squared_error(y,predictions, sample_weight=w)
        else:
            ASE = mean_squared_error(y,predictions)
        if ASE<1e-20:
            twoLL = -np.inf
        else:
            twoLL = n*(log(2*pi) + 1.0 + log(ASE))
        AIC  = twoLL + 2*k
        metrics[1] = AIC
        if (n-k-1)>0:
            AICc = AIC + 2*k*(k+1)/(n-k-1)
        else:
            AICc = AIC + 2*k*(k+1)
            
        metrics[2] = AICc
        BIC  = twoLL + log(n)*k
        metrics[3] = BIC
        return metrics
    
    def display_split_metrics(lr, Xt, yt, Xv, yv, wt=None, wv=None):
        predict_t = lr.predict(Xt)
        predict_v = lr.predict(Xv)
        nt  = Xt.shape[0]
        pt  = Xt.shape[1] # Notations uses Sheather's convention
        kt  = pt+2 # need to count the estimated variance and intercept
        nv  = Xv.shape[0]
        pv  = Xv.shape[1] # Notations uses Sheather's convention
        kv  = pv+2 # need to count the estimated variance and intercept
        print("\n")
        print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', \
                                      'Training', 'Validation'))
        print("{:.<23s}{:15d}{:15d}".format('Observations', \
                                          Xt.shape[0], Xv.shape[0]))
        print("{:.<23s}{:15d}{:15d}".format('Coefficients', \
                                          Xt.shape[1]+1, Xv.shape[1]+1))
        print("{:.<23s}{:15d}{:15d}".format('DF Error', \
                      Xt.shape[0]-Xt.shape[1]-1, Xv.shape[0]-Xv.shape[1]-1))
        R2t = r2_score(yt, predict_t)
        R2v = r2_score(yv, predict_v)
        print("{:.<23s}{:15.4f}{:15.4f}".format('R-Squared', R2t, R2v))
        adjr2t = 1.0-R2t 
        adjr2t = ((nt-1)/(nt-pt-1))*adjr2t
        adjr2t = 1.0 - adjr2t
        adjr2v = 1.0-R2v
        adjr2v = ((nv-1)/(nv-pv-1))*adjr2v
        adjr2v = 1.0 - adjr2v
        print("{:.<23s}{:15.4f}{:15.4f}".format('Adj. R-Squared', \
                      adjr2t, adjr2v))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(yt,predict_t), \
                      mean_absolute_error(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Median Absolute Error', \
                      median_absolute_error(yt,predict_t), \
                      median_absolute_error(yv,predict_v)))
        ASEt = mean_squared_error(yt,predict_t)
        ASEv = mean_squared_error(yv,predict_v)
        print("{:.<23s}{:15.4f}{:15.4f}".format('Avg Squared Error', \
                      ASEt, ASEv))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Square Root ASE', \
                      sqrt(ASEt), sqrt(ASEv)))
        if ASEt<1e-20:
            twoLLt = -np.inf
            LLt    = twoLLt
        else:
            twoLLt = nt*(log(2*pi) + 1.0 + log(ASEt))
            LLt    = twoLLt/(-2.0)
        if ASEv<1e-20:
            twoLLv = -np.inf
            LLv    = twoLLv
        else:
            twoLLv = nv*(log(2*pi) + 1.0 + log(ASEv))
            LLv    = twoLLv/(-2.0)
        print("{:.<23s}{:15.4f}{:15.4f}".format('Log Likelihood', \
                      LLt, LLv))
        AICt  = twoLLt + 2*kt
        AICv  = twoLLv + 2*kv
        print("{:.<23s}{:15.4f}{:15.4f}".format('AIC           ', \
                      AICt, AICv))
        if (nt-kt-1)>0:
            AICct = AICt + 2*kt*(kt+1)/(nt-kt-1)
        else:
            AICct = AICt + 2*kt*(kt+1)
        if (nv-kv-1)>0:
            AICcv = AICv + 2*kv*(kv+1)/(nv-kv-1)
        else:
            AICcv = AICv + 2*kv*(kv+1)
        print("{:.<23s}{:15.4f}{:15.4f}".format('AICc          ', \
                      AICct, AICcv))
        BICt  = twoLLt + log(nt)*kt
        BICv  = twoLLv + log(nv)*kv
        print("{:.<23s}{:15.4f}{:15.4f}".format('BIC           ', \
                      BICt, BICv))
        
class logreg(object):
    
    def display_coef(lr, nx, k, col):
        if col==None:
           raise RuntimeError("  Call to display_coef Invalid\n" +\
                              "  List of Coefficient Names is Empty. ", \
                              "Unable to Display Coefficients.")
           sys.exit()
        if nx<1:
           raise RuntimeError("]n  Call to display_coef Invalid. " +\
                              "  Number of attributes (nx) is invalid. ")
           sys.exit()
        if len(col)!=nx:
            raise RuntimeError("  Call to display_coef is Invalid.\n"+\
                  "  Number of Coefficient Names (col) is not equal to the"+\
                  " Number of Attributes (nx)")
            sys.exit()
        if k<1:
           raise RuntimeError("]n  Call to display_coef Invalid. " +\
                              "  Number of classes (k) is invalid. ")
           sys.exit()
        max_label = len('Intercept')+2
        for i in range(len(col)):
            if len(col[i]) > max_label:
                max_label = len(col[i])
        label_format = ("{:.<%i" %max_label)+"s}{:15.4f}"
        k2 = k
        if k <=2:
            k2 = 1
        for j in range(k2):
            if k == 2:
                print("\nCoefficients:")
            else:
                print("\nCoefficients for Target Class", lr.classes_[j])
            print(label_format.format('Intercept', lr.intercept_[j]))
            for i in range(nx):
                print(label_format.format(col[i], lr.coef_[j,i]))
    
    def display_confusion(conf_mat):
        if len(conf_mat) != 2:
            raise RuntimeError("  Call to display_confustion invalid"+\
               " Argument is not a 2x2 Matrix.")
            sys.exit()
        TP = int(conf_mat[1][1])
        TN = int(conf_mat[0][0])
        FP = int(conf_mat[0][1])
        FN = int(conf_mat[1][0])
        n_neg  = TN + FP
        n_pos  = FN + TP
        n_pneg = TN + FN
        n_ppos = FP + TP
        n_obs  = n_neg + n_pos
        print("\nModel Metrics")
        print("{:.<27s}{:10d}".format('Observations', n_obs))
        acc = np.nan
        pre = np.nan
        tpr = np.nan
        tnr = np.nan
        f1  = np.nan
        misc = np.nan
        miscc = [np.nan, np.nan]
        if n_obs>0:
            acc = (TP+TN)/n_obs
        print("{:.<27s}{:10.4f}".format('Accuracy', acc))
        if (TP+FP)>0:
            pre = TP/(TP+FP)
        print("{:.<27s}{:10.4f}".format('Precision', pre))
        if (TP+FN)>0:
            tpr = TP/(TP+FN)
        print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
        if (TN+FP)>0:
            tnr = TN/(TN+FP)
        print("{:.<27s}{:10.4f}".format('Recall (Specificity)', tnr))
        if (2*TP+FP+FN)>0:
            f1 = 2*TP/(2*TP + FP + FN)
        print("{:.<27s}{:10.4f}".format('F1-Score', f1))
        
        if n_obs>0:
            misc = 100*(FN + FP)/n_obs
        print("{:.<27s}{:9.1f}{:s}".format(\
                'MISC (Misclassification)', misc, '%'))
        if n_neg>0 and n_pos>0:
            miscc = [100*conf_mat[0][1]/n_neg, 100*conf_mat[1][0]/n_pos]
        lrcc  = [0, 1]
        for i in range(2):
            print("{:s}{:.<16.0f}{:>9.1f}{:<1s}".format(\
                  '     class ', lrcc[i], miscc[i], '%'))      

        print("\n\n     Confusion")
        print("       Matrix    ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', lrcc[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', lrcc[i]), end="")
            for j in range(2):
                print("{:>10d}".format(int(conf_mat[i][j])), end="")
            print("")
         
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, lr.classes_)
        #print("\n",cr)
        
    def display_binary_metrics(lr, X, y):
        if len(lr.classes_) != 2:
            raise RuntimeError("  Call to display_binary_metrics invalid"+\
               "\n  Target does not appear to be binary."+\
               "\n  Number of classes is not 2")
            sys.exit()
        z = np.zeros(len(y))
        predictions = lr.predict(X) # get binary class predictions
        conf_mat = confusion_matrix(y_true=y, y_pred=predictions)
        misc = 100*(conf_mat[0][1]+conf_mat[1][0])/(len(y))
        for i in range(len(y)):
            if y[i] == 1:
                z[i] = 1
        #probability = lr.predict_proba(X) # get binary probabilities
        probability = lr._predict_proba_lr(X)
        print("\nModel Metrics")
        print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
        print("{:.<27s}{:10d}".format('Coefficients', X.shape[1]+1))
        print("{:.<27s}{:10d}".format('DF Error', X.shape[0]-X.shape[1]-1))
        if type(y[0])==str:
            raise RuntimeError("  Call to display_binary_metrics invalid"+\
               "  Binary Target is encoded as strings rather than 0 & 1")
            sys.exit()
        print("{:.<27s}{:10.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(z,probability[:, 1])))
        print("{:.<27s}{:10.4f}".format('Avg Squared Error', \
                      mean_squared_error(z,probability[:, 1])))
        acc = accuracy_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Accuracy', acc))
        pre = precision_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Precision', pre))
        tpr = recall_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
        f1 =  f1_score(y,predictions)
        print("{:.<27s}{:10.4f}".format('F1-Score', f1))
        print("{:.<27s}{:9.1f}{:s}".format(\
                'MISC (Misclassification)', misc, '%'))
        n_    = [conf_mat[0][0]+conf_mat[0][1], conf_mat[1][0]+conf_mat[1][1]]
        miscc = [100*conf_mat[0][1]/n_[0], 100*conf_mat[1][0]/n_[1]]
        for i in range(2):
            print("{:s}{:.<16.0f}{:>9.1f}{:<1s}".format(\
                  '     class ', lr.classes_[i], miscc[i], '%'))      

        print("\n\n     Confusion")
        print("       Matrix    ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', lr.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', lr.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_mat[i][j]), end="")
            print("")
         
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, lr.classes_)
        #print("\n",cr)
        
    def display_nominal_metrics(lr, X, y):
        n_classes = len(lr.classes_)
        if n_classes == 2:
            raise RuntimeError("  Call to display_nominal_metrics invalid\n"+\
                "  Target has only 2 classes.")
            sys.exit()
        predict_ = lr.predict(X)
        prob_ = lr._predict_proba_lr(X)
        ase_sum = 0
        misc_ = 0
        misc  = []
        n_    = []
        n_obs = y.shape[0]
        conf_mat = []
        for i in range(n_classes):
            z = []
            for j in range(n_classes):
                z.append(0)
            conf_mat.append(z)
        y_ = np.array(y) # necessary because yt is a df with row keys
        for i in range(n_classes):
            misc.append(0)
            n_.append(0)
        for i in range(n_obs):
            for j in range(n_classes):
                if y_[i] == lr.classes_[j]:
                    ase_sum += (1-prob_[i,j])*(1-prob_[i,j])
                    idx = j
                else:
                    ase_sum += prob_[i,j]*prob_[i,j]
            for j in range(n_classes):
                if predict_[i] == lr.classes_[j]:
                        conf_mat[idx][j] += 1
                        break
            n_[idx] += 1
            if predict_[i] != y_[i]:
                misc_     += 1
                misc[idx] += 1
        misc_ = 100*misc_/n_obs
        ase   = ase_sum/(n_classes*n_obs)
        
        print("\nModel Metrics")
        print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
        n_coef = len(lr.coef_)*(len(lr.coef_[0])+1)
        print("{:.<27s}{:10d}".format('Coefficients', n_coef))
        print("{:.<27s}{:10d}".format('DF Error', X.shape[0]-n_coef))
        print("{:.<27s}{:10.2f}".format('ASE', ase))
        print("{:.<27s}{:10.2f}".format('Root ASE', sqrt(ase)))
        print("{:.<27s}{:10.1f}{:s}".format(\
                'MISC (Misclassification)', misc_, '%'))
        for i in range(n_classes):
            misc[i] = 100*misc[i]/n_[i]
            print("{:s}{:.<16.0f}{:>10.1f}{:<1s}".format(\
                  '     class ', lr.classes_[i], misc[i], '%'))
        print("\n\n     Confusion")
        print("       Matrix    ", end="")
        for i in range(n_classes):
            print("{:>7s}{:<3.0f}".format('Class ', lr.classes_[i]), end="")
        print("")
        for i in range(n_classes):
            print("{:s}{:.<6.0f}".format('Class ', lr.classes_[i]), end="")
            for j in range(n_classes):
                print("{:>10d}".format(conf_mat[i][j]), end="")
            print("")

        cr = classification_report(y, predict_, lr.classes_)
        print("\n",cr)
        
        
    def display_binary_split_metrics(lr, Xt, yt, Xv, yv):
        if len(lr.classes_) != 2:
            raise RuntimeError("  Call to display_split_metrics invalid.\n"+\
               "  This target does not appear to be binary."+\
               "  Number of target classes is not 2.")
            sys.exit()
        zt = np.zeros(len(yt))
        zv = np.zeros(len(yv))
        #zt = deepcopy(yt)
        for i in range(len(yt)):
            if yt[i] == 1:
                zt[i] = 1
        for i in range(len(yv)):
            if yv[i] == 1:
                zv[i] = 1
        predict_t = lr.predict(Xt)
        predict_v = lr.predict(Xv)
        conf_matt = confusion_matrix(y_true=yt, y_pred=predict_t)
        conf_matv = confusion_matrix(y_true=yv, y_pred=predict_v)
        prob_t = lr._predict_proba_lr(Xt)
        prob_v = lr._predict_proba_lr(Xv)
        #prob_t = lr.predict_proba(Xt)
        #prob_v = lr.predict_proba(Xv)
        print("\n")
        print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', \
                                      'Training', 'Validation'))
        print("{:.<23s}{:15d}{:15d}".format('Observations', \
                                          Xt.shape[0], Xv.shape[0]))
        n_coef = len(lr.coef_)*(len(lr.coef_[0])+1)
        print("{:.<23s}{:15d}{:15d}".format('Coefficients', \
                                          n_coef, n_coef))
        print("{:.<23s}{:15d}{:15d}".format('DF Error', \
                      Xt.shape[0]-n_coef, Xv.shape[0]-n_coef))
        if type(yt[0])==str:
            raise RuntimeError("  Call to display_binary_split_metrics "+\
               " invalid.\n  Target encoded as strings rather than"+\
               " integers.\n  Cannot properly calculate metrics.")
            sys.exit()
        print("{:.<23s}{:15.4f}{:15.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(zt,prob_t[:,1]), \
                      mean_absolute_error(zv,prob_v[:,1])))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Avg Squared Error', \
                      mean_squared_error(zt,prob_t[:,1]), \
                      mean_squared_error(zv,prob_v[:,1])))
        
        acct = accuracy_score(yt, predict_t)
        accv = accuracy_score(yv, predict_v)
        print("{:.<23s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
        pret = precision_score(yt,predict_t)
        prev = precision_score(yv,predict_v)
        print("{:.<23s}{:15.4f}{:15.4f}".format('Precision', pret, prev))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Recall (Sensitivity)', \
                      recall_score(yt,predict_t), \
                      recall_score(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('F1-score', \
                      f1_score(yt,predict_t), \
                      f1_score(yv,predict_v)))
        misct = conf_matt[0][1]+conf_matt[1][0]
        miscv = conf_matv[0][1]+conf_matv[1][0]
        misct = 100*misct/len(yt)
        miscv = 100*miscv/len(yv)
        n_t   = [conf_matt[0][0]+conf_matt[0][1], \
                 conf_matt[1][0]+conf_matt[1][1]]
        n_v   = [conf_matv[0][0]+conf_matv[0][1], \
                 conf_matv[1][0]+conf_matv[1][1]]
        misc_ = [[0,0], [0,0]]
        misc_[0][0] = 100*conf_matt[0][1]/n_t[0]
        misc_[0][1] = 100*conf_matt[1][0]/n_t[1]
        misc_[1][0] = 100*conf_matv[0][1]/n_v[0]
        misc_[1][1] = 100*conf_matv[1][0]/n_v[1]
        print("{:.<27s}{:10.1f}{:s}{:14.1f}{:s}".format(\
                'MISC (Misclassification)', misct, '%', miscv, '%'))
        for i in range(2):
            print("{:s}{:.<16.0f}{:>10.1f}{:<1s}{:>14.1f}{:<1s}".format(\
                  '     class ', lr.classes_[i], \
                  misc_[0][i], '%', misc_[1][i], '%'))
        print("\n\nTraining")
        print("Confusion Matrix ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', lr.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', lr.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_matt[i][j]), end="")
            print("")
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, lr.classes_)
        #print("\n",cr)
        
        print("\n\nValidation")
        print("Confusion Matrix ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', lr.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', lr.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_matv[i][j]), end="")
            print("")
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, lr.classes_)
        #print("\n",cr)
   
    def display_nominal_split_metrics(lr, Xt, yt, Xv, yv, target_names=None):
        n_classes = len(lr.classes_)
        if n_classes < 2:
            raise RuntimeError("  Call to display_nominal_split_metrics"+\
                               " invalid.\n"+\
               "  This target does not appear to be nominal."+\
               "  The number of target classes is less than 2.")
            sys.exit()
        predict_t = lr.predict(Xt)
        predict_v = lr.predict(Xv)
        prob_t = lr._predict_proba_lr(Xt)
        prob_v = lr._predict_proba_lr(Xv)
        #prob_t = lr.predict_proba(Xt)
        #prob_v = lr.predict_proba(Xv)
        ase_sumt = 0
        ase_sumv = 0
        misc_t = 0
        misc_v = 0
        misct  = []
        miscv  = []
        n_t    = []
        n_v    = []
        nt_obs = yt.shape[0]
        nv_obs = yv.shape[0]
        conf_matt = []
        conf_matv = []
        for i in range(nt_obs):
            z = []
            for j in range(n_classes):
                z.append(0)
            conf_matt.append(z)
            conf_matv.append(z)
        y_t = np.array(yt) # necessary because yt is a df with row keys
        y_v = np.array(yv) # likewise
        for i in range(n_classes):
            misct.append(0)
            n_t.append(0)
            miscv.append(0)
            n_v.append(0)
        for i in range(nt_obs):
            for j in range(n_classes):
                if y_t[i] == lr.classes_[j]:
                    ase_sumt += (1-prob_t[i,j])*(1-prob_t[i,j])
                    idx = j
                else:
                    ase_sumt += prob_t[i,j]*prob_t[i,j]
            for j in range(n_classes):
                if predict_t[i] == lr.classes_[j]:
                    conf_matt[idx][j] += 1
                    break
            n_t[idx] += 1
            if predict_t[i] != y_t[i]:
                misc_t     += 1
                misct[idx] += 1
                
        for i in range(nv_obs):
            for j in range(n_classes):
                if y_v[i] == lr.classes_[j]:
                    ase_sumv += (1-prob_v[i,j])*(1-prob_v[i,j])
                    idx = j
                else:
                    ase_sumv += prob_v[i,j]*prob_v[i,j]
            for j in range(n_classes):
                if predict_v[i] == lr.classes_[j]:
                    conf_matv[idx][j] += 1
                    break
            n_v[idx] += 1
            if predict_v[i] != y_v[i]:
                misc_v     += 1
                miscv[idx] += 1
        misc_t = 100*misc_t/nt_obs
        misc_v = 100*misc_v/nv_obs
        aset   = ase_sumt/(n_classes*nt_obs)
        asev   = ase_sumv/(n_classes*nv_obs)

        print("")
        print("{:.<27s}{:>11s}{:>13s}".format('Model Metrics', \
                                      'Training', 'Validation'))
        print("{:.<27s}{:10d}{:11d}".format('Observations', \
                                          Xt.shape[0], Xv.shape[0]))
        n_coef = len(lr.coef_)*(len(lr.coef_[0])+1)
        print("{:.<27s}{:10d}{:11d}".format('Coefficients', \
                                          n_coef, n_coef))
        print("{:.<27s}{:10d}{:11d}".format('DF Error', \
                      Xt.shape[0]-n_coef, Xv.shape[0]-n_coef))
        
        print("{:.<27s}{:10.2f}{:11.2f}".format(
                                'ASE', aset, asev))
        print("{:.<27s}{:10.2f}{:11.2f}".format(\
                                'Root ASE', sqrt(aset), sqrt(asev)))
        
        print("{:.<27s}{:10.1f}{:s}{:10.1f}{:s}".format(\
                'MISC (Misclassification)', misc_t, '%', misc_v, '%'))
        for i in range(n_classes):
            misct[i] = 100*misct[i]/n_t[i]
            miscv[i] = 100*miscv[i]/n_v[i]
            print("{:s}{:.<16d}{:>10.1f}{:<1s}{:>10.1f}{:<1s}".format(\
                  '     class ', lr.classes_[i], misct[i], '%', miscv[i], '%'))

        print("\n\nTraining")
        print("Confusion Matrix ", end="")
        for i in range(n_classes):
            print("{:>7s}{:<3.0f}".format('Class ', lr.classes_[i]), end="")
        print("")
        for i in range(n_classes):
            print("{:s}{:.<6.0f}".format('Class ', lr.classes_[i]), end="")
            for j in range(n_classes):
                print("{:>10d}".format(conf_matt[i][j]), end="")
            print("")
            
        ct = classification_report(yt, predict_t, target_names)
        print("\nTraining \nMetrics:\n",ct)
        
        print("\n\nValidation")
        print("Confusion Matrix ", end="")
        for i in range(n_classes):
            print("{:>7s}{:<3.0f}".format('Class ', lr.classes_[i]), end="")
        print("")
        for i in range(n_classes):
            print("{:s}{:.<6.0f}".format('Class ', lr.classes_[i]), end="")
            for j in range(n_classes):
                print("{:>10d}".format(conf_matv[i][j]), end="")
            print("")
        cv = classification_report(yv, predict_v, target_names)
        print("\nValidation \nMetrics:\n",cv)

class DecisionTree(object):
    
    def display_metrics(dt, X, y):
        predictions = dt.predict(X)
        depth = dt.max_depth
        print("\nModel Metrics")
        print("{:.<23s}{:9d}".format('Observations', X.shape[0]))
        print("{:.<23s}{:>9s}".format('Split Criterion', dt.criterion))
        if depth == None:
            print("{:.<23s}{:>9s}".format('Max Depth', 'None'))
        else:
            print("{:.<23s}{:9d}".format('Max Depth', depth))
        print("{:.<23s}{:9d}".format('Minimum Split Size', \
                                      dt.min_samples_split))
        print("{:.<23s}{:9d}".format('Minimum Leaf  Size', \
                                      dt.min_samples_leaf))
        R2 = r2_score(y, predictions)
        print("{:.<23s}{:9.4f}".format('R-Squared', R2))
        print("{:.<23s}{:9.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(y,predictions)))
        print("{:.<23s}{:9.4f}".format('Median Absolute Error', \
                      median_absolute_error(y,predictions)))
        print("{:.<23s}{:9.4f}".format('Avg Squared Error', \
                      mean_squared_error(y,predictions)))
        print("{:.<23s}{:9.4f}".format('Square Root ASE', \
                      sqrt(mean_squared_error(y,predictions))))
        
    def display_split_metrics(dt, Xt, yt, Xv, yv):
        predict_t = dt.predict(Xt)
        predict_v = dt.predict(Xv)
        depth = dt.max_depth
        print("{:.<23s}{:>10s}{:>15s}".format('\nModel Metrics', \
                                      'Training', 'Validation'))
        print("{:.<23s}{:9d}{:15d}".format('Observations', \
                                          Xt.shape[0], Xv.shape[0]))
        print("{:.<23s}{:>9s}{:>15s}".format('Split Criterion', \
                                      dt.criterion, dt.criterion))
        if depth==None:
            print("{:.<23s}{:>9s}{:>15s}".format('Max Depth', \
                                          'None', 'None'))
        else:
            print("{:.<23s}{:9d}{:15d}".format('Max Depth',   \
                                          depth, depth))
        print("{:.<23s}{:9d}{:15d}".format('Minimum Split Size',   \
                         dt.min_samples_split, dt.min_samples_split))
        print("{:.<23s}{:9d}{:15d}".format('Minimum Leaf  Size',   \
                         dt.min_samples_leaf, dt.min_samples_leaf))

        R2t = r2_score(yt, predict_t)
        R2v = r2_score(yv, predict_v)
        print("{:.<23s}{:9.4f}{:15.4f}".format('R-Squared', R2t, R2v))
        print("{:.<23s}{:9.4f}{:15.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(yt,predict_t), \
                      mean_absolute_error(yv,predict_v)))
        print("{:.<23s}{:9.4f}{:15.4f}".format('Median Absolute Error', \
                      median_absolute_error(yt,predict_t), \
                      median_absolute_error(yv,predict_v)))
        print("{:.<23s}{:9.4f}{:15.4f}".format('Avg Squared Error', \
                      mean_squared_error(yt,predict_t), \
                      mean_squared_error(yv,predict_v)))
        print("{:.<23s}{:9.4f}{:15.4f}".format('Square Root ASE', \
                      sqrt(mean_squared_error(yt,predict_t)), \
                      sqrt(mean_squared_error(yv,predict_v))))
        
    def display_importance(dt, col, top='all', plot=False):
        nx = dt.n_features_
        if nx != len(col):
            raise RuntimeError("  Call to display_importance invalid\n"+\
                  "  Number of feature labels (col) not equal to the " +\
                  "number of features in the decision tree.")
            sys.exit()
        if type(top) != int and type(top) != str:
            raise RuntimeError("   Call to display_importance invalid\n"+\
                  "   Value of top is invalid.  Must be set to 'all' or"+\
                  " an integer less than the number of columns in X.")
            sys.exit()
        if type(top) == str and top != 'all':
            raise RuntimeError("   Call to display_importance invalid\n"+\
                  "   Value of top is invalid.  Must be set to 'all' or"+\
                  " an integer less than the number of columns in X.")
            sys.exit()
        max_label = 6
        for i in range(len(col)):
            if len(col[i]) > max_label:
                max_label = len(col[i])+4
        label_format = ("{:.<%i" %max_label)+"s}{:9.4f}"
        
        features = []
        this_col = []
        for i in range(nx):
            features.append(dt.feature_importances_[i])
            this_col.append(col[i])
        sorted = False
        while (sorted==False):
            sorted = True
            for i in range(nx-1):
                if features[i]<features[i+1]:
                    sorted=False
                    x = features[i]
                    c = this_col[i]
                    features[i] = features[i+1]
                    this_col[i] = this_col[i+1]
                    features[i+1] = x
                    this_col[i+1] = c
        print("")
        label_format2 = ("{:.<%i" %max_label)+"s}{:s}"
        print(label_format2.format("FEATURE", " IMPORTANCE"))
        n_x = nx
        if type(top) == int:
            if top <= n_x and top > 0:
                n_x = top
        for i in range(n_x):
            print(label_format.format(this_col[i], features[i]))
        print("")
        
        if plot==False:
            return
        f = pd.DataFrame()
        f['feature'] = this_col[0:n_x]
        f['importance'] = features[0:n_x]
        f.sort_values(by=['importance'], ascending=True, inplace=True)
        f.set_index('feature', inplace=True)
        # Plot using Pandas plot which uses pyplot
        print("\nFeature Importances:")
        plt.figure() # clears any exiting plot
        plt_ = f.plot(kind='barh', figsize=(8, 10), fontsize=14)
        plt_.set_ylabel("Features", fontname="Arial", fontsize=14)
        plt.figure() # Forces immediate display and clears plot
        plt.show()
        
    def display_binary_metrics(dt, X, y):
        try:
            if len(dt.classes_) != 2:
                raise RuntimeError("  Call to display_binary_metrics invalid"+\
                  "  Target does not have two classes.\n  If target is "+\
                  "nominal, use display_nominal_metrics instead.")
                sys.exit()
        except:
            raise RuntimeError("  Call to display_binary_metrics invalid"+\
                  "  Target does not have two classes.\n  If target is "+\
                  "nominal, use display_nominal_metrics instead.")
            sys.exit()
        numpy_y = np.array(y)
        z = np.zeros(len(y))
        predictions = dt.predict(X) # get binary class predictions
        conf_mat = confusion_matrix(y_true=y, y_pred=predictions)
        misc = 100*(conf_mat[0][1]+conf_mat[1][0])/(len(y))
        for i in range(len(y)):
            if numpy_y[i] == 1:
                z[i] = 1
        probability = dt.predict_proba(X) # get binary probabilities
        #probability = dt.predict_proba(X)
        print("\nModel Metrics")
        print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
        print("{:.<27s}{:10d}".format('Features', X.shape[1]))
        if dt.max_depth==None:
            print("{:.<27s}{:>10s}".format('Maximum Tree Depth',\
                              "None"))
        else:
            print("{:.<27s}{:10d}".format('Maximum Tree Depth',\
                              dt.max_depth))
        print("{:.<27s}{:10d}".format('Minimum Leaf Size', \
                              dt.min_samples_leaf))
        print("{:.<27s}{:10d}".format('Minimum split Size', \
                              dt.min_samples_split))
        if type(numpy_y[0])==str:
            print("\n*** ERROR: Binary Target is not Encoded as 0 and 1")
            print("*** ERROR: Cannot Properly Calculate Metrics")
            sys.exit()
        print("{:.<27s}{:10.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(z,probability[:, 1])))
        print("{:.<27s}{:10.4f}".format('Avg Squared Error', \
                      mean_squared_error(z,probability[:, 1])))
        acc = accuracy_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Accuracy', acc))
        pre = precision_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Precision', pre))
        tpr = recall_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
        f1 =  f1_score(y,predictions)
        print("{:.<27s}{:10.4f}".format('F1-Score', f1))
        print("{:.<27s}{:9.1f}{:s}".format(\
                'MISC (Misclassification)', misc, '%'))
        n_    = [conf_mat[0][0]+conf_mat[0][1], conf_mat[1][0]+conf_mat[1][1]]
        miscc = [100*conf_mat[0][1]/n_[0], 100*conf_mat[1][0]/n_[1]]
        for i in range(2):
            print("{:s}{:.<16.0f}{:>9.1f}{:<1s}".format(\
                  '     class ', dt.classes_[i], miscc[i], '%'))      

        print("\n\n     Confusion")
        print("       Matrix    ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', dt.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', dt.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_mat[i][j]), end="")
            print("")
        print("")
        
    def display_binary_split_metrics(dt, Xt, yt, Xv, yv):
        try:
            if len(dt.classes_) != 2:
                raise RuntimeError("  Call to display_binary_split_metrics "+\
                  "invalid.\n  Target does not have two classes.\n  If "+\
                  "target is nominal, use display_nominal_metrics instead.")
                sys.exit()
        except:
            raise RuntimeError("  Call to display_binary_split_metrics "+\
                  "invalid.\n  Target does not have two classes.\n  If "+\
                  "target is nominal, use display_nominal_metrics instead.")
            sys.exit()
        numpy_yt = np.array(yt)
        numpy_yv = np.array(yv)
        zt = np.zeros(len(yt))
        zv = np.zeros(len(yv))
        #zt = deepcopy(yt)
        for i in range(len(yt)):
            if numpy_yt[i] == 1:
                zt[i] = 1
        for i in range(len(yv)):
            if numpy_yv[i] == 1:
                zv[i] = 1
        predict_t = dt.predict(Xt)
        predict_v = dt.predict(Xv)
        conf_matt = confusion_matrix(y_true=yt, y_pred=predict_t)
        conf_matv = confusion_matrix(y_true=yv, y_pred=predict_v)
        prob_t = dt.predict_proba(Xt)
        prob_v = dt.predict_proba(Xv)
        print("\n")
        print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', \
                                      'Training', 'Validation'))
        print("{:.<23s}{:15d}{:15d}".format('Observations', \
                                          Xt.shape[0], Xv.shape[0]))
        
        print("{:.<23s}{:15d}{:15d}".format('Features', Xt.shape[1], \
                                                          Xv.shape[1]))
        if dt.max_depth==None:
            print("{:.<23s}{:>15s}{:>15s}".format('Maximum Tree Depth',\
                              "None", "None"))
        else:
            print("{:.<23s}{:15d}{:15d}".format('Maximum Tree Depth',\
                              dt.max_depth, dt.max_depth))
        print("{:.<23s}{:15d}{:15d}".format('Minimum Leaf Size', \
                              dt.min_samples_leaf, dt.min_samples_leaf))
        print("{:.<23s}{:15d}{:15d}".format('Minimum split Size', \
                              dt.min_samples_split, dt.min_samples_split))
        if type(numpy_yt[0])==str:
            print("\n*** ERROR: Binary Target is not Encoded as 0 and 1")
            print("*** ERROR: Cannot Properly Calculate Metrics")
            sys.exit()
        print("{:.<23s}{:15.4f}{:15.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(zt,prob_t[:,1]), \
                      mean_absolute_error(zv,prob_v[:,1])))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Avg Squared Error', \
                      mean_squared_error(zt,prob_t[:,1]), \
                      mean_squared_error(zv,prob_v[:,1])))
        
        acct = accuracy_score(yt, predict_t)
        accv = accuracy_score(yv, predict_v)
        print("{:.<23s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
        
        print("{:.<23s}{:15.4f}{:15.4f}".format('Precision', \
                      precision_score(yt,predict_t), \
                      precision_score(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Recall (Sensitivity)', \
                      recall_score(yt,predict_t), \
                      recall_score(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('F1-score', \
                      f1_score(yt,predict_t), \
                      f1_score(yv,predict_v)))
        misct = conf_matt[0][1]+conf_matt[1][0]
        miscv = conf_matv[0][1]+conf_matv[1][0]
        misct = 100*misct/len(yt)
        miscv = 100*miscv/len(yv)
        n_t   = [conf_matt[0][0]+conf_matt[0][1], \
                 conf_matt[1][0]+conf_matt[1][1]]
        n_v   = [conf_matv[0][0]+conf_matv[0][1], \
                 conf_matv[1][0]+conf_matv[1][1]]
        misc_ = [[0,0], [0,0]]
        misc_[0][0] = 100*conf_matt[0][1]/n_t[0]
        misc_[0][1] = 100*conf_matt[1][0]/n_t[1]
        misc_[1][0] = 100*conf_matv[0][1]/n_v[0]
        misc_[1][1] = 100*conf_matv[1][0]/n_v[1]
        print("{:.<27s}{:10.1f}{:s}{:14.1f}{:s}".format(\
                'MISC (Misclassification)', misct, '%', miscv, '%'))
        for i in range(2):
            print("{:s}{:.<16.0f}{:>10.1f}{:<1s}{:>14.1f}{:<1s}".format(\
                  '     class ', dt.classes_[i], \
                  misc_[0][i], '%', misc_[1][i], '%'))
        print("\n\nTraining")
        print("Confusion Matrix ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', dt.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', dt.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_matt[i][j]), end="")
            print("")
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, dt.classes_)
        #print("\n",cr)
        
        print("\n\nValidation")
        print("Confusion Matrix ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', dt.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', dt.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_matv[i][j]), end="")
            print("")
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, dt.classes_)
        #print("\n",cr)
   
    def display_nominal_metrics(dt, X, y):
        try:
            if len(dt.classes_) < 2:
                raise RuntimeError("  Call to display_nominal_metrics "+\
                  "invalid.\n  Target has less than two classes.\n")
                sys.exit()
        except:
            raise RuntimeError("  Call to display_nominal_metrics "+\
                  "invalid.\n  Target has less than two classes.\n")
            sys.exit()
        np_y = np.array(y)
        predictions = dt.predict(X) # get binary class predictions
        conf_mat = confusion_matrix(y_true=y, y_pred=predictions)
        misc = 100*(conf_mat[0][1]+conf_mat[1][0])/(len(y))
        probability = dt.predict_proba(X) # get class probabilities
        print("\nModel Metrics")
        print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
        print("{:.<27s}{:10d}".format('Features', X.shape[1]))
        if dt.max_depth==None:
            print("{:.<27s}{:>10s}".format('Maximum Tree Depth',\
                              "None"))
        else:
            print("{:.<27s}{:10d}".format('Maximum Tree Depth',\
                              dt.max_depth))
        print("{:.<27s}{:10d}".format('Minimum Leaf Size', \
                              dt.min_samples_leaf))
        print("{:.<27s}{:10d}".format('Minimum split Size', \
                              dt.min_samples_split))
        print("{:.<27s}{:10.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(np_y,probability[:, 1])))
        print("{:.<27s}{:10.4f}".format('Avg Squared Error', \
                      mean_squared_error(np_y,probability[:, 1])))
        acc = accuracy_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Accuracy', acc))
        pre = precision_score(y, predictions, average='micro')
        print("{:.<27s}{:10.4f}".format('Precision', pre))
        tpr = recall_score(y, predictions, average='micro')
        print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
        f1 =  f1_score(y,predictions, average='micro')
        print("{:.<27s}{:10.4f}".format('F1-Score', f1))
        print("{:.<27s}{:9.1f}{:s}".format(\
                'MISC (Misclassification)', misc, '%'))
        n_    = [conf_mat[0][0]+conf_mat[0][1], conf_mat[1][0]+conf_mat[1][1]]
        miscc = [100*conf_mat[0][1]/n_[0], 100*conf_mat[1][0]/n_[1]]
        print("conf_mat:", conf_mat)
        print("Miscc:", miscc)
        for i in range(len(dt.classes_)):
            print("{:s}{:.<16.0f}{:>9.1f}{:<1s}".format(\
                  '     class ', dt.classes_[i], miscc[i], '%'))      

        print("\n\n     Confusion")
        print("       Matrix    ", end="")
        for i in range(dt.classes_):
            print("{:>7s}{:<3.0f}".format('Class ', dt.classes_[i]), end="")
        print("")
        for i in range(dt.classes_):
            print("{:s}{:.<6.0f}".format('Class ', dt.classes_[i]), end="")
            for j in range(dt.classes_):
                print("{:>10d}".format(conf_mat[i][j]), end="")
            print("")
        print("")
        
    def display_nominal_split_metrics(dt, Xt, yt, Xv, yv):
        try:
            if len(dt.classes_) < 2:
                raise RuntimeError("  Call to display_nominal_split_metrics "+\
                  "invalid.\n  Target has less than two classes.\n")
                sys.exit()
        except:
            raise RuntimeError("  Call to display_nominal_split_metrics "+\
                  "invalid.\n  Target has less than two classes.\n")
            sys.exit()
        np_yt = np.array(yt)
        np_yv = np.array(yv)
        predict_t = dt.predict(Xt)
        predict_v = dt.predict(Xv)
        conf_matt = confusion_matrix(y_true=yt, y_pred=predict_t)
        conf_matv = confusion_matrix(y_true=yv, y_pred=predict_v)
        prob_t = dt.predict_proba(Xt)
        prob_v = dt.predict_proba(Xv)
        print("\n")
        print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', \
                                      'Training', 'Validation'))
        print("{:.<23s}{:15d}{:15d}".format('Observations', \
                                          Xt.shape[0], Xv.shape[0]))
        
        print("{:.<23s}{:15d}{:15d}".format('Features', Xt.shape[1], \
                                                          Xv.shape[1]))
        if dt.max_depth==None:
            print("{:.<23s}{:>15s}{:>15s}".format('Maximum Tree Depth',\
                              "None", "None"))
        else:
            print("{:.<23s}{:15d}{:15d}".format('Maximum Tree Depth',\
                              dt.max_depth, dt.max_depth))
        print("{:.<23s}{:15d}{:15d}".format('Minimum Leaf Size', \
                              dt.min_samples_leaf, dt.min_samples_leaf))
        print("{:.<23s}{:15d}{:15d}".format('Minimum split Size', \
                              dt.min_samples_split, dt.min_samples_split))
        
        print("{:.<23s}{:15.4f}{:15.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(np_yt,prob_t[:,1]), \
                      mean_absolute_error(np_yv,prob_v[:,1])))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Avg Squared Error', \
                      mean_squared_error(np_yt,prob_t[:,1]), \
                      mean_squared_error(np_yv,prob_v[:,1])))
        
        acct = accuracy_score(yt, predict_t)
        accv = accuracy_score(yv, predict_v)
        print("{:.<23s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
        
        print("{:.<23s}{:15.4f}{:15.4f}".format('Precision', \
                      precision_score(yt,predict_t), \
                      precision_score(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Recall (Sensitivity)', \
                      recall_score(yt,predict_t), \
                      recall_score(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('F1-score', \
                      f1_score(yt,predict_t), \
                      f1_score(yv,predict_v)))
        misct = conf_matt[0][1]+conf_matt[1][0]
        miscv = conf_matv[0][1]+conf_matv[1][0]
        misct = 100*misct/len(yt)
        miscv = 100*miscv/len(yv)
        n_t   = [conf_matt[0][0]+conf_matt[0][1], \
                 conf_matt[1][0]+conf_matt[1][1]]
        n_v   = [conf_matv[0][0]+conf_matv[0][1], \
                 conf_matv[1][0]+conf_matv[1][1]]
        misc_ = [[0,0], [0,0]]
        misc_[0][0] = 100*conf_matt[0][1]/n_t[0]
        misc_[0][1] = 100*conf_matt[1][0]/n_t[1]
        misc_[1][0] = 100*conf_matv[0][1]/n_v[0]
        misc_[1][1] = 100*conf_matv[1][0]/n_v[1]
        print("{:.<27s}{:10.1f}{:s}{:14.1f}{:s}".format(\
                'MISC (Misclassification)', misct, '%', miscv, '%'))
        for i in range(2):
            print("{:s}{:.<16.0f}{:>10.1f}{:<1s}{:>14.1f}{:<1s}".format(\
                  '     class ', dt.classes_[i], \
                  misc_[0][i], '%', misc_[1][i], '%'))
        print("\n\nTraining")
        print("Confusion Matrix ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', dt.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', dt.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_matt[i][j]), end="")
            print("")
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, dt.classes_)
        #print("\n",cr)
        
        print("\n\nValidation")
        print("Confusion Matrix ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', dt.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', dt.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_matv[i][j]), end="")
            print("")
        # In the binary case, the classification report is incorrect
        # cr = classification_report(yv, predict_v, dt.classes_)
        # print("\n",cr)
        # cv = classification_report(yv, predict_v, target_names)
        # print("\nValidation \nMetrics:\n",cv)
        

class NeuralNetwork(object):
        
    def display_metrics(nn, X, y):
        predictions = nn.predict(X)
        #Calculate number of weights
        n_weights = 0
        for i in range(nn.n_layers_ - 1):
            n_weights += len(nn.intercepts_[i])
            n_weights += nn.coefs_[i].shape[0]*nn.coefs_[i].shape[1]
        
        print("\nModel Metrics")
        print("{:.<23s}{:15d}".format('Observations', X.shape[0]))
        print("{:.<23s}{:15d}".format('Features', X.shape[1]))
        print("{:.<23s}{:15d}".format('Number of Layers',\
                              nn.n_layers_-2))
        print("{:.<23s}{:15d}".format('Number of Outputs', \
                              nn.n_outputs_))
        n_neurons = 0
        nl = nn.n_layers_-2
        if nl>1:
            for i in range(nl):
                n_neurons += nn.hidden_layer_sizes[i]
        else:
            n_neurons = nn.hidden_layer_sizes
        print("{:.<23s}{:15d}".format('Number of Neurons',\
                              n_neurons))
        print("{:.<23s}{:15d}".format('Number of Weights', \
                             n_weights))
        print("{:.<23s}{:15d}".format('Number of Iterations', \
                             nn.n_iter_))
        print("{:.<23s}{:>15s}".format('Activation Function', \
                             nn.activation))
        print("{:.<23s}{:15.4f}".format('Loss', nn.loss_))
        print("{:.<23s}{:15.4f}".format('R-Squared', \
                      r2_score(y,predictions)))
        print("{:.<23s}{:15.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(y,predictions)))
        print("{:.<23s}{:15.4f}".format('Median Absolute Error', \
                      median_absolute_error(y,predictions)))
        print("{:.<23s}{:15.4f}".format('Avg Squared Error', \
                      mean_squared_error(y,predictions)))
        print("{:.<23s}{:15.4f}".format('Square Root ASE', \
                      sqrt(mean_squared_error(y,predictions))))
        
    def display_split_metrics(nn, Xt, yt, Xv, yv):
        predict_t = nn.predict(Xt)
        predict_v = nn.predict(Xv)
        #Calculate number of weights
        n_weights = 0
        for i in range(nn.n_layers_ - 1):
            n_weights += len(nn.intercepts_[i])
            n_weights += nn.coefs_[i].shape[0]*nn.coefs_[i].shape[1]
        print("\n")
        print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', \
                                      'Training', 'Validation'))
        print("{:.<23s}{:15d}{:15d}".format('Observations', \
                                          Xt.shape[0], Xv.shape[0]))
        print("{:.<23s}{:15d}{:15d}".format('Features',     \
                                          Xt.shape[1], Xv.shape[1]))
        print("{:.<23s}{:15d}{:15d}".format('Number of Layers',\
                              (nn.n_layers_-2),(nn.n_layers_-2)))
        n_neurons = 0
        nl = nn.n_layers_-2
        if nl>1:
            for i in range(nl):
                n_neurons += nn.hidden_layer_sizes[i]
        else:
            n_neurons = nn.hidden_layer_sizes
        print("{:.<23s}{:15d}{:15d}".format('Number of Neurons',\
                              n_neurons, n_neurons))
        print("{:.<23s}{:15d}{:15d}".format('Number of Outputs', \
                              nn.n_outputs_, nn.n_outputs_))
        print("{:.<23s}{:15d}{:15d}".format('Number of Weights', \
                              n_weights, n_weights))
        print("{:.<23s}{:15d}{:15d}".format('Number of Iterations', \
                              nn.n_iter_, nn.n_iter_))
        print("{:.<23s}{:>15s}{:>15s}".format('Activation Function', \
                              nn.activation, nn.activation))
        print("{:.<23s}{:15.4f}".format('Loss', nn.loss_))
        R2t = r2_score(yt, predict_t)
        R2v = r2_score(yv, predict_v)
        print("{:.<23s}{:15.4f}{:15.4f}".format('R-Squared', R2t, R2v))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(yt,predict_t), \
                      mean_absolute_error(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Median Absolute Error', \
                      median_absolute_error(yt,predict_t), \
                      median_absolute_error(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Avg Squared Error', \
                      mean_squared_error(yt,predict_t), \
                      mean_squared_error(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Square Root ASE', \
                      sqrt(mean_squared_error(yt,predict_t)), \
                      sqrt(mean_squared_error(yv,predict_v))))
        
    def display_binary_metrics(nn, X, y):
        try:
            if len(nn.classes_) != 2:
                raise RuntimeError("  Call to display_binary_metrics invalid"+\
                  "  Target does not have two classes.\n  If target is "+\
                  "nominal, use display_nominal_metrics instead.")
                sys.exit()
        except:
            raise RuntimeError("  Call to display_binary_metrics invalid"+\
                  "  Target does not have two classes.\n  If target is "+\
                  "nominal, use display_nominal_metrics instead.")
            sys.exit()
        numpy_y = np.array(y)
        z = np.zeros(len(y))
        predictions = nn.predict(X) # get binary class predictions
        conf_mat = confusion_matrix(y_true=y, y_pred=predictions)
        misc = 100*(conf_mat[0][1]+conf_mat[1][0])/(len(y))
        for i in range(len(y)):
            if numpy_y[i] == 1:
                z[i] = 1
        probability = nn.predict_proba(X) # get binary probabilities
        #Calculate number of weights
        n_weights = 0
        for i in range(nn.n_layers_ - 1):
            n_weights += len(nn.intercepts_[i])
            n_weights += nn.coefs_[i].shape[0]*nn.coefs_[i].shape[1]
        #probability = nn.predict_proba(X)
        print("\nModel Metrics")
        print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
        print("{:.<27s}{:10d}".format('Features', X.shape[1]))
        print("{:.<27s}{:10d}".format('Number of Layers',\
                              nn.n_layers_-2))
        print("{:.<27s}{:10d}".format('Number of Outputs', \
                              nn.n_outputs_))
        n_neurons = 0
        nl = nn.n_layers_-2
        if nl>1:
            for i in range(nl):
                n_neurons += nn.hidden_layer_sizes[i]
        else:
            n_neurons = nn.hidden_layer_sizes
        print("{:.<27s}{:10d}".format('Number of Neurons',\
                              n_neurons))
        print("{:.<27s}{:10d}".format('Number of Weights', \
                             n_weights))
        print("{:.<27s}{:10d}".format('Number of Iterations', \
                             nn.n_iter_))
        print("{:.<27s}{:>10s}".format('Activation Function', \
                             nn.out_activation_))
        
        if type(numpy_y[0])==str:
            print("\n*** ERROR: Binary Target is not Encoded as 0 and 1")
            print("*** ERROR: Cannot Properly Calculate Metrics")
            sys.exit()
        print("{:.<27s}{:10.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(z,probability[:, 1])))
        print("{:.<27s}{:10.4f}".format('Avg Squared Error', \
                      mean_squared_error(z,probability[:, 1])))
        acc = accuracy_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Accuracy', acc))
        pre = precision_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Precision', pre))
        tpr = recall_score(y, predictions)
        print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
        f1 =  f1_score(y,predictions)
        print("{:.<27s}{:10.4f}".format('F1-Score', f1))
        print("{:.<27s}{:9.1f}{:s}".format(\
                'MISC (Misclassification)', misc, '%'))
        n_    = [conf_mat[0][0]+conf_mat[0][1], conf_mat[1][0]+conf_mat[1][1]]
        miscc = [100*conf_mat[0][1]/n_[0], 100*conf_mat[1][0]/n_[1]]
        for i in range(2):
            print("{:s}{:.<16.0f}{:>9.1f}{:<1s}".format(\
                  '     class ', nn.classes_[i], miscc[i], '%'))      

        print("\n\n     Confusion")
        print("       Matrix    ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', nn.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', nn.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_mat[i][j]), end="")
            print("")
        print("")
        
    def display_binary_split_metrics(nn, Xt, yt, Xv, yv):
        try:
            if len(nn.classes_) != 2:
                raise RuntimeError("  Call to display_binary_split_metrics "+\
                  "invalid.\n  Target does not have two classes.\n  If "+\
                  "target is nominal, use display_nominal_metrics instead.")
                sys.exit()
        except:
            raise RuntimeError("  Call to display_binary_split_metrics "+\
                  "invalid.\n  Target does not have two classes.\n  If "+\
                  "target is nominal, use display_nominal_metrics instead.")
            sys.exit()
        #Calculate number of weights
        n_weights = 0
        for i in range(nn.n_layers_ - 1):
            n_weights += len(nn.intercepts_[i])
            n_weights += nn.coefs_[i].shape[0]*nn.coefs_[i].shape[1]
        numpy_yt = np.array(yt)
        numpy_yv = np.array(yv)
        zt = np.zeros(len(yt))
        zv = np.zeros(len(yv))
        #zt = deepcopy(yt)
        for i in range(len(yt)):
            if numpy_yt[i] == 1:
                zt[i] = 1
        for i in range(len(yv)):
            if numpy_yv[i] == 1:
                zv[i] = 1
        predict_t = nn.predict(Xt)
        predict_v = nn.predict(Xv)
        conf_matt = confusion_matrix(y_true=yt, y_pred=predict_t)
        conf_matv = confusion_matrix(y_true=yv, y_pred=predict_v)
        prob_t = nn.predict_proba(Xt)
        prob_v = nn.predict_proba(Xv)
        print("\n")
        print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', \
                                      'Training', 'Validation'))
        print("{:.<23s}{:15d}{:15d}".format('Observations', \
                                          Xt.shape[0], Xv.shape[0]))
        
        print("{:.<23s}{:15d}{:15d}".format('Features', Xt.shape[1], \
                                                          Xv.shape[1]))
        print("{:.<23s}{:15d}{:15d}".format('Number of Layers',\
                              nn.n_layers_-2, nn.n_layers_-2))
        print("{:.<23s}{:15d}{:15d}".format('Number of Outputs', \
                              nn.n_outputs_, nn.n_outputs_))
        n_neurons = 0
        nl = nn.n_layers_-2
        if nl>1:
            for i in range(nl):
                n_neurons += nn.hidden_layer_sizes[i]
        else:
            n_neurons = nn.hidden_layer_sizes
        print("{:.<23s}{:15d}{:15d}".format('Number of Neurons',\
                              n_neurons, n_neurons))
        print("{:.<23s}{:15d}{:15d}".format('Number of Weights', \
                             n_weights, n_weights))
        print("{:.<23s}{:15d}{:15d}".format('Number of Iterations', \
                              nn.n_iter_, nn.n_iter_))
        print("{:.<23s}{:>15s}{:>15s}".format('Activation Function', \
                             nn.out_activation_, nn.out_activation_))
        
        if type(numpy_yt[0])==str:
            print("\n*** ERROR: Binary Target is not Encoded as 0 and 1")
            print("*** ERROR: Cannot Properly Calculate Metrics")
            sys.exit()
        print("{:.<23s}{:15.4f}{:15.4f}".format('Mean Absolute Error', \
                      mean_absolute_error(zt,prob_t[:,1]), \
                      mean_absolute_error(zv,prob_v[:,1])))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Avg Squared Error', \
                      mean_squared_error(zt,prob_t[:,1]), \
                      mean_squared_error(zv,prob_v[:,1])))
        
        acct = accuracy_score(yt, predict_t)
        accv = accuracy_score(yv, predict_v)
        print("{:.<23s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
        
        print("{:.<23s}{:15.4f}{:15.4f}".format('Precision', \
                      precision_score(yt,predict_t), \
                      precision_score(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('Recall (Sensitivity)', \
                      recall_score(yt,predict_t), \
                      recall_score(yv,predict_v)))
        print("{:.<23s}{:15.4f}{:15.4f}".format('F1-score', \
                      f1_score(yt,predict_t), \
                      f1_score(yv,predict_v)))
        misct = conf_matt[0][1]+conf_matt[1][0]
        miscv = conf_matv[0][1]+conf_matv[1][0]
        misct = 100*misct/len(yt)
        miscv = 100*miscv/len(yv)
        n_t   = [conf_matt[0][0]+conf_matt[0][1], \
                 conf_matt[1][0]+conf_matt[1][1]]
        n_v   = [conf_matv[0][0]+conf_matv[0][1], \
                 conf_matv[1][0]+conf_matv[1][1]]
        misc_ = [[0,0], [0,0]]
        misc_[0][0] = 100*conf_matt[0][1]/n_t[0]
        misc_[0][1] = 100*conf_matt[1][0]/n_t[1]
        misc_[1][0] = 100*conf_matv[0][1]/n_v[0]
        misc_[1][1] = 100*conf_matv[1][0]/n_v[1]
        print("{:.<27s}{:10.1f}{:s}{:14.1f}{:s}".format(\
                'MISC (Misclassification)', misct, '%', miscv, '%'))
        for i in range(2):
            print("{:s}{:.<16.0f}{:>10.1f}{:<1s}{:>14.1f}{:<1s}".format(\
                  '     class ', nn.classes_[i], \
                  misc_[0][i], '%', misc_[1][i], '%'))
        print("\n\nTraining")
        print("Confusion Matrix ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', nn.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', nn.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_matt[i][j]), end="")
            print("")
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, nn.classes_)
        #print("\n",cr)
        
        print("\n\nValidation")
        print("Confusion Matrix ", end="")
        for i in range(2):
            print("{:>7s}{:<3.0f}".format('Class ', nn.classes_[i]), end="")
        print("")
        for i in range(2):
            print("{:s}{:.<6.0f}".format('Class ', nn.classes_[i]), end="")
            for j in range(2):
                print("{:>10d}".format(conf_matv[i][j]), end="")
            print("")
        # In the binary case, the classification report is incorrect
        #cr = classification_report(yv, predict_v, nn.classes_)
        #print("\n",cr)
   
    def display_nominal_metrics(nn, X, y):
        n_classes = len(nn.classes_)
        n_obs = y.shape[0]
        if n_classes < 2:
            raise RuntimeError("\n  Call to display_nominal_metrics invalid"+\
                    "\n  Target does not appear to be nominal.\n"+\
                    "  Try using NeuralNetwork.display_binary_metrics()"+\
                    " instead.\n")
            sys.exit()
        # Nominal class predictions incorrectly require prob(class)>0.5
        # With many nominal classes, most cases have no predictions
        predict_ = nn.predict(X) # garbage
        if len(predict_.shape) !=2:
            raise RuntimeError("\n  Call to display_nominal_metrics invalid"+\
               "\n  Appears the target is not"+\
               " encoded into multiple binary columns\n" + \
               "  Try using ReplaceImputeEncode to encode target\n")
            sys.exit()
        prob_ = nn.predict_proba(X)
        for i in range(n_obs):
            max_prob   = 0
            prediction = 0
            for j in range(n_classes):
                if prob_[i,j]>max_prob:
                    max_prob = prob_[i,j]
                    prediction = j
            predict_[i,prediction] = 1
        ase_sum = 0
        misc_ = 0
        misc  = []
        n_    = []
        conf_mat = []
        for i in range(n_classes):
            z = []
            for j in range(n_classes):
                z.append(0)
            conf_mat.append(z)
        y_ = np.array(y) # necessary because yt is a df with row keys
        for i in range(n_classes):
            misc.append(0)
            n_.append(0)
        for i in range(n_obs):
            for j in range(n_classes):
                if y_[i,j] == 1:
                    ase_sum += (1-prob_[i,j])*(1-prob_[i,j])
                    idx = j
                else:
                    ase_sum += prob_[i,j]*prob_[i,j]
            for j in range(n_classes):
                if predict_[i,j] == 1:
                        conf_mat[idx][j] += 1
                        break
            n_[idx] += 1
            if predict_[i, idx] != 1:
                misc_     += 1
                misc[idx] += 1
        misc_ = 100*misc_/n_obs
        ase   = ase_sum/(n_classes*n_obs)

        #Calculate number of weights
        n_weights = 0
        for i in range(nn.n_layers_ - 1):
            n_weights += len(nn.intercepts_[i])
            n_weights += nn.coefs_[i].shape[0]*nn.coefs_[i].shape[1]
        print("\nModel Metrics")
        print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
        print("{:.<27s}{:10d}".format('Features', X.shape[1]))
        print("{:.<27s}{:10d}".format('Number of Layers',\
                              nn.n_layers_-2))
        print("{:.<27s}{:10d}".format('Number of Outputs', \
                              nn.n_outputs_))
        print("{:.<27s}{:10d}".format('Number of Weights', \
                             n_weights))
        print("{:.<27s}{:10d}".format('Number of Iterations', \
                             nn.n_iter_))
        print("{:.<27s}{:>10s}".format('Activation Function', \
                             nn.out_activation_))
        print("{:.<27s}{:10.4f}".format('Avg. Squared Error', ase))
        print("{:.<27s}{:10.1f}{:s}".format(\
                'MISC (Misclassification)', misc_, '%'))
        for i in range(n_classes):
            misc[i] = 100*misc[i]/n_[i]
            print("{:s}{:.<16.0f}{:>10.1f}{:<1s}".format(\
                  '     class ', nn.classes_[i], misc[i], '%'))
        print("\n\n     Confusion")
        print("       Matrix    ", end="")
        for i in range(n_classes):
            print("{:>7s}{:<3.0f}".format('Class ', nn.classes_[i]), end="")
        print("")
        for i in range(n_classes):
            print("{:s}{:.<6.0f}".format('Class ', nn.classes_[i]), end="")
            for j in range(n_classes):
                print("{:>10d}".format(conf_mat[i][j]), end="")
            print("")

        cr = classification_report(y, predict_, nn.classes_)
        print("\n",cr)

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
    
class Sentiment(object):
    
    def __init__(self, sentiment_dic=None, n_terms=4):
        self.n_terms=n_terms
        self.sentiment_dic = {}
        if sentiment_dic==None or sentiment_dic.lower()=='afinn':
            dic={
                            "abandon": -2,
                            "abandoned": -2,
                            "abandons": -2,
                            "abducted": -2,
                            "abduction": -2,
                            "abductions": -2,
                            "abhor": -3,
                            "abhorred": -3,
                            "abhorrent": -3,
                            "abhors": -3,
                            "abilities": 2,
                            "ability": 2,
                            "aboard": 1,
                            "absentee": -1,
                            "absentees": -1,
                            "absolve": 2,
                            "absolved": 2,
                            "absolves": 2,
                            "absolving": 2,
                            "absorbed": 1,
                            "abuse": -3,
                            "abused": -3,
                            "abuses": -3,
                            "abusive": -3,
                            "accept": 1,
                            "accepted": 1,
                            "accepting": 1,
                            "accepts": 1,
                            "accident": -2,
                            "accidental": -2,
                            "accidentally": -2,
                            "accidents": -2,
                            "accomplish": 2,
                            "accomplished": 2,
                            "accomplishes": 2,
                            "accusation": -2,
                            "accusations": -2,
                            "accuse": -2,
                            "accused": -2,
                            "accuses": -2,
                            "accusing": -2,
                            "ache": -2,
                            "achievable": 1,
                            "aching": -2,
                            "acquit": 2,
                            "acquits": 2,
                            "acquitted": 2,
                            "acquitting": 2,
                            "acrimonious": -3,
                            "active": 1,
                            "adequate": 1,
                            "admire": 3,
                            "admired": 3,
                            "admires": 3,
                            "admiring": 3,
                            "admit": -1,
                            "admits": -1,
                            "admitted": -1,
                            "admonish": -2,
                            "admonished": -2,
                            "adopt": 1,
                            "adopts": 1,
                            "adorable": 3,
                            "adore": 3,
                            "adored": 3,
                            "adores": 3,
                            "advanced": 1,
                            "advantage": 2,
                            "advantages": 2,
                            "adventure": 2,
                            "adventures": 2,
                            "adventurous": 2,
                            "affected": -1,
                            "affection": 3,
                            "affectionate": 3,
                            "afflicted": -1,
                            "affronted": -1,
                            "afraid": -2,
                            "aggravate": -2,
                            "aggravated": -2,
                            "aggravates": -2,
                            "aggravating": -2,
                            "aggression": -2,
                            "aggressions": -2,
                            "aggressive": -2,
                            "aghast": -2,
                            "agog": 2,
                            "agonise": -3,
                            "agonised": -3,
                            "agonises": -3,
                            "agonising": -3,
                            "agonize": -3,
                            "agonized": -3,
                            "agonizes": -3,
                            "agonizing": -3,
                            "agree": 1,
                            "agreeable": 2,
                            "agreed": 1,
                            "agreement": 1,
                            "agrees": 1,
                            "alarm": -2,
                            "alarmed": -2,
                            "alarmist": -2,
                            "alarmists": -2,
                            "alas": -1,
                            "alert": -1,
                            "alienation": -2,
                            "alive": 1,
                            "allergic": -2,
                            "allow": 1,
                            "alone": -2,
                            "amaze": 2,
                            "amazed": 2,
                            "amazes": 2,
                            "amazing": 4,
                            "ambitious": 2,
                            "ambivalent": -1,
                            "amuse": 3,
                            "amused": 3,
                            "amusement": 3,
                            "amusements": 3,
                            "anger": -3,
                            "angers": -3,
                            "angry": -3,
                            "anguish": -3,
                            "anguished": -3,
                            "animosity": -2,
                            "annoy": -2,
                            "annoyance": -2,
                            "annoyed": -2,
                            "annoying": -2,
                            "annoys": -2,
                            "antagonistic": -2,
                            "anti": -1,
                            "anticipation": 1,
                            "anxiety": -2,
                            "anxious": -2,
                            "apathetic": -3,
                            "apathy": -3,
                            "apeshit": -3,
                            "apocalyptic": -2,
                            "apologise": -1,
                            "apologised": -1,
                            "apologises": -1,
                            "apologising": -1,
                            "apologize": -1,
                            "apologized": -1,
                            "apologizes": -1,
                            "apologizing": -1,
                            "apology": -1,
                            "appalled": -2,
                            "appalling": -2,
                            "appease": 2,
                            "appeased": 2,
                            "appeases": 2,
                            "appeasing": 2,
                            "applaud": 2,
                            "applauded": 2,
                            "applauding": 2,
                            "applauds": 2,
                            "applause": 2,
                            "appreciate": 2,
                            "appreciated": 2,
                            "appreciates": 2,
                            "appreciating": 2,
                            "appreciation": 2,
                            "apprehensive": -2,
                            "approval": 2,
                            "approved": 2,
                            "approves": 2,
                            "ardent": 1,
                            "arrest": -2,
                            "arrested": -3,
                            "arrests": -2,
                            "arrogant": -2,
                            "ashame": -2,
                            "ashamed": -2,
                            "ass": -4,
                            "assassination": -3,
                            "assassinations": -3,
                            "asset": 2,
                            "assets": 2,
                            "assfucking": -4,
                            "asshole": -4,
                            "astonished": 2,
                            "astound": 3,
                            "astounded": 3,
                            "astounding": 3,
                            "astoundingly": 3,
                            "astounds": 3,
                            "attack": -1,
                            "attacked": -1,
                            "attacking": -1,
                            "attacks": -1,
                            "attract": 1,
                            "attracted": 1,
                            "attracting": 2,
                            "attraction": 2,
                            "attractions": 2,
                            "attracts": 1,
                            "audacious": 3,
                            "authority": 1,
                            "avert": -1,
                            "averted": -1,
                            "averts": -1,
                            "avid": 2,
                            "avoid": -1,
                            "avoided": -1,
                            "avoids": -1,
                            "await": -1,
                            "awaited": -1,
                            "awaits": -1,
                            "award": 3,
                            "awarded": 3,
                            "awards": 3,
                            "awesome": 4,
                            "awful": -3,
                            "awkward": -2,
                            "axe": -1,
                            "axed": -1,
                            "backed": 1,
                            "backing": 2,
                            "backs": 1,
                            "bad": -3,
                            "badass": -3,
                            "badly": -3,
                            "bailout": -2,
                            "bamboozle": -2,
                            "bamboozled": -2,
                            "bamboozles": -2,
                            "ban": -2,
                            "banish": -1,
                            "bankrupt": -3,
                            "bankster": -3,
                            "banned": -2,
                            "bargain": 2,
                            "barrier": -2,
                            "bastard": -5,
                            "bastards": -5,
                            "battle": -1,
                            "battles": -1,
                            "beaten": -2,
                            "beatific": 3,
                            "beating": -1,
                            "beauties": 3,
                            "beautiful": 3,
                            "beautifully": 3,
                            "beautify": 3,
                            "belittle": -2,
                            "belittled": -2,
                            "beloved": 3,
                            "benefit": 2,
                            "benefits": 2,
                            "benefitted": 2,
                            "benefitting": 2,
                            "bereave": -2,
                            "bereaved": -2,
                            "bereaves": -2,
                            "bereaving": -2,
                            "best": 3,
                            "betray": -3,
                            "betrayal": -3,
                            "betrayed": -3,
                            "betraying": -3,
                            "betrays": -3,
                            "better": 2,
                            "bias": -1,
                            "biased": -2,
                            "big": 1,
                            "bitch": -5,
                            "bitches": -5,
                            "bitter": -2,
                            "bitterly": -2,
                            "bizarre": -2,
                            "blah": -2,
                            "blame": -2,
                            "blamed": -2,
                            "blames": -2,
                            "blaming": -2,
                            "bless": 2,
                            "blesses": 2,
                            "blessing": 3,
                            "blind": -1,
                            "bliss": 3,
                            "blissful": 3,
                            "blithe": 2,
                            "block": -1,
                            "blockbuster": 3,
                            "blocked": -1,
                            "blocking": -1,
                            "blocks": -1,
                            "bloody": -3,
                            "blurry": -2,
                            "boastful": -2,
                            "bold": 2,
                            "boldly": 2,
                            "bomb": -1,
                            "boost": 1,
                            "boosted": 1,
                            "boosting": 1,
                            "boosts": 1,
                            "bore": -2,
                            "bored": -2,
                            "boring": -3,
                            "bother": -2,
                            "bothered": -2,
                            "bothers": -2,
                            "bothersome": -2,
                            "boycott": -2,
                            "boycotted": -2,
                            "boycotting": -2,
                            "boycotts": -2,
                            "brainwashing": -3,
                            "brave": 2,
                            "breakthrough": 3,
                            "breathtaking": 5,
                            "bribe": -3,
                            "bright": 1,
                            "brightest": 2,
                            "brightness": 1,
                            "brilliant": 4,
                            "brisk": 2,
                            "broke": -1,
                            "broken": -1,
                            "brooding": -2,
                            "bullied": -2,
                            "bullshit": -4,
                            "bully": -2,
                            "bullying": -2,
                            "bummer": -2,
                            "buoyant": 2,
                            "burden": -2,
                            "burdened": -2,
                            "burdening": -2,
                            "burdens": -2,
                            "calm": 2,
                            "calmed": 2,
                            "calming": 2,
                            "calms": 2,
                            "not stand": -3,
                            "cancel": -1,
                            "cancelled": -1,
                            "cancelling": -1,
                            "cancels": -1,
                            "cancer": -1,
                            "capable": 1,
                            "captivated": 3,
                            "care": 2,
                            "carefree": 1,
                            "careful": 2,
                            "carefully": 2,
                            "careless": -2,
                            "cares": 2,
                            "cashing in": -2,
                            "casualty": -2,
                            "catastrophe": -3,
                            "catastrophic": -4,
                            "cautious": -1,
                            "celebrate": 3,
                            "celebrated": 3,
                            "celebrates": 3,
                            "celebrating": 3,
                            "censor": -2,
                            "censored": -2,
                            "censors": -2,
                            "certain": 1,
                            "chagrin": -2,
                            "chagrined": -2,
                            "challenge": -1,
                            "chance": 2,
                            "chances": 2,
                            "chaos": -2,
                            "chaotic": -2,
                            "charged": -3,
                            "charges": -2,
                            "charm": 3,
                            "charming": 3,
                            "charmless": -3,
                            "chastise": -3,
                            "chastised": -3,
                            "chastises": -3,
                            "chastising": -3,
                            "cheat": -3,
                            "cheated": -3,
                            "cheater": -3,
                            "cheaters": -3,
                            "cheats": -3,
                            "cheer": 2,
                            "cheered": 2,
                            "cheerful": 2,
                            "cheering": 2,
                            "cheerless": -2,
                            "cheers": 2,
                            "cheery": 3,
                            "cherish": 2,
                            "cherished": 2,
                            "cherishes": 2,
                            "cherishing": 2,
                            "chic": 2,
                            "childish": -2,
                            "chilling": -1,
                            "choke": -2,
                            "choked": -2,
                            "chokes": -2,
                            "choking": -2,
                            "clarifies": 2,
                            "clarity": 2,
                            "clash": -2,
                            "classy": 3,
                            "clean": 2,
                            "cleaner": 2,
                            "clear": 1,
                            "cleared": 1,
                            "clearly": 1,
                            "clears": 1,
                            "clever": 2,
                            "clouded": -1,
                            "clueless": -2,
                            "cock": -5,
                            "cocksucker": -5,
                            "cocksuckers": -5,
                            "cocky": -2,
                            "coerced": -2,
                            "collapse": -2,
                            "collapsed": -2,
                            "collapses": -2,
                            "collapsing": -2,
                            "collide": -1,
                            "collides": -1,
                            "colliding": -1,
                            "collision": -2,
                            "collisions": -2,
                            "colluding": -3,
                            "combat": -1,
                            "combats": -1,
                            "comedy": 1,
                            "comfort": 2,
                            "comfortable": 2,
                            "comforting": 2,
                            "comforts": 2,
                            "commend": 2,
                            "commended": 2,
                            "commit": 1,
                            "commitment": 2,
                            "commits": 1,
                            "committed": 1,
                            "committing": 1,
                            "compassionate": 2,
                            "compelled": 1,
                            "competent": 2,
                            "competitive": 2,
                            "complacent": -2,
                            "complain": -2,
                            "complained": -2,
                            "complains": -2,
                            "comprehensive": 2,
                            "conciliate": 2,
                            "conciliated": 2,
                            "conciliates": 2,
                            "conciliating": 2,
                            "condemn": -2,
                            "condemnation": -2,
                            "condemned": -2,
                            "condemns": -2,
                            "confidence": 2,
                            "confident": 2,
                            "conflict": -2,
                            "conflicting": -2,
                            "conflictive": -2,
                            "conflicts": -2,
                            "confuse": -2,
                            "confused": -2,
                            "confusing": -2,
                            "congrats": 2,
                            "congratulate": 2,
                            "congratulation": 2,
                            "congratulations": 2,
                            "consent": 2,
                            "consents": 2,
                            "consolable": 2,
                            "conspiracy": -3,
                            "constrained": -2,
                            "contagion": -2,
                            "contagions": -2,
                            "contagious": -1,
                            "contempt": -2,
                            "contemptuous": -2,
                            "contemptuously": -2,
                            "contend": -1,
                            "contender": -1,
                            "contending": -1,
                            "contentious": -2,
                            "contestable": -2,
                            "controversial": -2,
                            "controversially": -2,
                            "convince": 1,
                            "convinced": 1,
                            "convinces": 1,
                            "convivial": 2,
                            "cool": 1,
                            "cool stuff": 3,
                            "cornered": -2,
                            "corpse": -1,
                            "costly": -2,
                            "courage": 2,
                            "courageous": 2,
                            "courteous": 2,
                            "courtesy": 2,
                            "cover-up": -3,
                            "coward": -2,
                            "cowardly": -2,
                            "coziness": 2,
                            "cramp": -1,
                            "crap": -3,
                            "crash": -2,
                            "crazier": -2,
                            "craziest": -2,
                            "crazy": -2,
                            "creative": 2,
                            "crestfallen": -2,
                            "cried": -2,
                            "cries": -2,
                            "crime": -3,
                            "criminal": -3,
                            "criminals": -3,
                            "crisis": -3,
                            "critic": -2,
                            "criticism": -2,
                            "criticize": -2,
                            "criticized": -2,
                            "criticizes": -2,
                            "criticizing": -2,
                            "critics": -2,
                            "cruel": -3,
                            "cruelty": -3,
                            "crush": -1,
                            "crushed": -2,
                            "crushes": -1,
                            "crushing": -1,
                            "cry": -1,
                            "crying": -2,
                            "cunt": -5,
                            "curious": 1,
                            "curse": -1,
                            "cut": -1,
                            "cute": 2,
                            "cuts": -1,
                            "cutting": -1,
                            "cynic": -2,
                            "cynical": -2,
                            "cynicism": -2,
                            "damage": -3,
                            "damages": -3,
                            "damn": -4,
                            "damned": -4,
                            "damnit": -4,
                            "danger": -2,
                            "daredevil": 2,
                            "daring": 2,
                            "darkest": -2,
                            "darkness": -1,
                            "dauntless": 2,
                            "dead": -3,
                            "deadlock": -2,
                            "deafening": -1,
                            "dear": 2,
                            "dearly": 3,
                            "death": -2,
                            "debonair": 2,
                            "debt": -2,
                            "deceit": -3,
                            "deceitful": -3,
                            "deceive": -3,
                            "deceived": -3,
                            "deceives": -3,
                            "deceiving": -3,
                            "deception": -3,
                            "decisive": 1,
                            "dedicated": 2,
                            "defeated": -2,
                            "defect": -3,
                            "defects": -3,
                            "defender": 2,
                            "defenders": 2,
                            "defenseless": -2,
                            "defer": -1,
                            "deferring": -1,
                            "defiant": -1,
                            "deficit": -2,
                            "degrade": -2,
                            "degraded": -2,
                            "degrades": -2,
                            "dehumanize": -2,
                            "dehumanized": -2,
                            "dehumanizes": -2,
                            "dehumanizing": -2,
                            "deject": -2,
                            "dejected": -2,
                            "dejecting": -2,
                            "dejects": -2,
                            "delay": -1,
                            "delayed": -1,
                            "delight": 3,
                            "delighted": 3,
                            "delighting": 3,
                            "delights": 3,
                            "demand": -1,
                            "demanded": -1,
                            "demanding": -1,
                            "demands": -1,
                            "demonstration": -1,
                            "demoralized": -2,
                            "denied": -2,
                            "denier": -2,
                            "deniers": -2,
                            "denies": -2,
                            "denounce": -2,
                            "denounces": -2,
                            "deny": -2,
                            "denying": -2,
                            "depressed": -2,
                            "depressing": -2,
                            "derail": -2,
                            "derailed": -2,
                            "derails": -2,
                            "deride": -2,
                            "derided": -2,
                            "derides": -2,
                            "deriding": -2,
                            "derision": -2,
                            "desirable": 2,
                            "desire": 1,
                            "desired": 2,
                            "desirous": 2,
                            "despair": -3,
                            "despairing": -3,
                            "despairs": -3,
                            "desperate": -3,
                            "desperately": -3,
                            "despondent": -3,
                            "destroy": -3,
                            "destroyed": -3,
                            "destroying": -3,
                            "destroys": -3,
                            "destruction": -3,
                            "destructive": -3,
                            "detached": -1,
                            "detain": -2,
                            "detained": -2,
                            "detention": -2,
                            "determined": 2,
                            "devastate": -2,
                            "devastated": -2,
                            "devastating": -2,
                            "devoted": 3,
                            "diamond": 1,
                            "dick": -4,
                            "dickhead": -4,
                            "die": -3,
                            "died": -3,
                            "difficult": -1,
                            "diffident": -2,
                            "dilemma": -1,
                            "dipshit": -3,
                            "dire": -3,
                            "direful": -3,
                            "dirt": -2,
                            "dirtier": -2,
                            "dirtiest": -2,
                            "dirty": -2,
                            "disabling": -1,
                            "disadvantage": -2,
                            "disadvantaged": -2,
                            "disappear": -1,
                            "disappeared": -1,
                            "disappears": -1,
                            "disappoint": -2,
                            "disappointed": -2,
                            "disappointing": -2,
                            "disappointment": -2,
                            "disappointments": -2,
                            "disappoints": -2,
                            "disaster": -2,
                            "disasters": -2,
                            "disastrous": -3,
                            "disbelieve": -2,
                            "discard": -1,
                            "discarded": -1,
                            "discarding": -1,
                            "discards": -1,
                            "disconsolate": -2,
                            "disconsolation": -2,
                            "discontented": -2,
                            "discord": -2,
                            "discounted": -1,
                            "discouraged": -2,
                            "discredited": -2,
                            "disdain": -2,
                            "disgrace": -2,
                            "disgraced": -2,
                            "disguise": -1,
                            "disguised": -1,
                            "disguises": -1,
                            "disguising": -1,
                            "disgust": -3,
                            "disgusted": -3,
                            "disgusting": -3,
                            "disheartened": -2,
                            "dishonest": -2,
                            "disillusioned": -2,
                            "disinclined": -2,
                            "disjointed": -2,
                            "dislike": -2,
                            "dismal": -2,
                            "dismayed": -2,
                            "disorder": -2,
                            "disorganized": -2,
                            "disoriented": -2,
                            "disparage": -2,
                            "disparaged": -2,
                            "disparages": -2,
                            "disparaging": -2,
                            "displeased": -2,
                            "dispute": -2,
                            "disputed": -2,
                            "disputes": -2,
                            "disputing": -2,
                            "disqualified": -2,
                            "disquiet": -2,
                            "disregard": -2,
                            "disregarded": -2,
                            "disregarding": -2,
                            "disregards": -2,
                            "disrespect": -2,
                            "disrespected": -2,
                            "disruption": -2,
                            "disruptions": -2,
                            "disruptive": -2,
                            "dissatisfied": -2,
                            "distort": -2,
                            "distorted": -2,
                            "distorting": -2,
                            "distorts": -2,
                            "distract": -2,
                            "distracted": -2,
                            "distraction": -2,
                            "distracts": -2,
                            "distress": -2,
                            "distressed": -2,
                            "distresses": -2,
                            "distressing": -2,
                            "distrust": -3,
                            "distrustful": -3,
                            "disturb": -2,
                            "disturbed": -2,
                            "disturbing": -2,
                            "disturbs": -2,
                            "dithering": -2,
                            "dizzy": -1,
                            "dodging": -2,
                            "dodgy": -2,
                            "does not work": -3,
                            "dolorous": -2,
                            "dont like": -2,
                            "doom": -2,
                            "doomed": -2,
                            "doubt": -1,
                            "doubted": -1,
                            "doubtful": -1,
                            "doubting": -1,
                            "doubts": -1,
                            "douche": -3,
                            "douchebag": -3,
                            "downcast": -2,
                            "downhearted": -2,
                            "downside": -2,
                            "drag": -1,
                            "dragged": -1,
                            "drags": -1,
                            "drained": -2,
                            "dread": -2,
                            "dreaded": -2,
                            "dreadful": -3,
                            "dreading": -2,
                            "dream": 1,
                            "dreams": 1,
                            "dreary": -2,
                            "droopy": -2,
                            "drop": -1,
                            "drown": -2,
                            "drowned": -2,
                            "drowns": -2,
                            "drunk": -2,
                            "dubious": -2,
                            "dud": -2,
                            "dull": -2,
                            "dumb": -3,
                            "dumbass": -3,
                            "dump": -1,
                            "dumped": -2,
                            "dumps": -1,
                            "dupe": -2,
                            "duped": -2,
                            "dysfunction": -2,
                            "eager": 2,
                            "earnest": 2,
                            "ease": 2,
                            "easy": 1,
                            "ecstatic": 4,
                            "eerie": -2,
                            "eery": -2,
                            "effective": 2,
                            "effectively": 2,
                            "elated": 3,
                            "elation": 3,
                            "elegant": 2,
                            "elegantly": 2,
                            "embarrass": -2,
                            "embarrassed": -2,
                            "embarrasses": -2,
                            "embarrassing": -2,
                            "embarrassment": -2,
                            "embittered": -2,
                            "embrace": 1,
                            "emergency": -2,
                            "empathetic": 2,
                            "emptiness": -1,
                            "empty": -1,
                            "enchanted": 2,
                            "encourage": 2,
                            "encouraged": 2,
                            "encouragement": 2,
                            "encourages": 2,
                            "endorse": 2,
                            "endorsed": 2,
                            "endorsement": 2,
                            "endorses": 2,
                            "enemies": -2,
                            "enemy": -2,
                            "energetic": 2,
                            "engage": 1,
                            "engages": 1,
                            "engrossed": 1,
                            "enjoy": 2,
                            "enjoying": 2,
                            "enjoys": 2,
                            "enlighten": 2,
                            "enlightened": 2,
                            "enlightening": 2,
                            "enlightens": 2,
                            "ennui": -2,
                            "enrage": -2,
                            "enraged": -2,
                            "enrages": -2,
                            "enraging": -2,
                            "enrapture": 3,
                            "enslave": -2,
                            "enslaved": -2,
                            "enslaves": -2,
                            "ensure": 1,
                            "ensuring": 1,
                            "enterprising": 1,
                            "entertaining": 2,
                            "enthral": 3,
                            "enthusiastic": 3,
                            "entitled": 1,
                            "entrusted": 2,
                            "envies": -1,
                            "envious": -2,
                            "envy": -1,
                            "envying": -1,
                            "erroneous": -2,
                            "error": -2,
                            "errors": -2,
                            "escape": -1,
                            "escapes": -1,
                            "escaping": -1,
                            "esteemed": 2,
                            "ethical": 2,
                            "euphoria": 3,
                            "euphoric": 4,
                            "eviction": -1,
                            "evil": -3,
                            "exaggerate": -2,
                            "exaggerated": -2,
                            "exaggerates": -2,
                            "exaggerating": -2,
                            "exasperated": 2,
                            "excellence": 3,
                            "excellent": 3,
                            "excite": 3,
                            "excited": 3,
                            "excitement": 3,
                            "exciting": 3,
                            "exclude": -1,
                            "excluded": -2,
                            "exclusion": -1,
                            "exclusive": 2,
                            "excuse": -1,
                            "exempt": -1,
                            "exhausted": -2,
                            "exhilarated": 3,
                            "exhilarates": 3,
                            "exhilarating": 3,
                            "exonerate": 2,
                            "exonerated": 2,
                            "exonerates": 2,
                            "exonerating": 2,
                            "expand": 1,
                            "expands": 1,
                            "expel": -2,
                            "expelled": -2,
                            "expelling": -2,
                            "expels": -2,
                            "exploit": -2,
                            "exploited": -2,
                            "exploiting": -2,
                            "exploits": -2,
                            "exploration": 1,
                            "explorations": 1,
                            "expose": -1,
                            "exposed": -1,
                            "exposes": -1,
                            "exposing": -1,
                            "extend": 1,
                            "extends": 1,
                            "exuberant": 4,
                            "exultant": 3,
                            "exultantly": 3,
                            "fabulous": 4,
                            "fad": -2,
                            "fag": -3,
                            "faggot": -3,
                            "faggots": -3,
                            "fail": -2,
                            "failed": -2,
                            "failing": -2,
                            "fails": -2,
                            "failure": -2,
                            "failures": -2,
                            "fainthearted": -2,
                            "fair": 2,
                            "faith": 1,
                            "faithful": 3,
                            "fake": -3,
                            "fakes": -3,
                            "faking": -3,
                            "fallen": -2,
                            "falling": -1,
                            "falsified": -3,
                            "falsify": -3,
                            "fame": 1,
                            "fan": 3,
                            "fantastic": 4,
                            "farce": -1,
                            "fascinate": 3,
                            "fascinated": 3,
                            "fascinates": 3,
                            "fascinating": 3,
                            "fascist": -2,
                            "fascists": -2,
                            "fatalities": -3,
                            "fatality": -3,
                            "fatigue": -2,
                            "fatigued": -2,
                            "fatigues": -2,
                            "fatiguing": -2,
                            "favor": 2,
                            "favored": 2,
                            "favorite": 2,
                            "favorited": 2,
                            "favorites": 2,
                            "favors": 2,
                            "fear": -2,
                            "fearful": -2,
                            "fearing": -2,
                            "fearless": 2,
                            "fearsome": -2,
                            "fed up": -3,
                            "feeble": -2,
                            "feeling": 1,
                            "felonies": -3,
                            "felony": -3,
                            "fervent": 2,
                            "fervid": 2,
                            "festive": 2,
                            "fiasco": -3,
                            "fidgety": -2,
                            "fight": -1,
                            "fine": 2,
                            "fire": -2,
                            "fired": -2,
                            "firing": -2,
                            "fit": 1,
                            "fitness": 1,
                            "flagship": 2,
                            "flees": -1,
                            "flop": -2,
                            "flops": -2,
                            "flu": -2,
                            "flustered": -2,
                            "focused": 2,
                            "fond": 2,
                            "fondness": 2,
                            "fool": -2,
                            "foolish": -2,
                            "fools": -2,
                            "forced": -1,
                            "foreclosure": -2,
                            "foreclosures": -2,
                            "forget": -1,
                            "forgetful": -2,
                            "forgive": 1,
                            "forgiving": 1,
                            "forgotten": -1,
                            "fortunate": 2,
                            "frantic": -1,
                            "fraud": -4,
                            "frauds": -4,
                            "fraudster": -4,
                            "fraudsters": -4,
                            "fraudulence": -4,
                            "fraudulent": -4,
                            "free": 1,
                            "freedom": 2,
                            "frenzy": -3,
                            "fresh": 1,
                            "friendly": 2,
                            "fright": -2,
                            "frightened": -2,
                            "frightening": -3,
                            "frikin": -2,
                            "frisky": 2,
                            "frowning": -1,
                            "frustrate": -2,
                            "frustrated": -2,
                            "frustrates": -2,
                            "frustrating": -2,
                            "frustration": -2,
                            "ftw": 3,
                            "fuck": -4,
                            "fucked": -4,
                            "fucker": -4,
                            "fuckers": -4,
                            "fuckface": -4,
                            "fuckhead": -4,
                            "fucking": -4,
                            "fucktard": -4,
                            "fud": -3,
                            "fuked": -4,
                            "fuking": -4,
                            "fulfill": 2,
                            "fulfilled": 2,
                            "fulfills": 2,
                            "fuming": -2,
                            "fun": 4,
                            "funeral": -1,
                            "funerals": -1,
                            "funky": 2,
                            "funnier": 4,
                            "funny": 4,
                            "furious": -3,
                            "futile": 2,
                            "gag": -2,
                            "gagged": -2,
                            "gain": 2,
                            "gained": 2,
                            "gaining": 2,
                            "gains": 2,
                            "gallant": 3,
                            "gallantly": 3,
                            "gallantry": 3,
                            "generous": 2,
                            "genial": 3,
                            "ghost": -1,
                            "giddy": -2,
                            "gift": 2,
                            "glad": 3,
                            "glamorous": 3,
                            "glamourous": 3,
                            "glee": 3,
                            "gleeful": 3,
                            "gloom": -1,
                            "gloomy": -2,
                            "glorious": 2,
                            "glory": 2,
                            "glum": -2,
                            "god": 1,
                            "goddamn": -3,
                            "godsend": 4,
                            "good": 3,
                            "goodness": 3,
                            "grace": 1,
                            "gracious": 3,
                            "grand": 3,
                            "grant": 1,
                            "granted": 1,
                            "granting": 1,
                            "grants": 1,
                            "grateful": 3,
                            "gratification": 2,
                            "grave": -2,
                            "gray": -1,
                            "great": 3,
                            "greater": 3,
                            "greatest": 3,
                            "greed": -3,
                            "greedy": -2,
                            "green wash": -3,
                            "green washing": -3,
                            "greenwash": -3,
                            "greenwasher": -3,
                            "greenwashers": -3,
                            "greenwashing": -3,
                            "greet": 1,
                            "greeted": 1,
                            "greeting": 1,
                            "greetings": 2,
                            "greets": 1,
                            "grey": -1,
                            "grief": -2,
                            "grieved": -2,
                            "gross": -2,
                            "growing": 1,
                            "growth": 2,
                            "guarantee": 1,
                            "guilt": -3,
                            "guilty": -3,
                            "gullibility": -2,
                            "gullible": -2,
                            "gun": -1,
                            "ha": 2,
                            "hacked": -1,
                            "haha": 3,
                            "hahaha": 3,
                            "hahahah": 3,
                            "hail": 2,
                            "hailed": 2,
                            "hapless": -2,
                            "haplessness": -2,
                            "happiness": 3,
                            "happy": 3,
                            "hard": -1,
                            "hardier": 2,
                            "hardship": -2,
                            "hardy": 2,
                            "harm": -2,
                            "harmed": -2,
                            "harmful": -2,
                            "harming": -2,
                            "harms": -2,
                            "harried": -2,
                            "harsh": -2,
                            "harsher": -2,
                            "harshest": -2,
                            "hate": -3,
                            "hated": -3,
                            "haters": -3,
                            "hates": -3,
                            "hating": -3,
                            "haunt": -1,
                            "haunted": -2,
                            "haunting": 1,
                            "haunts": -1,
                            "havoc": -2,
                            "healthy": 2,
                            "heartbreaking": -3,
                            "heartbroken": -3,
                            "heartfelt": 3,
                            "heaven": 2,
                            "heavenly": 4,
                            "heavyhearted": -2,
                            "hell": -4,
                            "help": 2,
                            "helpful": 2,
                            "helping": 2,
                            "helpless": -2,
                            "helps": 2,
                            "hero": 2,
                            "heroes": 2,
                            "heroic": 3,
                            "hesitant": -2,
                            "hesitate": -2,
                            "hid": -1,
                            "hide": -1,
                            "hides": -1,
                            "hiding": -1,
                            "highlight": 2,
                            "hilarious": 2,
                            "hindrance": -2,
                            "hoax": -2,
                            "homesick": -2,
                            "honest": 2,
                            "honor": 2,
                            "honored": 2,
                            "honoring": 2,
                            "honour": 2,
                            "honoured": 2,
                            "honouring": 2,
                            "hooligan": -2,
                            "hooliganism": -2,
                            "hooligans": -2,
                            "hope": 2,
                            "hopeful": 2,
                            "hopefully": 2,
                            "hopeless": -2,
                            "hopelessness": -2,
                            "hopes": 2,
                            "hoping": 2,
                            "horrendous": -3,
                            "horrible": -3,
                            "horrific": -3,
                            "horrified": -3,
                            "hostile": -2,
                            "huckster": -2,
                            "hug": 2,
                            "huge": 1,
                            "hugs": 2,
                            "humerous": 3,
                            "humiliated": -3,
                            "humiliation": -3,
                            "humor": 2,
                            "humorous": 2,
                            "humour": 2,
                            "humourous": 2,
                            "hunger": -2,
                            "hurrah": 5,
                            "hurt": -2,
                            "hurting": -2,
                            "hurts": -2,
                            "hypocritical": -2,
                            "hysteria": -3,
                            "hysterical": -3,
                            "hysterics": -3,
                            "idiot": -3,
                            "idiotic": -3,
                            "ignorance": -2,
                            "ignorant": -2,
                            "ignore": -1,
                            "ignored": -2,
                            "ignores": -1,
                            "ill": -2,
                            "illegal": -3,
                            "illiteracy": -2,
                            "illness": -2,
                            "illnesses": -2,
                            "imbecile": -3,
                            "immobilized": -1,
                            "immortal": 2,
                            "immune": 1,
                            "impatient": -2,
                            "imperfect": -2,
                            "importance": 2,
                            "important": 2,
                            "impose": -1,
                            "imposed": -1,
                            "imposes": -1,
                            "imposing": -1,
                            "impotent": -2,
                            "impress": 3,
                            "impressed": 3,
                            "impresses": 3,
                            "impressive": 3,
                            "imprisoned": -2,
                            "improve": 2,
                            "improved": 2,
                            "improvement": 2,
                            "improves": 2,
                            "improving": 2,
                            "inability": -2,
                            "inaction": -2,
                            "inadequate": -2,
                            "incapable": -2,
                            "incapacitated": -2,
                            "incensed": -2,
                            "incompetence": -2,
                            "incompetent": -2,
                            "inconsiderate": -2,
                            "inconvenience": -2,
                            "inconvenient": -2,
                            "increase": 1,
                            "increased": 1,
                            "indecisive": -2,
                            "indestructible": 2,
                            "indifference": -2,
                            "indifferent": -2,
                            "indignant": -2,
                            "indignation": -2,
                            "indoctrinate": -2,
                            "indoctrinated": -2,
                            "indoctrinates": -2,
                            "indoctrinating": -2,
                            "ineffective": -2,
                            "ineffectively": -2,
                            "infatuated": 2,
                            "infatuation": 2,
                            "infected": -2,
                            "inferior": -2,
                            "inflamed": -2,
                            "influential": 2,
                            "infringement": -2,
                            "infuriate": -2,
                            "infuriated": -2,
                            "infuriates": -2,
                            "infuriating": -2,
                            "inhibit": -1,
                            "injured": -2,
                            "injury": -2,
                            "injustice": -2,
                            "innovate": 1,
                            "innovates": 1,
                            "innovation": 1,
                            "innovative": 2,
                            "inquisition": -2,
                            "inquisitive": 2,
                            "insane": -2,
                            "insanity": -2,
                            "insecure": -2,
                            "insensitive": -2,
                            "insensitivity": -2,
                            "insignificant": -2,
                            "insipid": -2,
                            "inspiration": 2,
                            "inspirational": 2,
                            "inspire": 2,
                            "inspired": 2,
                            "inspires": 2,
                            "inspiring": 3,
                            "insult": -2,
                            "insulted": -2,
                            "insulting": -2,
                            "insults": -2,
                            "intact": 2,
                            "integrity": 2,
                            "intelligent": 2,
                            "intense": 1,
                            "interest": 1,
                            "interested": 2,
                            "interesting": 2,
                            "interests": 1,
                            "interrogated": -2,
                            "interrupt": -2,
                            "interrupted": -2,
                            "interrupting": -2,
                            "interruption": -2,
                            "interrupts": -2,
                            "intimidate": -2,
                            "intimidated": -2,
                            "intimidates": -2,
                            "intimidating": -2,
                            "intimidation": -2,
                            "intricate": 2,
                            "intrigues": 1,
                            "invincible": 2,
                            "invite": 1,
                            "inviting": 1,
                            "invulnerable": 2,
                            "irate": -3,
                            "ironic": -1,
                            "irony": -1,
                            "irrational": -1,
                            "irresistible": 2,
                            "irresolute": -2,
                            "irresponsible": 2,
                            "irreversible": -1,
                            "irritate": -3,
                            "irritated": -3,
                            "irritating": -3,
                            "isolated": -1,
                            "itchy": -2,
                            "jackass": -4,
                            "jackasses": -4,
                            "jailed": -2,
                            "jaunty": 2,
                            "jealous": -2,
                            "jeopardy": -2,
                            "jerk": -3,
                            "jesus": 1,
                            "jewel": 1,
                            "jewels": 1,
                            "jocular": 2,
                            "join": 1,
                            "joke": 2,
                            "jokes": 2,
                            "jolly": 2,
                            "jovial": 2,
                            "joy": 3,
                            "joyful": 3,
                            "joyfully": 3,
                            "joyless": -2,
                            "joyous": 3,
                            "jubilant": 3,
                            "jumpy": -1,
                            "justice": 2,
                            "justifiably": 2,
                            "justified": 2,
                            "keen": 1,
                            "kill": -3,
                            "killed": -3,
                            "killing": -3,
                            "kills": -3,
                            "kind": 2,
                            "kinder": 2,
                            "kiss": 2,
                            "kudos": 3,
                            "lack": -2,
                            "lackadaisical": -2,
                            "lag": -1,
                            "lagged": -2,
                            "lagging": -2,
                            "lags": -2,
                            "lame": -2,
                            "landmark": 2,
                            "laugh": 1,
                            "laughed": 1,
                            "laughing": 1,
                            "laughs": 1,
                            "laughting": 1,
                            "launched": 1,
                            "lawl": 3,
                            "lawsuit": -2,
                            "lawsuits": -2,
                            "lazy": -1,
                            "leak": -1,
                            "leaked": -1,
                            "leave": -1,
                            "legal": 1,
                            "legally": 1,
                            "lenient": 1,
                            "lethargic": -2,
                            "lethargy": -2,
                            "liar": -3,
                            "liars": -3,
                            "libelous": -2,
                            "lied": -2,
                            "lifesaver": 4,
                            "lighthearted": 1,
                            "like": 2,
                            "liked": 2,
                            "likes": 2,
                            "limitation": -1,
                            "limited": -1,
                            "limits": -1,
                            "litigation": -1,
                            "litigious": -2,
                            "lively": 2,
                            "livid": -2,
                            "lmao": 4,
                            "lmfao": 4,
                            "loathe": -3,
                            "loathed": -3,
                            "loathes": -3,
                            "loathing": -3,
                            "lobby": -2,
                            "lobbying": -2,
                            "lol": 3,
                            "lonely": -2,
                            "lonesome": -2,
                            "longing": -1,
                            "loom": -1,
                            "loomed": -1,
                            "looming": -1,
                            "looms": -1,
                            "loose": -3,
                            "looses": -3,
                            "loser": -3,
                            "losing": -3,
                            "loss": -3,
                            "lost": -3,
                            "lovable": 3,
                            "love": 3,
                            "loved": 3,
                            "lovelies": 3,
                            "lovely": 3,
                            "loving": 2,
                            "lowest": -1,
                            "loyal": 3,
                            "loyalty": 3,
                            "luck": 3,
                            "luckily": 3,
                            "lucky": 3,
                            "lugubrious": -2,
                            "lunatic": -3,
                            "lunatics": -3,
                            "lurk": -1,
                            "lurking": -1,
                            "lurks": -1,
                            "mad": -3,
                            "maddening": -3,
                            "made-up": -1,
                            "madly": -3,
                            "madness": -3,
                            "mandatory": -1,
                            "manipulated": -1,
                            "manipulating": -1,
                            "manipulation": -1,
                            "marvel": 3,
                            "marvelous": 3,
                            "marvels": 3,
                            "masterpiece": 4,
                            "masterpieces": 4,
                            "matter": 1,
                            "matters": 1,
                            "mature": 2,
                            "meaningful": 2,
                            "meaningless": -2,
                            "medal": 3,
                            "mediocrity": -3,
                            "meditative": 1,
                            "melancholy": -2,
                            "menace": -2,
                            "menaced": -2,
                            "mercy": 2,
                            "merry": 3,
                            "mess": -2,
                            "messed": -2,
                            "messing up": -2,
                            "methodical": 2,
                            "mindless": -2,
                            "miracle": 4,
                            "mirth": 3,
                            "mirthful": 3,
                            "mirthfully": 3,
                            "misbehave": -2,
                            "misbehaved": -2,
                            "misbehaves": -2,
                            "misbehaving": -2,
                            "mischief": -1,
                            "mischiefs": -1,
                            "miserable": -3,
                            "misery": -2,
                            "misgiving": -2,
                            "misinformation": -2,
                            "misinformed": -2,
                            "misinterpreted": -2,
                            "misleading": -3,
                            "misread": -1,
                            "misreporting": -2,
                            "misrepresentation": -2,
                            "miss": -2,
                            "missed": -2,
                            "missing": -2,
                            "mistake": -2,
                            "mistaken": -2,
                            "mistakes": -2,
                            "mistaking": -2,
                            "misunderstand": -2,
                            "misunderstanding": -2,
                            "misunderstands": -2,
                            "misunderstood": -2,
                            "moan": -2,
                            "moaned": -2,
                            "moaning": -2,
                            "moans": -2,
                            "mock": -2,
                            "mocked": -2,
                            "mocking": -2,
                            "mocks": -2,
                            "mongering": -2,
                            "monopolize": -2,
                            "monopolized": -2,
                            "monopolizes": -2,
                            "monopolizing": -2,
                            "moody": -1,
                            "mope": -1,
                            "moping": -1,
                            "moron": -3,
                            "motherfucker": -5,
                            "motherfucking": -5,
                            "motivate": 1,
                            "motivated": 2,
                            "motivating": 2,
                            "motivation": 1,
                            "mourn": -2,
                            "mourned": -2,
                            "mournful": -2,
                            "mourning": -2,
                            "mourns": -2,
                            "mumpish": -2,
                            "murder": -2,
                            "murderer": -2,
                            "murdering": -3,
                            "murderous": -3,
                            "murders": -2,
                            "myth": -1,
                            "n00b": -2,
                            "naive": -2,
                            "nasty": -3,
                            "natural": 1,
                            "naïve": -2,
                            "needy": -2,
                            "negative": -2,
                            "negativity": -2,
                            "neglect": -2,
                            "neglected": -2,
                            "neglecting": -2,
                            "neglects": -2,
                            "nerves": -1,
                            "nervous": -2,
                            "nervously": -2,
                            "nice": 3,
                            "nifty": 2,
                            "no": -1,
                            "no fun": -3,
                            "noble": 2,
                            "noisy": -1,
                            "nonsense": -2,
                            "noob": -2,
                            "nosey": -2,
                            "not good": -2,
                            "not working": -3,
                            "notorious": -2,
                            "novel": 2,
                            "numb": -1,
                            "nuts": -3,
                            "obliterate": -2,
                            "obliterated": -2,
                            "obnoxious": -3,
                            "obscene": -2,
                            "obsessed": 2,
                            "obsolete": -2,
                            "obstacle": -2,
                            "obstacles": -2,
                            "obstinate": -2,
                            "odd": -2,
                            "offend": -2,
                            "offended": -2,
                            "offender": -2,
                            "offending": -2,
                            "offends": -2,
                            "offline": -1,
                            "oks": 2,
                            "ominous": 3,
                            "once-in-a-lifetime": 3,
                            "opportunities": 2,
                            "opportunity": 2,
                            "oppressed": -2,
                            "oppressive": -2,
                            "optimism": 2,
                            "optimistic": 2,
                            "optionless": -2,
                            "outcry": -2,
                            "outmaneuvered": -2,
                            "outrage": -3,
                            "outraged": -3,
                            "outreach": 2,
                            "outstanding": 5,
                            "overjoyed": 4,
                            "overload": -1,
                            "overlooked": -1,
                            "overreact": -2,
                            "overreacted": -2,
                            "overreaction": -2,
                            "overreacts": -2,
                            "oversell": -2,
                            "overselling": -2,
                            "oversells": -2,
                            "oversimplification": -2,
                            "oversimplified": -2,
                            "oversimplifies": -2,
                            "oversimplify": -2,
                            "overstatement": -2,
                            "overstatements": -2,
                            "overweight": -1,
                            "oxymoron": -1,
                            "pain": -2,
                            "pained": -2,
                            "panic": -3,
                            "panicked": -3,
                            "panics": -3,
                            "paradise": 3,
                            "paradox": -1,
                            "pardon": 2,
                            "pardoned": 2,
                            "pardoning": 2,
                            "pardons": 2,
                            "parley": -1,
                            "passionate": 2,
                            "passive": -1,
                            "passively": -1,
                            "pathetic": -2,
                            "pay": -1,
                            "peace": 2,
                            "peaceful": 2,
                            "peacefully": 2,
                            "penalty": -2,
                            "pensive": -1,
                            "perfect": 3,
                            "perfected": 2,
                            "perfectly": 3,
                            "perfects": 2,
                            "peril": -2,
                            "perjury": -3,
                            "perpetrator": -2,
                            "perpetrators": -2,
                            "perplexed": -2,
                            "persecute": -2,
                            "persecuted": -2,
                            "persecutes": -2,
                            "persecuting": -2,
                            "perturbed": -2,
                            "pesky": -2,
                            "pessimism": -2,
                            "pessimistic": -2,
                            "petrified": -2,
                            "phobic": -2,
                            "picturesque": 2,
                            "pileup": -1,
                            "pique": -2,
                            "piqued": -2,
                            "piss": -4,
                            "pissed": -4,
                            "pissing": -3,
                            "piteous": -2,
                            "pitied": -1,
                            "pity": -2,
                            "playful": 2,
                            "pleasant": 3,
                            "please": 1,
                            "pleased": 3,
                            "pleasure": 3,
                            "poised": -2,
                            "poison": -2,
                            "poisoned": -2,
                            "poisons": -2,
                            "pollute": -2,
                            "polluted": -2,
                            "polluter": -2,
                            "polluters": -2,
                            "pollutes": -2,
                            "poor": -2,
                            "poorer": -2,
                            "poorest": -2,
                            "popular": 3,
                            "positive": 2,
                            "positively": 2,
                            "possessive": -2,
                            "postpone": -1,
                            "postponed": -1,
                            "postpones": -1,
                            "postponing": -1,
                            "poverty": -1,
                            "powerful": 2,
                            "powerless": -2,
                            "praise": 3,
                            "praised": 3,
                            "praises": 3,
                            "praising": 3,
                            "pray": 1,
                            "praying": 1,
                            "prays": 1,
                            "prblm": -2,
                            "prblms": -2,
                            "prepared": 1,
                            "pressure": -1,
                            "pressured": -2,
                            "pretend": -1,
                            "pretending": -1,
                            "pretends": -1,
                            "pretty": 1,
                            "prevent": -1,
                            "prevented": -1,
                            "preventing": -1,
                            "prevents": -1,
                            "prick": -5,
                            "prison": -2,
                            "prisoner": -2,
                            "prisoners": -2,
                            "privileged": 2,
                            "proactive": 2,
                            "problem": -2,
                            "problems": -2,
                            "profiteer": -2,
                            "progress": 2,
                            "prominent": 2,
                            "promise": 1,
                            "promised": 1,
                            "promises": 1,
                            "promote": 1,
                            "promoted": 1,
                            "promotes": 1,
                            "promoting": 1,
                            "propaganda": -2,
                            "prosecute": -1,
                            "prosecuted": -2,
                            "prosecutes": -1,
                            "prosecution": -1,
                            "prospect": 1,
                            "prospects": 1,
                            "prosperous": 3,
                            "protect": 1,
                            "protected": 1,
                            "protects": 1,
                            "protest": -2,
                            "protesters": -2,
                            "protesting": -2,
                            "protests": -2,
                            "proud": 2,
                            "proudly": 2,
                            "provoke": -1,
                            "provoked": -1,
                            "provokes": -1,
                            "provoking": -1,
                            "pseudoscience": -3,
                            "punish": -2,
                            "punished": -2,
                            "punishes": -2,
                            "punitive": -2,
                            "pushy": -1,
                            "puzzled": -2,
                            "quaking": -2,
                            "questionable": -2,
                            "questioned": -1,
                            "questioning": -1,
                            "racism": -3,
                            "racist": -3,
                            "racists": -3,
                            "rage": -2,
                            "rageful": -2,
                            "rainy": -1,
                            "rant": -3,
                            "ranter": -3,
                            "ranters": -3,
                            "rants": -3,
                            "rape": -4,
                            "rapist": -4,
                            "rapture": 2,
                            "raptured": 2,
                            "raptures": 2,
                            "rapturous": 4,
                            "rash": -2,
                            "ratified": 2,
                            "reach": 1,
                            "reached": 1,
                            "reaches": 1,
                            "reaching": 1,
                            "reassure": 1,
                            "reassured": 1,
                            "reassures": 1,
                            "reassuring": 2,
                            "rebellion": -2,
                            "recession": -2,
                            "reckless": -2,
                            "recommend": 2,
                            "recommended": 2,
                            "recommends": 2,
                            "redeemed": 2,
                            "refuse": -2,
                            "refused": -2,
                            "refusing": -2,
                            "regret": -2,
                            "regretful": -2,
                            "regrets": -2,
                            "regretted": -2,
                            "regretting": -2,
                            "reject": -1,
                            "rejected": -1,
                            "rejecting": -1,
                            "rejects": -1,
                            "rejoice": 4,
                            "rejoiced": 4,
                            "rejoices": 4,
                            "rejoicing": 4,
                            "relaxed": 2,
                            "relentless": -1,
                            "reliant": 2,
                            "relieve": 1,
                            "relieved": 2,
                            "relieves": 1,
                            "relieving": 2,
                            "relishing": 2,
                            "remarkable": 2,
                            "remorse": -2,
                            "repulse": -1,
                            "repulsed": -2,
                            "rescue": 2,
                            "rescued": 2,
                            "rescues": 2,
                            "resentful": -2,
                            "resign": -1,
                            "resigned": -1,
                            "resigning": -1,
                            "resigns": -1,
                            "resolute": 2,
                            "resolve": 2,
                            "resolved": 2,
                            "resolves": 2,
                            "resolving": 2,
                            "respected": 2,
                            "responsible": 2,
                            "responsive": 2,
                            "restful": 2,
                            "restless": -2,
                            "restore": 1,
                            "restored": 1,
                            "restores": 1,
                            "restoring": 1,
                            "restrict": -2,
                            "restricted": -2,
                            "restricting": -2,
                            "restriction": -2,
                            "restricts": -2,
                            "retained": -1,
                            "retard": -2,
                            "retarded": -2,
                            "retreat": -1,
                            "revenge": -2,
                            "revengeful": -2,
                            "revered": 2,
                            "revive": 2,
                            "revives": 2,
                            "reward": 2,
                            "rewarded": 2,
                            "rewarding": 2,
                            "rewards": 2,
                            "rich": 2,
                            "ridiculous": -3,
                            "rig": -1,
                            "rigged": -1,
                            "right direction": 3,
                            "rigorous": 3,
                            "rigorously": 3,
                            "riot": -2,
                            "riots": -2,
                            "risk": -2,
                            "risks": -2,
                            "rob": -2,
                            "robber": -2,
                            "robed": -2,
                            "robing": -2,
                            "robs": -2,
                            "robust": 2,
                            "rofl": 4,
                            "roflcopter": 4,
                            "roflmao": 4,
                            "romance": 2,
                            "rotfl": 4,
                            "rotflmfao": 4,
                            "rotflol": 4,
                            "ruin": -2,
                            "ruined": -2,
                            "ruining": -2,
                            "ruins": -2,
                            "sabotage": -2,
                            "sad": -2,
                            "sadden": -2,
                            "saddened": -2,
                            "sadly": -2,
                            "safe": 1,
                            "safely": 1,
                            "safety": 1,
                            "salient": 1,
                            "sappy": -1,
                            "sarcastic": -2,
                            "satisfied": 2,
                            "save": 2,
                            "saved": 2,
                            "scam": -2,
                            "scams": -2,
                            "scandal": -3,
                            "scandalous": -3,
                            "scandals": -3,
                            "scapegoat": -2,
                            "scapegoats": -2,
                            "scare": -2,
                            "scared": -2,
                            "scary": -2,
                            "sceptical": -2,
                            "scold": -2,
                            "scoop": 3,
                            "scorn": -2,
                            "scornful": -2,
                            "scream": -2,
                            "screamed": -2,
                            "screaming": -2,
                            "screams": -2,
                            "screwed": -2,
                            "screwed up": -3,
                            "scumbag": -4,
                            "secure": 2,
                            "secured": 2,
                            "secures": 2,
                            "sedition": -2,
                            "seditious": -2,
                            "seduced": -1,
                            "self-confident": 2,
                            "self-deluded": -2,
                            "selfish": -3,
                            "selfishness": -3,
                            "sentence": -2,
                            "sentenced": -2,
                            "sentences": -2,
                            "sentencing": -2,
                            "serene": 2,
                            "severe": -2,
                            "sexy": 3,
                            "shaky": -2,
                            "shame": -2,
                            "shamed": -2,
                            "shameful": -2,
                            "share": 1,
                            "shared": 1,
                            "shares": 1,
                            "shattered": -2,
                            "shit": -4,
                            "shithead": -4,
                            "shitty": -3,
                            "shock": -2,
                            "shocked": -2,
                            "shocking": -2,
                            "shocks": -2,
                            "shoot": -1,
                            "short-sighted": -2,
                            "short-sightedness": -2,
                            "shortage": -2,
                            "shortages": -2,
                            "shrew": -4,
                            "shy": -1,
                            "sick": -2,
                            "sigh": -2,
                            "significance": 1,
                            "significant": 1,
                            "silencing": -1,
                            "silly": -1,
                            "sincere": 2,
                            "sincerely": 2,
                            "sincerest": 2,
                            "sincerity": 2,
                            "sinful": -3,
                            "singleminded": -2,
                            "skeptic": -2,
                            "skeptical": -2,
                            "skepticism": -2,
                            "skeptics": -2,
                            "slam": -2,
                            "slash": -2,
                            "slashed": -2,
                            "slashes": -2,
                            "slashing": -2,
                            "slavery": -3,
                            "sleeplessness": -2,
                            "slick": 2,
                            "slicker": 2,
                            "slickest": 2,
                            "sluggish": -2,
                            "slut": -5,
                            "smart": 1,
                            "smarter": 2,
                            "smartest": 2,
                            "smear": -2,
                            "smile": 2,
                            "smiled": 2,
                            "smiles": 2,
                            "smiling": 2,
                            "smog": -2,
                            "sneaky": -1,
                            "snub": -2,
                            "snubbed": -2,
                            "snubbing": -2,
                            "snubs": -2,
                            "sobering": 1,
                            "solemn": -1,
                            "solid": 2,
                            "solidarity": 2,
                            "solution": 1,
                            "solutions": 1,
                            "solve": 1,
                            "solved": 1,
                            "solves": 1,
                            "solving": 1,
                            "somber": -2,
                            "son-of-a-bitch": -5,
                            "soothe": 3,
                            "soothed": 3,
                            "soothing": 3,
                            "sophisticated": 2,
                            "sore": -1,
                            "sorrow": -2,
                            "sorrowful": -2,
                            "sorry": -1,
                            "spam": -2,
                            "spammer": -3,
                            "spammers": -3,
                            "spamming": -2,
                            "spark": 1,
                            "sparkle": 3,
                            "sparkles": 3,
                            "sparkling": 3,
                            "speculative": -2,
                            "spirit": 1,
                            "spirited": 2,
                            "spiritless": -2,
                            "spiteful": -2,
                            "splendid": 3,
                            "sprightly": 2,
                            "squelched": -1,
                            "stab": -2,
                            "stabbed": -2,
                            "stable": 2,
                            "stabs": -2,
                            "stall": -2,
                            "stalled": -2,
                            "stalling": -2,
                            "stamina": 2,
                            "stampede": -2,
                            "startled": -2,
                            "starve": -2,
                            "starved": -2,
                            "starves": -2,
                            "starving": -2,
                            "steadfast": 2,
                            "steal": -2,
                            "steals": -2,
                            "stereotype": -2,
                            "stereotyped": -2,
                            "stifled": -1,
                            "stimulate": 1,
                            "stimulated": 1,
                            "stimulates": 1,
                            "stimulating": 2,
                            "stingy": -2,
                            "stolen": -2,
                            "stop": -1,
                            "stopped": -1,
                            "stopping": -1,
                            "stops": -1,
                            "stout": 2,
                            "straight": 1,
                            "strange": -1,
                            "strangely": -1,
                            "strangled": -2,
                            "strength": 2,
                            "strengthen": 2,
                            "strengthened": 2,
                            "strengthening": 2,
                            "strengthens": 2,
                            "stressed": -2,
                            "stressor": -2,
                            "stressors": -2,
                            "stricken": -2,
                            "strike": -1,
                            "strikers": -2,
                            "strikes": -1,
                            "strong": 2,
                            "stronger": 2,
                            "strongest": 2,
                            "struck": -1,
                            "struggle": -2,
                            "struggled": -2,
                            "struggles": -2,
                            "struggling": -2,
                            "stubborn": -2,
                            "stuck": -2,
                            "stunned": -2,
                            "stunning": 4,
                            "stupid": -2,
                            "stupidly": -2,
                            "suave": 2,
                            "substantial": 1,
                            "substantially": 1,
                            "subversive": -2,
                            "success": 2,
                            "successful": 3,
                            "suck": -3,
                            "sucks": -3,
                            "suffer": -2,
                            "suffering": -2,
                            "suffers": -2,
                            "suicidal": -2,
                            "suicide": -2,
                            "suing": -2,
                            "sulking": -2,
                            "sulky": -2,
                            "sullen": -2,
                            "sunshine": 2,
                            "super": 3,
                            "superb": 5,
                            "superior": 2,
                            "support": 2,
                            "supported": 2,
                            "supporter": 1,
                            "supporters": 1,
                            "supporting": 1,
                            "supportive": 2,
                            "supports": 2,
                            "survived": 2,
                            "surviving": 2,
                            "survivor": 2,
                            "suspect": -1,
                            "suspected": -1,
                            "suspecting": -1,
                            "suspects": -1,
                            "suspend": -1,
                            "suspended": -1,
                            "suspicious": -2,
                            "swear": -2,
                            "swearing": -2,
                            "swears": -2,
                            "sweet": 2,
                            "swift": 2,
                            "swiftly": 2,
                            "swindle": -3,
                            "swindles": -3,
                            "swindling": -3,
                            "sympathetic": 2,
                            "sympathy": 2,
                            "tard": -2,
                            "tears": -2,
                            "tender": 2,
                            "tense": -2,
                            "tension": -1,
                            "terrible": -3,
                            "terribly": -3,
                            "terrific": 4,
                            "terrified": -3,
                            "terror": -3,
                            "terrorize": -3,
                            "terrorized": -3,
                            "terrorizes": -3,
                            "thank": 2,
                            "thankful": 2,
                            "thanks": 2,
                            "thorny": -2,
                            "thoughtful": 2,
                            "thoughtless": -2,
                            "threat": -2,
                            "threaten": -2,
                            "threatened": -2,
                            "threatening": -2,
                            "threatens": -2,
                            "threats": -2,
                            "thrilled": 5,
                            "thwart": -2,
                            "thwarted": -2,
                            "thwarting": -2,
                            "thwarts": -2,
                            "timid": -2,
                            "timorous": -2,
                            "tired": -2,
                            "tits": -2,
                            "tolerant": 2,
                            "toothless": -2,
                            "top": 2,
                            "tops": 2,
                            "torn": -2,
                            "torture": -4,
                            "tortured": -4,
                            "tortures": -4,
                            "torturing": -4,
                            "totalitarian": -2,
                            "totalitarianism": -2,
                            "tout": -2,
                            "touted": -2,
                            "touting": -2,
                            "touts": -2,
                            "tragedy": -2,
                            "tragic": -2,
                            "tranquil": 2,
                            "trap": -1,
                            "trapped": -2,
                            "trauma": -3,
                            "traumatic": -3,
                            "travesty": -2,
                            "treason": -3,
                            "treasonous": -3,
                            "treasure": 2,
                            "treasures": 2,
                            "trembling": -2,
                            "tremulous": -2,
                            "tricked": -2,
                            "trickery": -2,
                            "triumph": 4,
                            "triumphant": 4,
                            "trouble": -2,
                            "troubled": -2,
                            "troubles": -2,
                            "TRUE": 2,
                            "trust": 1,
                            "trusted": 2,
                            "tumor": -2,
                            "twat": -5,
                            "ugly": -3,
                            "unacceptable": -2,
                            "unappreciated": -2,
                            "unapproved": -2,
                            "unaware": -2,
                            "unbelievable": -1,
                            "unbelieving": -1,
                            "unbiased": 2,
                            "uncertain": -1,
                            "unclear": -1,
                            "uncomfortable": -2,
                            "unconcerned": -2,
                            "unconfirmed": -1,
                            "unconvinced": -1,
                            "uncredited": -1,
                            "undecided": -1,
                            "underestimate": -1,
                            "underestimated": -1,
                            "underestimates": -1,
                            "underestimating": -1,
                            "undermine": -2,
                            "undermined": -2,
                            "undermines": -2,
                            "undermining": -2,
                            "undeserving": -2,
                            "undesirable": -2,
                            "uneasy": -2,
                            "unemployment": -2,
                            "unequal": -1,
                            "unequaled": 2,
                            "unethical": -2,
                            "unfair": -2,
                            "unfocused": -2,
                            "unfulfilled": -2,
                            "unhappy": -2,
                            "unhealthy": -2,
                            "unified": 1,
                            "unimpressed": -2,
                            "unintelligent": -2,
                            "united": 1,
                            "unjust": -2,
                            "unlovable": -2,
                            "unloved": -2,
                            "unmatched": 1,
                            "unmotivated": -2,
                            "unprofessional": -2,
                            "unresearched": -2,
                            "unsatisfied": -2,
                            "unsecured": -2,
                            "unsettled": -1,
                            "unsophisticated": -2,
                            "unstable": -2,
                            "unstoppable": 2,
                            "unsupported": -2,
                            "unsure": -1,
                            "untarnished": 2,
                            "unwanted": -2,
                            "unworthy": -2,
                            "upset": -2,
                            "upsets": -2,
                            "upsetting": -2,
                            "uptight": -2,
                            "urgent": -1,
                            "useful": 2,
                            "usefulness": 2,
                            "useless": -2,
                            "uselessness": -2,
                            "vague": -2,
                            "validate": 1,
                            "validated": 1,
                            "validates": 1,
                            "validating": 1,
                            "verdict": -1,
                            "verdicts": -1,
                            "vested": 1,
                            "vexation": -2,
                            "vexing": -2,
                            "vibrant": 3,
                            "vicious": -2,
                            "victim": -3,
                            "victimize": -3,
                            "victimized": -3,
                            "victimizes": -3,
                            "victimizing": -3,
                            "victims": -3,
                            "vigilant": 3,
                            "vile": -3,
                            "vindicate": 2,
                            "vindicated": 2,
                            "vindicates": 2,
                            "vindicating": 2,
                            "violate": -2,
                            "violated": -2,
                            "violates": -2,
                            "violating": -2,
                            "violence": -3,
                            "violent": -3,
                            "virtuous": 2,
                            "virulent": -2,
                            "vision": 1,
                            "visionary": 3,
                            "visioning": 1,
                            "visions": 1,
                            "vitality": 3,
                            "vitamin": 1,
                            "vitriolic": -3,
                            "vivacious": 3,
                            "vociferous": -1,
                            "vulnerability": -2,
                            "vulnerable": -2,
                            "walkout": -2,
                            "walkouts": -2,
                            "wanker": -3,
                            "want": 1,
                            "war": -2,
                            "warfare": -2,
                            "warm": 1,
                            "warmth": 2,
                            "warn": -2,
                            "warned": -2,
                            "warning": -3,
                            "warnings": -3,
                            "warns": -2,
                            "waste": -1,
                            "wasted": -2,
                            "wasting": -2,
                            "wavering": -1,
                            "weak": -2,
                            "weakness": -2,
                            "wealth": 3,
                            "wealthy": 2,
                            "weary": -2,
                            "weep": -2,
                            "weeping": -2,
                            "weird": -2,
                            "welcome": 2,
                            "welcomed": 2,
                            "welcomes": 2,
                            "whimsical": 1,
                            "whitewash": -3,
                            "whore": -4,
                            "wicked": -2,
                            "widowed": -1,
                            "willingness": 2,
                            "win": 4,
                            "winner": 4,
                            "winning": 4,
                            "wins": 4,
                            "winwin": 3,
                            "wish": 1,
                            "wishes": 1,
                            "wishing": 1,
                            "withdrawal": -3,
                            "woebegone": -2,
                            "woeful": -3,
                            "won": 3,
                            "wonderful": 4,
                            "woo": 3,
                            "woohoo": 3,
                            "wooo": 4,
                            "woow": 4,
                            "worn": -1,
                            "worried": -3,
                            "worry": -3,
                            "worrying": -3,
                            "worse": -3,
                            "worsen": -3,
                            "worsened": -3,
                            "worsening": -3,
                            "worsens": -3,
                            "worshiped": 3,
                            "worst": -3,
                            "worth": 2,
                            "worthless": -2,
                            "worthy": 2,
                            "wow": 4,
                            "wowow": 4,
                            "wowww": 4,
                            "wrathful": -3,
                            "wreck": -2,
                            "wrong": -2,
                            "wronged": -2,
                            "wtf": -4,
                            "yeah": 1,
                            "yearning": 1,
                            "yeees": 2,
                            "yes": 1,
                            "youthful": 2,
                            "yucky": -2,
                            "yummy": 3,
                            "zealot": -2,
                            "zealots": -2,
                            "zealous": 2                                        
            }
            
            for k, v in dic.items():
                self.sentiment_dic[k] = v
                if k != "no" and k != "fun":
                    self.sentiment_dic["no " + k] = -1*v
                if k != "not" and k != "good" and k != "stand":
                    self.sentiment_dic["not " + k] = -1*v
        else: 
            self.sentiment_dic = sentiment_dic
        self.sentiment_word_dic = deepcopy(self.sentiment_dic)
        i = 0
        for term in self.sentiment_word_dic:
            self.sentiment_word_dic[term] = i
            i += 1
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
        s = s.replace("wouldent", "would not")
        s = s.replace("n't", " not")
        punc = string.punctuation
        for i in range(len(punc)):
            s = s.replace(punc[i], ' ')
        for i in range(10):
            j = str(i)
            s = s.replace(j, " ")
        s = Sentiment.reduce_lengthening(s)
        return s
    
    def reduce_lengthening(text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)

    # Customized analyzer for Sentiment Analysis
    def analyzer(self, s):
        # Synonym List - Map Keys to Values
        syns = { \
                  'wont':'would not', \
                  'cant':'can not', 'cannot':'can not', \
                  'couldnt':'could not', \
                  'shouldnt':'should not', \
                  'wouldnt':'would not'}
        
        # Preprocess String s
        s = Sentiment.preprocessor(s)
    
        # Tokenize 
        tokens = word_tokenize(s)
        #tokens = [word.replace(',','') for word in tokens ]
        tokens = [word for word in tokens if ('*' not in word) and \
                  ("''" != word) and ("``" != word) and \
                  (word!='description') and (word !='dtype') \
                  and (word != 'object') and (word!="'s")]
        
        # Map synonyms
        for i in range(len(tokens)):
            #if checker:
            #    tokens[i] = spell.correction(tokens[i])
            if tokens[i] in syns:
                tokens[i] = syns[tokens[i]]
                
        # Remove stop words
        punctuation = list(string.punctuation)+['..', '...']
        pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
        others   = ["'d", "co", "ed", "put", "say", "get", "can", "become",\
                    "los", "sta", "la", "use", "iii", "else", "could", \
                    "would", "come", "take"]
        stop = punctuation + pronouns + others 
        filtered_terms = [word for word in tokens if (word not in stop) and \
                      (len(word)>1) and (not word.replace('.','',1).isnumeric()) \
                      and (not word.replace("'",'',2).isnumeric())]
        for word in filtered_terms:
            word = Sentiment.reduce_lengthening(word)
        return filtered_terms
    
    def scores(self, tf, terms):
        # tf is an scipy CSR (compressed sparse matrix)
        min_sentiment = +5
        max_sentiment = -5
        avg_sentiment =  0  
        self.min_list, self.max_list = [],[]
        n_reviews = tf.shape[0]
        sentiment_score = [0]*n_reviews
        for i in range(n_reviews):
            # Iterate over the terms with nonzero scores
            n_sw = 0
            term_list = tf[i].nonzero()[1]
            if len(term_list)>0:
                for t in np.nditer(term_list):
                    score = self.sentiment_dic.get(terms[t])
                    if score != None:
                        #sentiment_score[i] += score * tf[i,t]
                        #n_sw += tf[i,t]
                        sentiment_score[i] += score * tf[i,t]
                        n_sw += tf[i,t]
            if n_sw>0:
                sentiment_score[i] = sentiment_score[i]/n_sw
            if sentiment_score[i]==max_sentiment and n_sw>=self.n_terms:
                self.max_list.append(i)
            if sentiment_score[i]>max_sentiment and n_sw>=self.n_terms:
                max_sentiment=sentiment_score[i]
                self.max_list = [i]
            
            if sentiment_score[i]==min_sentiment and n_sw>=self.n_terms:
                self.min_list.append(i)
            if sentiment_score[i]<min_sentiment and n_sw>=self.n_terms:
                min_sentiment=sentiment_score[i]
                self.min_list = [i]
            avg_sentiment += sentiment_score[i]
        avg_sentiment = avg_sentiment/n_reviews
        print("\nCorpus Average Sentiment: ", avg_sentiment)
        print("\nMost Negative Reviews with", \
              self.n_terms, "or more Sentiment Words:")
        for i in range(len(self.min_list)):
            print("{:<s}{:<d}{:<s}{:<5.2f}".format("    Review ", \
                  self.min_list[i], " Sentiment is ", min_sentiment))
            
        print("\nMost Positive Reviews with", \
              self.n_terms, "or more Sentiment Words:")
        for i in range(len(self.max_list)):
            print("{:<s}{:<d}{:<s}{:<5.2f}".format("    Review ", \
                  self.max_list[i],  " Sentiment is ", max_sentiment))
            
        return sentiment_score
    
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
    
class calculate:
    # Function for calculating loss and confusion matrix
    def binary_loss(y, y_predict, fp_cost, fn_cost, display=True):
        loss     = [0, 0]       #False Neg Cost, False Pos Cost
        conf_mat = [0, 0, 0, 0] #tn, fp, fn, tp
        for j in range(len(y)):
            if y[j]==0:
                if y_predict[j]==0:
                    conf_mat[0] += 1 #True Negative
                else:
                    conf_mat[1] += 1 #False Positive
                    loss[1] += fp_cost[j]
            else:
                if y_predict[j]==1:
                    conf_mat[3] += 1 #True Positive
                else:
                    conf_mat[2] += 1 #False Negative
                    loss[0] += fn_cost[j]
        if display:
            fn_loss = loss[0]
            fp_loss = loss[1]
            total_loss = fn_loss + fp_loss
            misc    = conf_mat[1] + conf_mat[2]
            misc    = misc/len(y)
            print("{:.<23s}{:10.4f}".format("Misclassification Rate", misc))
            print("{:.<23s}{:10.0f}".format("False Negative Loss", fn_loss))
            print("{:.<23s}{:10.0f}".format("False Positive Loss", fp_loss))
            print("{:.<23s}{:10.0f}".format("Total Loss", total_loss))
        return loss, conf_mat