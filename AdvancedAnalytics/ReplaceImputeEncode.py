"""

@author: Edward R Jones
@version 1.14
@copyright 2020 - Edward R Jones, all rights reserved.
"""
#from DT import DT
import sys
import warnings
import numpy  as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute  import SimpleImputer
from copy import deepcopy #Used to create sentiment word dictionary

import pickle
from enum import Enum
#Class DT - DataType This is setup to provide a clean
#notation for data maps used by ReplaceImputeEncode
class DT(Enum):
    # @attributes: characters recognized in RIE code
    Interval = 'I' #Expected values (lowest value, highest value)
    Binary   = 'B' #Expected values (class0, class1)
    Nominal  = 'N' #Expected values (class0, class1, ... classk)
    Ordinal  = 'O' #Expected values ordered classes (class0, class1, ...)
    String   = 'S' #Expected values ("")
    ID       = 'Z' #Expected values ("")
    Label    = 'L' #Expected values ("")
    Text     = 'T' #Expected values ("")
    Ignore   = 'Z' #Expected values ("")
    interval = 'I' #Allow lower case
    binary   = 'B' #Allow lower case
    nominal  = 'N' #Allow lower case
    ordinal  = 'O' #Allow lower case
    string   = 'S' #Allow lower case
    id       = 'Z' #Expected values ("")
    label    = 'L' #Expected values ("")
    text     = 'T' #Allow lower case
    ignore   = 'Z' #Allow lower case

    # @methods
    def getDataTypes():
        dtype = [
                DT.Interval,
                DT.Binary, 
                DT.Nominal, 
                DT.Ordinal,
                DT.ID, 
                DT.Label,
                DT.Text , 
                DT.String,
                DT.Ignore 
                ]
        return dtype #Returns data type list
    
    def convertDataType(atype):
        if   atype==DT.Interval:
             ctype ='DT.Interval'
        elif atype==DT.Binary:
             ctype ='DT.Binary'
        elif atype==DT.Nominal:
             ctype ='DT.Nominal'
        elif atype==DT.Ordinal:
             ctype ='DT.Ordinal'
        elif atype==DT.String:
             ctype ='DT.String'
        elif atype==DT.ID:
             ctype ='DT.ID'
        elif atype==DT.Label:
             ctype ='DT.Label'
        elif atype==DT.Text:
             ctype ='DT.Text'
        else:ctype ='DT.Ignore'
        return ctype
    
"""
class ReplaceImputeEncode

@parameters:
    *** __init__() ***
    data_map - The metadata dictionary.  
                
    nominal_encoding - Can be 'one-hot', 'SAS' or default None.
    
    interval_scale   - Can be 'std', 'robust' or default None.
    
    no_impute        - default None or list of attributes to exclude from 
                       imputation
    
    drop             - True or default False.  True drops the last nominal 
                       encoded column.  False keeps all nominal encoded 
                       columns.
    
    display          - True or default False.  True displays the number of 
                       missing and outliers found in the data.
                        
    *** fit_transform () ***
    df  - a pandas DataFrame containing the data description
                        by the metadata found in data_map (required)
    data_map - See above description.
    
@Cautions:
    The incoming dataframe, df, and the data_map are deep copied to
    ensure that changes to the dataframe are only held within the class
    object self.copy_df.  The attributes_map is deep copied into
    self.features_map.  All binary and nominal values are encoded to 
    numeric values.
    
    The method draft_data_map returns a of the data_map based upon
    the data.  This must be examined to ensure the data types and 
    allowed values are correct.  This behavior is controled by k_min
    and k_max.  See API and examples for details.
"""

class ReplaceImputeEncode(object):
    def __init__(self, data_map=None, binary_encoding=None,
               nominal_encoding=None, interval_scale=None, no_impute=None, 
               drop=False, display=False): 
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
        if binary_encoding=='None' or binary_encoding=='none':
            self.binary_encoding = None
        else:
            self.binary_encoding = binary_encoding
        #nominal_encoding can be 'SAS' or 'one-hot'
        if binary_encoding != 'SAS' and binary_encoding != 'one-hot' \
            and binary_encoding != None:
            raise ValueError("***Call to ReplaceImputeEncode invalid. "+
                 "***   binary_encoding="+binary_encoding+" is invalid."+
                 "***   must use None, 'one-hot' or 'SAS'")
            sys.exit()
        if nominal_encoding=='None' or nominal_encoding=='none':
            self.nominal_encoding = None
        else:
            self.nominal_encoding = nominal_encoding
        #nominal_encoding can be 'SAS' or 'one-hot'
        if nominal_encoding != 'SAS' and nominal_encoding != 'one-hot' \
            and nominal_encoding != None:
            raise ValueError("***Call to ReplaceImputeEncode invalid. "+
                 "***   nominal_encoding="+nominal_encoding+" is invalid."+
                 "***   must use None, 'one-hot' or 'SAS'")
            sys.exit()
        if interval_scale != 'std' and interval_scale != 'robust' \
            and interval_scale != None:
            raise ValueError("***Call to ReplaceImputeEncode invalid. "+
                     "***   interval_scale="+interval_scale+" is invalid."+
                     "***   must use None, 'std' or 'robust'")
            sys.exit()
        if data_map==None:
            print("Attributes Map is required.")
            print("Please pass map using data_map attribute.")
            print("If one is not available, try creating one using "+
                  "call to draft_features_map(df)")
            return
        if type(data_map)==str:
            try:
                self.features_map = self.load_data_map(data_map)
            except:
                raise ValueError("Unable to load data map:", data_map)
                sys.exit()
        elif type(data_map)==dict:
            self.features_map = data_map
        else:
            raise ValueError("Supplied Data Map not Dictionary or File")
            sys.exit()
        self.interval_attributes = []
        self.nominal_attributes  = []
        self.binary_attributes   = []
        self.onehot_attributes   = []
        self.onehot_cats         = []
        self.hot_drop_list       = []
        self.missing_counts      = {}
        self.outlier_counts      = {}
        for feature,v in self.features_map.items():
            # Initialize data map missing and outlier counters to zero
            self.missing_counts[feature] = 0
            self.outlier_counts[feature] = 0

            if v[0] not in DT.getDataTypes():
                raise TypeError(
                  "\n***Data Map in call to ReplaceImputeEncode invalid.\n"+
                  "***Data Type for '"+ feature + "' is not recognized. "+
                  "\n***Valid types are: DT.Interval, DT.Binary, DT.Nominal, "+
                  "DT.Text, DT.String, DT.ID, or DT.Ignore")
            if v[0]==DT.Interval:
                self.interval_attributes.append(feature)
            else:
                if v[0]==DT.Binary:
                    self.binary_attributes.append(feature)
                else:
                    if v[0]!=DT.Binary and v[0]!=DT.Nominal: 
                        # Ignore, don't touch this attribute
                        continue
                    # Attribute must be Nominal
                    self.nominal_attributes.append(feature)
                    # Setup column names for Nominal encoding
                    n_cat = len(v[1])
                    self.onehot_cats.append(list(v[1]))
                    data_type = type(v[1][n_cat-1])
                    if self.drop == True:
                        n_cat -= 1
                    for i in range(n_cat):
                        if type(v[1][i]) != data_type:
                            raise TypeError(
                              "\n***Classes invalid for--> '"+feature+"'"+
                              "\n***Must be all numeric or strings, not both.")
                        if type(v[1][i])==int:
                            my_str = feature+str(v[1][i])
                        else:
                            my_str = feature+("%i" %i)+":"+str(v[1][i])[0:10]
                        self.onehot_attributes.append(my_str)

        self.n_interval = len(self.interval_attributes)
        self.n_binary   = len(self.binary_attributes)
        self.n_nominal  = len(self.nominal_attributes)
        self.n_onehot   = len(self.onehot_attributes)
        self.cat        = self.n_binary + self.n_nominal
        if nominal_encoding=='SAS' and drop==False and self.n_nominal>0:
            raise ValueError("***Call to ReplaceImputeEncode invalid. "+
                  "***nominal_encoding='SAS' requested with drop=False "+
                  "***'SAS' encoding requires drop=True")
            sys.exit()
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
            raise ValueError("  Call to ReplaceImputeEncode missing required"+
              " Data Map.\n Use function draft_data_map to draft a map.")
            sys.exit()
        if type(self.features_map)==dict:
            pass
        elif type(data_map)==str:
            try:
                self.features_map = self.load_data_map(data_map)
            except:
                raise ValueError("Unable to load data map:", data_map)
                sys.exit()
        elif type(data_map)==dict:
            self.features_map = data_map
        else:
            raise ValueError("Supplied Data Map not Dictionary or File")
            sys.exit()
            
        self.interval_attributes = []
        self.nominal_attributes  = []
        self.binary_attributes   = []
        self.onehot_attributes   = []
        self.onehot_cats         = []
        self.hot_drop_list       = []
        for feature,v in self.features_map.items():
            if v[0] not in DT.getDataTypes():
                raise TypeError(
                  "\n***Data Map in call to ReplaceImputeEncode invalid.\n"+
                  "***Data Type for '"+ feature + "' is not recognized. "+
                  "\n***Valid types are: DT.Interval, DT.Binary, DT.Nominal, "+
                  "DT.Text, DT.String, DT.ID, or DT.Ignore")
            if v[0]==DT.Interval:
                self.interval_attributes.append(feature)
            else:
                if v[0]==DT.Binary:
                    self.binary_attributes.append(feature)
                else:
                    if v[0]==DT.Nominal:
                        self.nominal_attributes.append(feature)
                        self.onehot_cats.append(list(v[1]))
                        for i in range(len(v[1])):
                            if type(v[1][i])==int:
                                my_str = feature+str(v[1][i])
                            else:
                                my_str = feature+("%i" %i)+":"+ \
                                                str(v[1][i])[0:10]
                            self.onehot_attributes.append(my_str)
                        if self.drop==True:
                            self.hot_drop_list.append(my_str)
                    else:
                        if v[0] in DT.getDataTypes():
                            continue
                        else:
                        # Data Map Invalid
                            raise TypeError( 
                  "***Data Map in call to ReplaceImputeEncode invalid.\n"+
                  "***Data Type for '"+ feature + "' invalid")
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
                warnings.warn(feature+":has more than 50% missing." +
                              "Recommend setting Data Type set to DT.Ignore.")
        # Initialize number missing in attribute_map
        for feature,v in self.features_map.items():
            try:
                self.missing_counts[feature] = self.initial_missing[feature]
            except:
                raise RuntimeError(feature + " is not found in Data_Map. ")
                sys.exit()

        # Scan for outliers among interval attributes
        nan_map = df.isnull()
        for index in df.iterrows():
            i = index[0]
        # Check for outliers in interval attributes
            for feature, v in self.features_map.items():
                if nan_map.loc[i,feature]==True:
                    continue
                if v[0]==DT.Interval: # Interval Attribute
                    if type(v[1]) != tuple or len(v[1]) != 2:
                       raise ValueError("\n" +\
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
                    if v[0]!=DT.Binary and v[0]!=DT.Nominal: 
                        # don't touch this attribute
                        continue
                    # Categorical Attribute
                    in_cat = False
                    for cat in v[1]:
                        if df.loc[i,feature]==cat:
                            in_cat=True
                            continue
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
            
    def draft_data_map(self, df, max_n=10, max_s=30, display_map=True,
                       out=None, replace=False):
        feature_names = np.array(df.columns.values)
        draft_features_map = {}
        print("\nGenerating DATA_MAP for use in ReplaceImputeEncode.")
        print("String attributes with fewer than", max_s, 
                  "unique values are labeled as Binary or Nominal; "+
                  "otherwise Text or String.")
        print("Numerical attributes with fewer than", max_n,
                  "unique values are considered Binary or Nominal;"+
                  " otherwise Interval")
        
        for feature in feature_names:
            n = df[feature].value_counts()
            if type(df[feature].iloc[0]) != str:
                min_ = round(df[feature].min()-0.5,4)
                max_ = round(df[feature].max()+0.5,4)
            if type(df[feature].iloc[0]) !=str:
                #Numerical attribute
                if len(n) < max_n:
                    # Numerical Attribute is Binary or Nominal
                    a   = df[feature].unique()
                    # Look for string in a
                    j = 0
                    for i in range(len(a)):
                        if type(a[i])==str:
                            j += 1
                    if j>0:
                        print("WARNING: ", feature, "contains both numbers "+
                              "and strings. Dropping from draft data map.")
                        break
                    a.sort()
                    categories = tuple(a)
                    if len(a) == 2:
                        draft_features_map[feature]=[DT.Binary, 
                                          categories]
                    else:
                        draft_features_map[feature]=[DT.Nominal,
                                          categories]
                else:
                    # Attribute is Interval
                    draft_features_map[feature]=[DT.Interval,
                                      (min_, max_)]

            else:
                # String Attribute is Binary, Nominal or Text or String
                if len(n) < max_s: 
                    # String Attribute is Binary or Nominal
                    a = df[feature].unique()
                    # Look for nan in a
                    no_nan = False
                    while no_nan == False:
                        j = -1
                        for i in range(len(a)):
                            if type(a[i]) != str:
                                j = i
                        if j>0:
                            a = np.delete(a,j)
                        else:
                            no_nan=True
                    a.sort()
                    categories = tuple(a)
                    if len(a) == 2:
                        draft_features_map[feature]=[DT.Binary, 
                                          categories]
                    else:
                        draft_features_map[feature]=[DT.Nominal,
                                          categories]
                else:
                    k = df[feature].apply(len).median()
                    if k>100:
                        # Set attribute to text field
                        draft_features_map[feature]=[DT.Text,("")]
                    else:
                        draft_features_map[feature]=[DT.String,("")]
        if display_map:
            # print the features map
            print("************* DRAFT DATA MAP **************\n")
            print("data_map = {")
            for feature,v in draft_features_map.items():
                w = DT.convertDataType(v[0])
                s = "\t["
                if len(feature)<5:
                    s = "\t\t["
                print("\t'"+feature+"':",s,str(w),",",v[1],"],")
            print("\n}")
        if replace==True:
            # Use this draft map for RIE processing
            self.features_map = draft_features_map
            print("Using Draft Data Map for ReplaceImputeEncode.\n"+
                          "Review Draft for Data Type Accuracy.")
        if out != None:
            #Save this draft map as a pickle file <out>
            self.save_data_map(draft_features_map, out)
        print(draft_features_map)
        return draft_features_map
    
    def update_feature(self, feature, datatype, dataval):
        if type(feature)!=str:
            raise ValueError("feature name not string")
            sys.exit()
        if datatype!=DT.Interval and datatype!=DT.Binary and \
           datatype!=DT.Nominal and datatype!=DT.Text and \
           datatype!=DT.Ignore and datatype!=DT.String:
            raise ValueError("Data Type Value Invalid")
            sys.exit()
        if type(self.features_map)!=dict:
            self.features_map = {}
        self.features_map[feature] = [datatype, dataval]
    
    def save_data_map(self, data_map, fname):
        if type(data_map)!=dict:
            raise RuntimeError("Data Map invalid")
            sys.exit()
        try:
            with open(fname, 'wb') as f:
                pickle.dump(data_map, f, 
                            pickle.DEFAULT_PROTOCOL)
            print("Data Map Saved to Pickle File: ", fname)
        except:
            warnings.warn("Cannot save data map into file: ")
    
    def get_data_map(self): 
        # Returns current data map
        return self.features_map
        
    def load_data_map(self, fname):
        try:
            with open(fname, 'rb') as f:
                data_map = pickle.load(f)
        except:
            raise ValueError("Unable to load data map from:", fname)
            sys.exit()
        if type(data_map)!=dict:
                raise ValueError("Unable to load data map from:", fname)
                sys.exit()
        return data_map
            
    def display_data_map(self):
        # Display Data Map Dictionary
        try:
            if self.features_map==None:
                raise RuntimeError("Data Map Does not Exist")
        except:
            raise RuntimeError("Data Map Does not Exist")
        # print the features map
        print("************* CURRENT DATA MAP **************\n")
        print("data_map = {")
        for feature,v in self.features_map.items():
            w = DT.convertDataType(v[0])
            c = ','
            if w=='DT.Ignore' or w=='DT.Binary':
                c = '  ,'
            elif w=='DT.Nominal':
                c = ' ,'
            elif w=='DT.Text' or w=='DT.String':
                c = '    ,'
            s = "\t["
            if len(feature)<5:
                s = "\t\t["
            print("\t'"+feature+"':",s,str(w), c, v[1],"],")
        print("\n}")
    
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
        # Put the binary data from the dataframe into a numpy array
        #cat_df = df[self.binary_attributes]
        cat_df = pd.DataFrame(columns=self.binary_attributes)
        for feature in self.binary_attributes:
            #cat_df[feature]= self.df_copy[feature].astype('category').cat.codes
            cat_df[feature]= self.df_copy[feature].astype('category')
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
        # Put the nominal data from the dataframe into a numpy array
        cat_df  = pd.DataFrame(columns=self.nominal_attributes)
        for feature in self.nominal_attributes:
            #self.cat_df[feature]= self.df_copy[feature].astype('category').cat.codes
            cat_df[feature]= self.df_copy[feature].astype('category')
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
        if self.n_binary == 0 or self.binary_encoding == None:
            return
        if self.binary_encoding == 'SAS':
            low = -1
        else:  # One-hot encoding
            low = 0
        for j in range(self.n_binary):
            k = self.imputed_binary_data[0:,j].argmin()
            smallest = self.imputed_binary_data[k,j]
            for i in range(self.n_obs):
                if self.imputed_binary_data[i,j] == smallest:
                    self.imputed_binary_data[i,j] = low
                else:
                    self.imputed_binary_data[i,j] = 1
   
    def encode_nominal(self):
        if (self.n_nominal==0 or self.nominal_encoding==None):
            return
        # Create an instance of the OneHotEncoder & Selecting Attributes
        # Attributes must all be non-negative integers
        # Missing values may show up as -1 values, which will cause an error
        onehot = preprocessing.OneHotEncoder(categories=self.onehot_cats)
        self.hot_array = \
                onehot.fit_transform(self.imputed_nominal_data).toarray()
        n_features = []
        nominal_categories = 0
        for i in range(self.n_nominal):
            feature = self.nominal_attributes[i]
            v = self.features_map[feature]
            n_features.append(len(v[1]))
            nominal_categories += len(v[1])
        if nominal_categories < self.hot_array.shape[1]:
            raise RuntimeError('  Call to ReplaceImputeEncode Invalid '+ \
               '  Number of one-hot columns is', self.hot_array.shape[1], \
               'but nominal categories is ', nominal_categories, \
               '  Data contains more nominal attributes than '+ \
               'found in the data_map.')
            sys.exit()
            
        # SAS Encoding subtracts the last one-hot vector from the others, 
        # for each nominal attribute.
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
                for i in range(self.n_nominal): #WEIRD! n_nominal=0?
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
        #Check for constant data columns
        feature_names = np.array(self.encoded_data_df.columns.values)
        for feature in feature_names:
            if feature in self.interval_attributes:
                self.encoded_data_df[feature] = \
                    self.encoded_data_df[feature].astype('float64')
            elif feature in self.onehot_attributes:
                self.encoded_data_df[feature] = \
                    self.encoded_data_df[feature].astype('int')
            elif feature in self.binary_attributes:
                if self.binary_encoding != None:
                    self.encoded_data_df[feature] = \
                        self.encoded_data_df[feature].astype('int')
                else:
                    self.encoded_data_df[feature] = \
                        self.encoded_data_df[feature].astype(\
                                        self.df_copy[feature].dtype)
            else:
                self.encoded_data_df[feature] = \
                    self.encoded_data_df[feature].astype('O')
            n = self.encoded_data_df[feature].value_counts()
            if len(n)==1:
                print("WARNING:  Data for ", feature, " is constant.")
        return self.encoded_data_df
        
    def fit_transform(self, df, data_map=None):
        self.fit(df, data_map)
        self.transform()
        return self.encoded_data_df
