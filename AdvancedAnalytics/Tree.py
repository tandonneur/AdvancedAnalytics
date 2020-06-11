
"""
@author: Edward R Jones
@version 1.16
@copyright 2020 - Edward R Jones, all rights reserved.
"""

import sys
import numpy  as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
               
class tree_regressor(object):
    
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
            print("NX=", nx)
            print("col length ", len(col))
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
               
class tree_classifier(object):
    
    def display_importance(dt, col, top='all', plot=False):
        nx = dt.n_features_
        if nx != len(col):
            print("NX=", nx)
            print("col length ", len(col))
            raise RuntimeError("  Call to display_importance invalid\n"+
                  "  Number of feature labels (col) not equal to the " +
                  "number of features in the decision tree.")
            sys.exit()
        if type(top) != int and type(top) != str:
            raise RuntimeError("   Call to display_importance invalid\n"+
                  "   Value of top is invalid.  Must be set to 'all' or"+
                  " an integer less than the number of columns in X.")
            sys.exit()
        if type(top) == str and top != 'all':
            raise RuntimeError("   Call to display_importance invalid\n"+
                  "   Value of top is invalid.  Must be set to 'all' or"+
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
        
    def display_metrics(dt, X, y):
        if len(dt.classes_) == 2:
            numpy_y = np.ravel(y)
            if type(numpy_y[0])==str:
                classes_ = dt.classes_
            else:
                classes_ = [str(int(dt.classes_[0])), str(int(dt.classes_[1]))]
            z = np.zeros(len(y))
            predictions = dt.predict(X) # get binary class predictions
            conf_mat = confusion_matrix(y_true=y, y_pred=predictions)
            tmisc = conf_mat[0][1]+conf_mat[1][0]
            misc = 100*(tmisc)/(len(y))
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
            print("{:.<27s}{:10.4f}".format('Mean Absolute Error', \
                          mean_absolute_error(z,probability[:, 1])))
            print("{:.<27s}{:10.4f}".format('Avg Squared Error', \
                          mean_squared_error(z,probability[:, 1])))
            acc = accuracy_score(y, predictions)
            print("{:.<27s}{:10.4f}".format('Accuracy', acc))
            if type(numpy_y[0]) == str:
                pre = precision_score(y, predictions, pos_label=classes_[1])
                tpr = recall_score(y, predictions, pos_label=classes_[1])
                f1  =  f1_score(y,predictions, pos_label=classes_[1])
            else:
                pre = precision_score(y, predictions)
                tpr = recall_score(y, predictions)
                f1 =  f1_score(y,predictions)
            print("{:.<27s}{:10.4f}".format('Precision', pre))
            print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
            print("{:.<27s}{:10.4f}".format('F1-Score', f1))
            print("{:.<27s}{:10d}".format(\
                    'Total Misclassifications', tmisc))
            print("{:.<27s}{:9.1f}{:s}".format(\
                    'MISC (Misclassification)', misc, '%'))
            n_    = [conf_mat[0][0]+conf_mat[0][1], conf_mat[1][0]+conf_mat[1][1]]
            miscc = [100*conf_mat[0][1]/n_[0], 100*conf_mat[1][0]/n_[1]]
            for i in range(2):
                print("{:s}{:<16s}{:>9.1f}{:<1s}".format(\
                      '     class ', classes_[i], miscc[i], '%'))
            print("\n\n     Confusion     Class     Class")
            print("       Matrix", end="")
            print("{:1s}{:>10s}{:>10s}".format(" ", classes_[0], classes_[1]))
            
            for i in range(2):
                print("{:s}{:.<6s}".format('  Class ', classes_[i]), end="")
                for j in range(2):
                    print("{:>10d}".format(conf_mat[i][j]), end="")
                print("")
            print("")
        
        else:
            n_classes = len(dt.classes_)
            n_obs     = len(y)
            try:
                if n_classes < 2:
                    raise RuntimeError("  Call to display_nominal_metrics "+
                      "invalid.\n  Target has less than two classes.\n")
                    sys.exit()
            except:
                raise RuntimeError("  Call to display_nominal_metrics "+
                      "invalid.\n  Target has less than two classes.\n")
                sys.exit()
    
            np_y = np.ravel(y)
            classes_ = [" "]*len(dt.classes_)
            if type(np_y[0])==str:
                classes_ = dt.classes_
            else:
                for i in range(len(dt.classes_)):
                    classes_[i] = str(int(dt.classes_[i]))
            probability = dt.predict_proba(X) # get class probabilitie
            predictions = dt.predict(X) # get nominal class predictions
            conf_mat = confusion_matrix(y_true=y, y_pred=predictions)
            misc  = 0
            miscc = []
            n_    = []
            for i in range(n_classes):
                miscc.append(0)
                n_.append(0)
                for j in range(n_classes):
                    n_[i] = n_[i] + conf_mat[i][j]
                    if i != j:
                        misc = misc + conf_mat[i][j]
                        miscc[i] = miscc[i] + conf_mat[i][j]
                miscc[i] = 100*miscc[i]/n_[i]
            tmisc    = misc
            misc     = 100*misc/n_obs
            ase_sum  = 0
            mase_sum = 0
            for i in range(n_obs):
                for j in range(n_classes):
                    if np_y[i] == dt.classes_[j]:
                        ase_sum  += (1-probability[i,j])*(1-probability[i,j])
                        mase_sum += 1-probability[i,j]
                    else:
                        ase_sum  += probability[i,j]*probability[i,j]
                        mase_sum += probability[i,j]
            ase  = ase_sum/(n_classes*n_obs)
            mase = mase_sum/(n_classes*n_obs)
            print("\nModel Metrics")
            print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
            print("{:.<27s}{:10d}".format('Features', X.shape[1]))
            if type(dt) == RandomForestClassifier:
                print("{:.<27s}{:10d}".format('Trees in Forest', \
                                  dt.n_estimators))
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
            
            print("{:.<27s}{:10.4f}".format('ASE', ase))
            print("{:.<27s}{:10.4f}".format('Root ASE', sqrt(ase)))
            print("{:.<27s}{:10.4f}".format('Mean Absolute Error', mase))
            acc = accuracy_score(np_y, predictions)
            print("{:.<27s}{:10.4f}".format('Accuracy', acc))
            pre = precision_score(np_y, predictions, average='macro')
            print("{:.<27s}{:10.4f}".format('Precision', pre))
            tpr = recall_score(np_y, predictions, average='macro')
            print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
            f1 =  f1_score(np_y,predictions, average='macro')
            print("{:.<27s}{:10.4f}".format('F1-Score', f1))
            print("{:.<27s}{:10d}".format(\
                    'Total Misclassifications', tmisc))
            print("{:.<27s}{:9.1f}{:s}".format(\
                    'MISC (Misclassification)', misc, '%'))
            
            if type(dt.classes_[0]) == str:
                fstr = "{:s}{:.<16s}{:>9.1f}{:<1s}"
            else:
                fstr = "{:s}{:.<16.0f}{:>9.1f}{:<1s}"
            for i in range(len(dt.classes_)):
                print(fstr.format(\
                      '     class ', dt.classes_[i], miscc[i], '%'))      
                
            print("\n\n     Confusion")
            print("       Matrix    ", end="")
            
            if type(dt.classes_[0]) == str:
                fstr1 = "{:>7s}{:<3s}"
                fstr2 = "{:s}{:.<6s}"
            else:
                fstr1 = "{:>7s}{:<3.0f}"
                fstr2 = "{:s}{:.<6.0f}"
            for i in range(n_classes):
                print(fstr1.format('Class ', dt.classes_[i]), 
                      end="")
            print("")
            for i in range(n_classes):
                print(fstr2.format('Class ', dt.classes_[i]), 
                      end="")
                for j in range(n_classes):
                    print("{:>10d}".format(conf_mat[i][j]), end="")
                print("")
            print("")
            
            cr = classification_report(np_y, predictions, dt.classes_)
            print("\n",cr)
        
    def display_split_metrics(dt, Xt, yt, Xv, yv, target_names = None):
        if len(dt.classes_) == 2:
            numpy_yt = np.ravel(yt)
            numpy_yv = np.ravel(yv)
            if type(numpy_yt[0])==str:
                classes_ = dt.classes_
            else:
                classes_ = [str(int(dt.classes_[0])), str(int(dt.classes_[1]))]
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
            print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', 
                                          'Training', 'Validation'))
            print("{:.<23s}{:15d}{:15d}".format('Observations', 
                                              Xt.shape[0], Xv.shape[0]))
            
            print("{:.<23s}{:15d}{:15d}".format('Features', Xt.shape[1], 
                                                              Xv.shape[1]))
            if dt.max_depth==None:
                print("{:.<23s}{:>15s}{:>15s}".format('Maximum Tree Depth',
                                  "None", "None"))
            else:
                print("{:.<23s}{:15d}{:15d}".format('Maximum Tree Depth',
                                  dt.max_depth, dt.max_depth))
            print("{:.<23s}{:15d}{:15d}".format('Minimum Leaf Size', 
                                  dt.min_samples_leaf, dt.min_samples_leaf))
            print("{:.<23s}{:15d}{:15d}".format('Minimum split Size', 
                                  dt.min_samples_split, dt.min_samples_split))
    
            print("{:.<23s}{:15.4f}{:15.4f}".format('Mean Absolute Error', 
                          mean_absolute_error(zt,prob_t[:,1]), 
                          mean_absolute_error(zv,prob_v[:,1])))
            print("{:.<23s}{:15.4f}{:15.4f}".format('Avg Squared Error', 
                          mean_squared_error(zt,prob_t[:,1]), 
                          mean_squared_error(zv,prob_v[:,1])))
            
            acct = accuracy_score(yt, predict_t)
            accv = accuracy_score(yv, predict_v)
            print("{:.<23s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
            if type(numpy_yt[0])==str:
                pre_t = precision_score(yt, predict_t, pos_label=classes_[1])
                tpr_t = recall_score(yt, predict_t, pos_label=classes_[1])
                f1_t  = f1_score(yt,predict_t, pos_label=classes_[1])
                pre_v = precision_score(yv, predict_v, pos_label=classes_[1])
                tpr_v = recall_score(yv, predict_v, pos_label=classes_[1])
                f1_v  = f1_score(yv,predict_v, pos_label=classes_[1])
            else:
                pre_t = precision_score(yt, predict_t)
                tpr_t = recall_score(yt, predict_t)
                f1_t  = f1_score(yt,predict_t)
                pre_v = precision_score(yv, predict_v)
                tpr_v = recall_score(yv, predict_v)
                f1_v  = f1_score(yv,predict_v)
                
            print("{:.<27s}{:11.4f}{:15.4f}".format('Precision', pre_t, pre_v))
            print("{:.<27s}{:11.4f}{:15.4f}".format('Recall (Sensitivity)', 
                  tpr_t, tpr_v))
            print("{:.<27s}{:11.4f}{:15.4f}".format('F1-score', f1_t, f1_v))
            misct_ = conf_matt[0][1]+conf_matt[1][0]
            miscv_ = conf_matv[0][1]+conf_matv[1][0]
            misct = 100*misct_/len(yt)
            miscv = 100*miscv_/len(yv)
            n_t   = [conf_matt[0][0]+conf_matt[0][1], \
                     conf_matt[1][0]+conf_matt[1][1]]
            n_v   = [conf_matv[0][0]+conf_matv[0][1], \
                     conf_matv[1][0]+conf_matv[1][1]]
            misc_ = [[0,0], [0,0]]
            misc_[0][0] = 100*conf_matt[0][1]/n_t[0]
            misc_[0][1] = 100*conf_matt[1][0]/n_t[1]
            misc_[1][0] = 100*conf_matv[0][1]/n_v[0]
            misc_[1][1] = 100*conf_matv[1][0]/n_v[1]
            print("{:.<27s}{:11d}{:15d}".format(\
                    'Total Misclassifications', misct_, miscv_))
            print("{:.<27s}{:10.1f}{:s}{:14.1f}{:s}".format(\
                    'MISC (Misclassification)', misct, '%', miscv, '%'))
            for i in range(2):
                print("{:s}{:.<16s}{:>10.1f}{:<1s}{:>14.1f}{:<1s}".format(
                      '     class ', classes_[i], 
                      misc_[0][i], '%', misc_[1][i], '%'))
            print("\n\nTraining                  Class     Class")
            print("{:<21s}{:>10s}{:>10s}".format("Confusion Matrix", 
                          classes_[0], classes_[1]) )
            for i in range(2):
                print("{:6s}{:.<15s}".format('Class ', classes_[i]), end="")
                for j in range(2):
                    print("{:>10d}".format(conf_matt[i][j]), end="")
                print("")
            
            print("\n\nValidation                Class     Class")
            print("{:<21s}{:>10s}{:>10s}".format("Confusion Matrix", 
                          classes_[0], classes_[1]) )
            for i in range(2):
                print("{:6s}{:.<15s}".format('Class ', classes_[i]), end="")
                for j in range(2):
                    print("{:>10d}".format(conf_matv[i][j]), end="")
                print("")
            # In the binary case, the classification report is incorrect
            #cr = classification_report(yv, predict_v, dt.classes_)
            #print("\n",cr)
        else:
            try:
                if len(dt.classes_) < 2:
                    raise RuntimeError("  Call to display_nominal_split_metrics "+
                      "invalid.\n  Target has less than two classes.\n")
                    sys.exit()
            except:
                raise RuntimeError("  Call to display_nominal_split_metrics "+
                      "invalid.\n  Target has less than two classes.\n")
                sys.exit()
            predict_t = dt.predict(Xt)
            predict_v = dt.predict(Xv)
            conf_mat_t = confusion_matrix(y_true=yt, y_pred=predict_t)
            conf_mat_v = confusion_matrix(y_true=yv, y_pred=predict_v)
            prob_t = dt.predict_proba(Xt) # or is this dt._predict_proba_dt ?
            prob_v = dt.predict_proba(Xv)
            
            n_classes = len(dt.classes_)
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
            for i in range(n_classes):
                conf_matt.append(np.zeros(n_classes))
                conf_matv.append(np.zeros(n_classes))
            y_t = np.ravel(yt) # necessary because yt is a df with row keys
            y_v = np.ravel(yv) # likewise
            for i in range(n_classes):
                misct.append(0)
                n_t.append(0)
                miscv.append(0)
                n_v.append(0)
            for i in range(nt_obs):
                for j in range(n_classes):
                    if y_t[i] == dt.classes_[j]:
                        ase_sumt += (1-prob_t[i,j])*(1-prob_t[i,j])
                        idx = j
                    else:
                        ase_sumt += prob_t[i,j]*prob_t[i,j]
                for j in range(n_classes):
                    if predict_t[i] == dt.classes_[j]:
                        conf_matt[idx][j] += 1
                        break
                n_t[idx] += 1
                if predict_t[i] != y_t[i]:
                    misc_t     += 1
                    misct[idx] += 1
                    
            for i in range(nv_obs):
                for j in range(n_classes):
                    if y_v[i] == dt.classes_[j]:
                        ase_sumv += (1-prob_v[i,j])*(1-prob_v[i,j])
                        idx = j
                    else:
                        ase_sumv += prob_v[i,j]*prob_v[i,j]
                for j in range(n_classes):
                    if predict_v[i] == dt.classes_[j]:
                        conf_matv[idx][j] += 1
                        break
                n_v[idx] += 1
                if predict_v[i] != y_v[i]:
                    misc_v     += 1
                    miscv[idx] += 1
            misct_ = misc_t
            miscv_ = misc_v
            misc_t = 100*misc_t/nt_obs
            misc_v = 100*misc_v/nv_obs
            aset   = ase_sumt/(n_classes*nt_obs)
            asev   = ase_sumv/(n_classes*nv_obs)
            print("\n")
            print("{:.<23s}{:>15s}{:>15s}".format('Model Metrics', 
                                          'Training', 'Validation'))
            print("{:.<23s}{:15d}{:15d}".format('Observations', \
                                              Xt.shape[0], Xv.shape[0]))
            
            print("{:.<23s}{:15d}{:15d}".format('Features', Xt.shape[1], 
                                                            Xv.shape[1]))
            if type(dt) == RandomForestClassifier:
                print("{:.<23s}{:15d}{:15d}".format(\
                      'Trees in Forest', \
                      dt.n_estimators, dt.n_estimators))
            if dt.max_depth==None:
                print("{:.<23s}{:>15s}{:>15s}".format('Maximum Tree Depth',
                                  "None", "None"))
            else:
                print("{:.<23s}{:15d}{:15d}".format('Maximum Tree Depth',
                             dt.max_depth, dt.max_depth))
            print("{:.<23s}{:15d}{:15d}".format('Minimum Leaf Size', 
                             dt.min_samples_leaf, dt.min_samples_leaf))
            print("{:.<23s}{:15d}{:15d}".format('Minimum split Size', 
                             dt.min_samples_split, dt.min_samples_split))
    
            print("{:.<23s}{:15.4f}{:15.4f}".format('Avg Squared Error', 
                          aset, asev))
            
            print("{:.<23s}{:15.4f}{:15.4f}".format(\
                                    'Root ASE', sqrt(aset), sqrt(asev)))
            
            acct = accuracy_score(yt, predict_t)
            accv = accuracy_score(yv, predict_v)
            print("{:.<23s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
            
            print("{:.<23s}{:15.4f}{:15.4f}".format('Precision', 
                          precision_score(yt,predict_t, average='macro'), 
                          precision_score(yv,predict_v, average='macro')))
            print("{:.<23s}{:15.4f}{:15.4f}".format('Recall (Sensitivity)', 
                          recall_score(yt,predict_t, average='macro'), 
                          recall_score(yv,predict_v, average='macro')))
            print("{:.<23s}{:15.4f}{:15.4f}".format('F1-score', 
                          f1_score(yt,predict_t, average='macro'), 
                          f1_score(yv,predict_v, average='macro')))
            print("{:.<27s}{:11d}{:15d}".format(\
                    'Total Misclassifications', misct_, miscv_))
            print("{:.<27s}{:10.1f}{:s}{:14.1f}{:s}".format(\
                    'MISC (Misclassification)', misc_t, '%', misc_v, '%'))
            
            fstr0="{:s}{:.<16s}{:>10.1f}{:<1s}{:>14.1f}{:<1s}"
            fstr1="{:>7s}{:<3s}"
            fstr2="{:s}{:.<6s}"
            classes_ = []
            if type(dt.classes_[0])==str:
                classes_ = dt.classes_
            else:
                for i in range(n_classes):
                    classes_.append(str(int(dt.classes_[i])))
            for i in range(n_classes):
                misct[i] = 100*misct[i]/n_t[i]
                miscv[i] = 100*miscv[i]/n_v[i]
                print(fstr0.format(
                            '     class ', classes_[i], misct[i], 
                            '%', miscv[i], '%'))
    
            print("\n\nTraining")
            print("Confusion Matrix ", end="")
            for i in range(n_classes):
                print(fstr1.format('Class ', classes_[i]), 
                      end="")
            print("")
            for i in range(n_classes):
                print(fstr2.format('Class ', classes_[i]), 
                      end="")
                for j in range(n_classes):
                    print("{:>10d}".format(conf_mat_t[i][j]), end="")
                print("")
                
            ct = classification_report(yt, predict_t, target_names)
            print("\nTraining \nMetrics:\n",ct)
            
            print("\n\nValidation")
            print("Confusion Matrix ", end="")
            for i in range(n_classes):
                print(fstr1.format('Class ', classes_[i]), 
                      end="")
            print("")
            for i in range(n_classes):
                print(fstr2.format('Class ', classes_[i]), 
                      end="")
                for j in range(n_classes):
                    print("{:>10d}".format(conf_mat_v[i][j]), end="")
                print("")
            cv = classification_report(yv, predict_v, target_names)
            print("\nValidation \nMetrics:\n",cv)
