#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Edward R Jones
@version 1.14
@copyright 2020 - Edward R Jones, all rights reserved.
"""

import sys
import warnings
from copy import deepcopy #Used to create sentiment word dictionary
import numpy  as np
import pandas as pd
from math import sqrt, log, pi
import statsmodels.api as sm
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, \
                            mean_squared_error, r2_score
from sklearn.metrics import f1_score, confusion_matrix, \
                            classification_report 

class linreg(object):
    
    def display_coef(lr, X, y, col=None):
        if type(col)==type(None):
            try:
                col = X.columns
            except:
                raise RuntimeError("  Call to display_coef is Invalid.\n"+
                  "  When X is not a pandas dataframe.  Parameter col "+
                  "required.")
        if len(col)!=X.shape[1]:
            raise RuntimeError("  Call to display_coef is Invalid.\n"+\
                  "  Number of Coefficient Names is not equal to the"+\
                  " Number of Columns in X")
            sys.exit()
        max_label = len('Intercept')+2
        for i in range(len(col)):
            if len(col[i]) > max_label:
                max_label = len(col[i])
        label_format = ("{:.<%i" %max_label)+"s}{:15.4f}"
        
        if type(lr) != sm.regression.linear_model.RegressionResultsWrapper:
            print("TYPE: ", type(lr))
            print(label_format.format('Intercept', lr.intercept_))
            for i in range(X.shape[1]):
                print(label_format.format(col[i], lr.coef_[i]))
        else:
            for i in range(X.shape[1]):
                print(label_format.format(col[i], lr.params[i]))
    
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
    
    def display_coef(lr, X, y, col=None):
        if type(col)==type(None):
            try:
                col = X.columns
            except:
                raise RuntimeError("  Call to display_coef is Invalid.\n"+
                  "  When X is not a pandas dataframe.  Parameter col "+
                  "required.")
        if len(col)!=X.shape[1]:
            raise RuntimeError("  Call to display_coef is Invalid.\n"+\
                  "  Number of Coefficient Names is not equal to the"+\
                  " Number of Columns in X")
            sys.exit()
        max_label = len('Intercept')+2
        for i in range(len(col)):
            if len(col[i]) > max_label:
                max_label = len(col[i])
        label_format = ("{:.<%i" %max_label)+"s}{:15.4f}"
        if type(y) == np.ndarray:
            k = len(np.unique(y)) #numpy array
        else:
            k = len(lr.classes_) #pandas vector
        nx = X.shape[1]
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
        print("{:.<27s}{:10.4f}".format('Sensitivity (Recall)', tpr))
        if (TN+FP)>0:
            tnr = TN/(TN+FP)
        print("{:.<27s}{:10.4f}".format('Specificity (Selectivity)', tnr))
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
        
    def display_metrics(lr, X, y):
        if len(lr.classes_) == 2:
            y_ = np.ravel(y) # necessary because yt is a df with row keys
            if type(y_[0])==str:
                classes_ = lr.classes_
            else:
                classes_ = [str(int(lr.classes_[0])), str(int(lr.classes_[1]))]
            z  = np.zeros(len(y_))
            predictions = lr.predict(X) # get binary class predictions
            conf_mat = confusion_matrix(y_true=y, y_pred=predictions)
            tmisc = conf_mat[0][1]+conf_mat[1][0]
            misc  = 100*(tmisc)/(len(y_))
            for i in range(len(y_)):
                if y_[i] == 1:
                    z[i] = 1
            #probability = lr.predict_proba(X) # get binary probabilities
            try:
                probability = lr.predict_proba(X)
            except:
                probability = lr._predict_proba_lr(X)
            print("\nModel Metrics")
            print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
            print("{:.<27s}{:10d}".format('Coefficients', X.shape[1]+1))
            print("{:.<27s}{:10d}".format('DF Error', X.shape[0]-X.shape[1]-1))
            if lr.n_iter_ == None:
                print("{:.<27s}{:>10s}".format('Iterations', 'None'))
            elif type(lr.n_iter_)==np.ndarray:
                print("{:.<27s}{:10d}".format('Iterations', lr.n_iter_[0]))
            else:
                print("{:.<27s}{:10d}".format('Iterations', lr.n_iter_))
            print("{:.<27s}{:10.4f}".format('Mean Absolute Error', \
                          mean_absolute_error(z,probability[:, 1])))
            print("{:.<27s}{:10.4f}".format('Avg Squared Error', \
                          mean_squared_error(z,probability[:, 1])))
            acc = accuracy_score(y, predictions)
            print("{:.<27s}{:10.4f}".format('Accuracy', acc))
            if type(y_[0]) == str:
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
             
            # In the binary case, the classification report is incorrect
            #cr = classification_report(yv, predict_v, lr.classes_)
            
        else:
            n_classes = len(lr.classes_)
            predict_ = lr.predict(X)
            try:
                prob_ = lr.predict_proba(X) #ver>=21
            except:
                prob_ = lr._predict_proba_lr(X) 
            ase_sum  = 0
            mase_sum = 0
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
            y_ = np.ravel(y) # necessary because yt is a df with row keys
            for i in range(n_classes):
                misc.append(0)
                n_.append(0)
            for i in range(n_obs):
                for j in range(n_classes):
                    if y_[i] == lr.classes_[j]:
                        ase_sum  += (1-prob_[i,j])*(1-prob_[i,j])
                        mase_sum += 1-prob_[i,j]
                        idx = j
                    else:
                        ase_sum  += prob_[i,j]*prob_[i,j]
                        mase_sum += prob_[i,j]
                for j in range(n_classes):
                    if predict_[i] == lr.classes_[j]:
                            conf_mat[idx][j] += 1
                            break
                n_[idx] += 1
                if predict_[i] != y_[i]:
                    misc_     += 1
                    misc[idx] += 1
            tmisc = misc_
            misc_ = 100*misc_/n_obs
            ase   = ase_sum/(n_classes*n_obs)
            mase  = mase_sum/(n_classes*n_obs)
            
            print("\nModel Metrics")
            print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
            n_coef = len(lr.coef_)*(len(lr.coef_[0])+1)
            print("{:.<27s}{:10d}".format('Coefficients', n_coef))
            print("{:.<27s}{:10d}".format('DF Error', X.shape[0]-n_coef))
            print("{:.<27s}{:10d}".format('Iterations', lr.n_iter_.max()))
            print("{:.<27s}{:10.4f}".format('Avg Squared Error', ase))
            print("{:.<27s}{:10.4f}".format('Root ASE', sqrt(ase)))
            print("{:.<27s}{:10.4f}".format('Mean Absolute Error', mase))
            acc = accuracy_score(y_, predict_)
            print("{:.<27s}{:10.4f}".format('Accuracy', acc))
            pre = precision_score(y_, predict_, average='macro')
            print("{:.<27s}{:10.4f}".format('Precision', pre))
            tpr = recall_score(y_, predict_, average='macro')
            print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
            f1 =  f1_score(y_,predict_, average='macro')
            print("{:.<27s}{:10.4f}".format('F1-Score', f1))
            print("{:.<27s}{:10d}".format(\
                    'Total Misclassifications', tmisc))
            print("{:.<27s}{:9.1f}{:s}".format(\
                    'MISC (Misclassification)', misc_, '%'))
            
            if type(lr.classes_[0]) == str:
                fstr = "{:s}{:.<16s}{:>9.1f}{:<1s}"
            else:
                fstr = "{:s}{:.<16.0f}{:>9.1f}{:<1s}"
            for i in range(n_classes):
                misc[i] = 100*misc[i]/n_[i]
                print(fstr.format(\
                      '     class ', lr.classes_[i], misc[i], '%'))
            print("\n\n     Confusion")
            print("       Matrix    ", end="")
            
            if type(lr.classes_[0]) == str:
                fstr1 = "{:>7s}{:<3s}"
                fstr2 = "{:s}{:.<6s}"
            else:
                fstr1 = "{:>7s}{:<3.0f}"
                fstr2 = "{:s}{:.<6.0f}"
            for i in range(n_classes):
                print(fstr1.format('Class ', lr.classes_[i]), 
                      end="")
            print("")
            for i in range(n_classes):
                print(fstr2.format('Class ', lr.classes_[i]), 
                      end="")
                for j in range(n_classes):
                    print("{:>10d}".format(conf_mat[i][j]), end="")
                print("")
    
            cr = classification_report(y, predict_, lr.classes_)
            print("\n",cr)
        
        
    def display_split_metrics(lr, Xt, yt, Xv, yv, target_names=None):
        if len(lr.classes_) == 2:
            yt_= np.ravel(yt)
            yv_= np.ravel(yv)
            if type(yt_[0])==str:
                classes_ = lr.classes_
            else:
                classes_ = [str(int(lr.classes_[0])), str(int(lr.classes_[1]))]
            zt = np.zeros(len(yt_))
            zv = np.zeros(len(yv_))
            #zt = deepcopy(yt)
            for i in range(len(yt)):
                if yt_[i] == 1:
                    zt[i] = 1
            for i in range(len(yv)):
                if yv_[i] == 1:
                    zv[i] = 1
            predict_t = lr.predict(Xt)
            predict_v = lr.predict(Xv)
            conf_matt = confusion_matrix(y_true=yt_, y_pred=predict_t)
            conf_matv = confusion_matrix(y_true=yv_, y_pred=predict_v)
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
            print("{:.<23s}{:15d}{:15d}".format('Iterations', \
                                  lr.n_iter_.max(), lr.n_iter_.max()))
            print("{:.<23s}{:15.4f}{:15.4f}".format('Mean Absolute Error', \
                          mean_absolute_error(zt,prob_t[:,1]), \
                          mean_absolute_error(zv,prob_v[:,1])))
            print("{:.<23s}{:15.4f}{:15.4f}".format('Avg Squared Error', \
                          mean_squared_error(zt,prob_t[:,1]), \
                          mean_squared_error(zv,prob_v[:,1])))
            
            acct = accuracy_score(yt_, predict_t)
            accv = accuracy_score(yv_, predict_v)
            print("{:.<23s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
            if type(yt_[0])==str:
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
            #cr = classification_report(yv, predict_v, lr.classes_)
            #print("\n",cr)
   
        else:
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
            conf_mat_t = confusion_matrix(y_true=yt, y_pred=predict_t)
            conf_mat_v = confusion_matrix(y_true=yv, y_pred=predict_v)
            #prob_t = lr.predict_proba(Xt)
            #prob_v = lr.predict_proba(Xv)
            ase_sumt  = 0
            ase_sumv  = 0
            mase_sumt = 0
            mase_sumv = 0
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
                    if y_t[i] == lr.classes_[j]:
                        ase_sumt  += (1-prob_t[i,j])*(1-prob_t[i,j])
                        mase_sumt += 1-prob_t[i,j]
                        idx = j
                    else:
                        ase_sumt  += prob_t[i,j]*prob_t[i,j]
                        mase_sumt += prob_t[i,j]
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
                        ase_sumv  += (1-prob_v[i,j])*(1-prob_v[i,j])
                        mase_sumv += 1-prob_v[i,j]
                        idx = j
                    else:
                        ase_sumv  += prob_v[i,j]*prob_v[i,j]
                        mase_sumv += prob_v[i,j]
                for j in range(n_classes):
                    if predict_v[i] == lr.classes_[j]:
                        conf_matv[idx][j] += 1
                        break
                n_v[idx] += 1
                if predict_v[i] != y_v[i]:
                    misc_v     += 1
                    miscv[idx] += 1
            misct_  = misc_t
            miscv_  = misc_v
            misc_t  = 100*misc_t/nt_obs
            misc_v  = 100*misc_v/nv_obs
            aset    = ase_sumt/(n_classes*nt_obs)
            asev    = ase_sumv/(n_classes*nv_obs)
            maset   = mase_sumt/(n_classes*nt_obs)
            masev   = mase_sumv/(n_classes*nv_obs)
            for i in range(n_classes):
                for j in range(n_classes):
                    if conf_mat_t[i][j] != conf_matt[i][j] or \
                       conf_mat_v[i][j] != conf_matv[i][j]:
                        raise RuntimeError("***SKLEARN CONFLICT!!. ")
                        sys.exit()
            print("")
            print("{:.<27s}{:>11s}{:>13s}".format('Model Metrics', \
                                          'Training', 'Validation'))
            print("{:.<27s}{:10d}{:11d}".format('Observations', \
                                              Xt.shape[0], Xv.shape[0]))
            n_coef = len(lr.coef_)*(len(lr.coef_[0])+1)
            print("{:.<27s}{:10d}{:11d}".format('Coefficients', \
                                              n_coef, n_coef))
            print("{:.<27s}{:10d}{:11d}".format('DF Error', \
                          Xt.shape[0]-n_coef, Xt.shape[0]-n_coef))
            print("{:.<27s}{:10d}{:11d}".format('Iterations', \
                                  lr.n_iter_.max(), lr.n_iter_.max()))
            
            print("{:.<27s}{:10.4f}{:11.4f}".format(
                                    'ASE', aset, asev))
            print("{:.<27s}{:10.4f}{:11.4f}".format(\
                                    'Root ASE', sqrt(aset), sqrt(asev)))
            print("{:.<27s}{:10.4f}{:11.4f}".format(
                                    'Mean Absolute Error', maset, masev))
            acct = accuracy_score(y_t, predict_t)
            accv = accuracy_score(y_v, predict_v)
            print("{:.<27s}{:10.4f}{:11.4f}".format('Accuracy', acct, accv))
            
            print("{:.<27s}{:10.4f}{:11.4f}".format('Precision', \
                          precision_score(y_t,predict_t, average='macro'), \
                          precision_score(y_v,predict_v, average='macro')))
            print("{:.<27s}{:10.4f}{:11.4f}".format('Recall (Sensitivity)', \
                          recall_score(y_t,predict_t, average='macro'), \
                          recall_score(y_v,predict_v, average='macro')))
            print("{:.<27s}{:10.4f}{:11.4f}".format('F1-score', \
                          f1_score(y_t,predict_t, average='macro'), \
                          f1_score(y_v,predict_v, average='macro')))
            print("{:.<27s}{:10d}{:11d}".format(\
                    'Total Misclassifications', misct_, miscv_))
            print("{:.<27s}{:9.1f}{:s}{:10.1f}{:s}".format(\
                    'MISC (Misclassification)', misc_t, '%', misc_v, '%'))

            fstr0="{:s}{:.<16s}{:>9.1f}{:<1s}{:>10.1f}{:<1s}"
            fstr1="{:>7s}{:<3s}"
            fstr2="{:s}{:.<6s}"
            classes_ = []
            if type(lr.classes_[0])==str:
                classes_ = lr.classes_
            else:
                for i in range(n_classes):
                    classes_.append(str(int(lr.classes_[i])))
            for i in range(n_classes):
                misct[i] = 100*misct[i]/n_t[i]
                miscv[i] = 100*miscv[i]/n_v[i]
                print(fstr0.format(\
                      '     class ', classes_[i], misct[i], '%', miscv[i], '%'))
    
            print("\n\nTraining")
            print("Confusion Matrix ", end="")
            for i in range(n_classes):
                print(fstr1.format('Class ', classes_[i]), end="")
            print("")
            for i in range(n_classes):
                print(fstr2.format('Class ', classes_[i]), end="")
                for j in range(n_classes):
                    print("{:>10d}".format(conf_mat_t[i][j]), end="")
                print("")
                
            ct = classification_report(yt, predict_t, target_names)
            print("\nTraining \nMetrics:\n",ct)
            
            print("\n\nValidation")
            print("Confusion Matrix ", end="")
            for i in range(n_classes):
                print(fstr1.format('Class ', classes_[i]), end="")
            print("")
            for i in range(n_classes):
                print(fstr2.format('Class ', classes_[i]), end="")
                for j in range(n_classes):
                    print("{:>10d}".format(conf_mat_v[i][j]), end="")
                print("")
            cv = classification_report(yv, predict_v, target_names)
            print("\nValidation \nMetrics:\n",cv)
    
     
    # *********************************************************************
        
class stepwise(object):
        
    def __init__(self, df, yname, reg, xnames=None, \
                 method="stepwise", crit_in=0.1, crit_out=0.1, \
                 x_force=None, verbose=False, deep=True):
        
        warnings.simplefilter(action="ignore", category=FutureWarning)
        if reg!="linear" and reg!="logistic":
            raise RuntimeError("***Call to stepwise invalid. "+\
            "***   Reg must be set to 'linear' or 'logistic'.")
            sys.exit()
        if type(df)!= pd.DataFrame:
            #raise RuntimeError("***Call to stepwise invalid. "+\
            #"***   DF Not DataFrame  ***")
            pass
        if df.shape[0] < 2:
            raise RuntimeError("***Call to stepwise invalid. "+\
            "***   Required Dataframe has less the 2 observations.")    
        if type(yname)!= str:
            raise RuntimeError("***Call to stepwise invalid. "+\
                "***   Parameter yname not a string name in DataFrame.")
            sys.exit()
        if not(yname in df.columns):
            raise RuntimeError("***Call to stepwise invalid. "+\
                "***   Required parameter yname not in DataFrame.")
            sys.exit()
        if reg=='logistic':
            yvalues = df[yname].unique()
            if len(yvalues) != 2:
                raise RuntimeError("***Call to stepwise invalid. "+\
                "***  The target is not binary.")
                sys.exit()
        if type(xnames)!= type(None):
            if not(all(item in df.columns for item in xnames)):
                raise RuntimeError("***Call to stepwise invalid. "+\
                                   "***   xnames are not all in DataFrame.")
                sys.exit()
        if method!="stepwise" and method!="forward" and method!="backward":
                raise RuntimeError("***Call to stepwise invalid. "+\
                                   "***   method is invalid.")
                sys.exit()
        if type(crit_in)==str:
            if crit_in!="AIC" and crit_in!="BIC":
                raise RuntimeError("***Call to stepwise invalid. "+\
                                   "***   crit_in is invalid.")
                sys.exit()
        else:
            if type(crit_in)!=float:
                raise RuntimeError("***Call to stepwise invalid. "+\
                                   "***   crit_in is invalid.")
                sys.exit()
            else:
                if crit_in>1.0 or crit_in<0.0:
                    raise RuntimeError("***Call to stepwise invalid. "+\
                                   "***   crit_in is invalid.")
                    sys.exit()
        if type(crit_out)==str:
            if crit_out!="AIC" and crit_out!="BIC":
                raise RuntimeError("***Call to stepwise invalid. "+\
                                   "***   crit_out is invalid.")
                sys.exit()
        else:
            if type(crit_out)!=float:
                raise RuntimeError("***Call to stepwise invalid. "+\
                                   "***   crit_out is invalid.")
                sys.exit()
            else:
                if crit_out>1.0 or crit_out<0:
                    raise RuntimeError("***Call to stepwise invalid. "+\
                                   "***   crit_out is invalid.")
                    sys.exit()
        if type(x_force)!=type(None) and \
            not(all(item in df.columns for item in x_force)):
                raise RuntimeError("***Call to stepwise invalid. "+\
                                   "***   x_force is invalid.")
                sys.exit()
        if deep==True:
            self.df_copy = deepcopy(df)
        else:
            self.df_copy = df
        
        # string - column name in df for y
        self.yname     = yname
        # None or string = list of column names in df for X var.
        if type(xnames)!= type(None):
            self.xnames = xnames   # list of strings (col names)
        else:
            self.xnames = list(set(df.columns)-set([yname]))
        # string - "stepwise", "backward" or "forward"    
        self.method    = method   # string
        # string - "linear" or "logistic"
        self.reg       = reg      # string
        # string = "AIC" or "BIC", or p=[0,1]
        if type(crit_in)==str or type(crit_out)==str:
            warnings.warn("\n***Call to stepwise invalid: "+ \
                " crit_in and crit_out must be a number between 0 and 1.")
            self.crit_in  = 0.1
            self.crit_out = 0.1
        else:
            self.crit_in   = crit_in  # float
            self.crit_out  = crit_out # float
        # [] of string = list of column names in df forced into model
        if type(x_force)!= type(None):
            self.x_force = x_force   # list of strings (col names)
        else:
            self.x_force = []
        # True or False, control display of steps selected
        self.verbose = verbose
        # initialized list of selected columns in df
        self.selected_ = []
        
        return         
# *************************************************************************
    def stepwise_(self):
        """
        Linear Regression Stepwise Selection  
        Author: Mahitha RAJENDRAN THANGADURAI
        """ 
        initial_list = []
        included = initial_list
        if self.crit_out<self.crit_in:
            raise RuntimeError("\n***Call to stepwise invalid: "+ \
                "crit_out smaller than crit_in.")
            sys.exit()
        X = self.df_copy[self.xnames]
        y = self.df_copy[self.yname]
        warnings.filterwarnings("once", category=UserWarning)
        while True:
            changed=False
            # forward step
            excluded = list(set(X.columns)-set(included))
            new_pval = pd.Series(index=excluded) 
            if self.reg=="linear":
                for new_column in excluded:
                    model = sm.OLS(y, \
                            sm.add_constant(pd.DataFrame(\
                            X[included+[new_column]]))).fit()
                    new_pval[new_column] = model.pvalues.loc[new_column]
            else:            
                for new_column in excluded:
                    Xc      = sm.add_constant(pd.DataFrame(X[included+[new_column]]))
                    model   = sm.Logit(y, Xc)
                    results = model.fit(disp=False)
                    new_pval[new_column] = results.pvalues.loc[new_column]
            best_pval = new_pval.min()
            if best_pval < self.crit_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
                if self.verbose:
                    print('Add  {:30} with p-value {:.6}'.
                          format(best_feature, best_pval))
            # backward step
            if self.reg=="linear":
                model   = sm.OLS(y, sm.add_constant(\
                                                  pd.DataFrame(X[included])))
                results = model.fit()
            else:
                Xc      = sm.add_constant(pd.DataFrame(X[included]))
                model   = sm.Logit(y, Xc)
                results = model.fit(disp=False)
            pvalues = results.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > self.crit_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed=True
                if self.verbose:
                    print('Remove {:30} with p-value {:.6}'.
                          format(worst_feature,worst_pval))
            if not changed:
                break
        return included

# **************************************************************************
    def forward_(self):
        """
        Linear Regression Forward Stepwise Selection
        Author: SHAOFANG
        """  
        initial_list = []
        included = list(initial_list)
        X = self.df_copy[self.xnames]
        y = self.df_copy[self.yname]
        warnings.filterwarnings("once", category=UserWarning)
        while True:
            changed=False
            excluded = list(set(X.columns)-set(included))
            new_pval = pd.Series(index=excluded)
            if self.reg=="linear":
                for new_column in excluded:
                    model = sm.OLS(y, \
                            sm.add_constant(pd.DataFrame(\
                            X[included+[new_column]])))
                    results = model.fit(disp=False)
                    new_pval[new_column] = results.pvalues.loc[new_column]
            else:
                for new_column in excluded:
                    Xc = sm.add_constant(pd.DataFrame(X[included+[new_column]]))
                    model   = sm.Logit(y, Xc)
                    results = model.fit(disp=False)
                    new_pval[new_column] = results.pvalues.loc[new_column]
            best_pval = new_pval.min()
            if best_pval < self.crit_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
                if self.verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature,\
                          best_pval))
    
            if not changed:
                break
        return included

# **************************************************************************
        
    def backward_(self):
        """
        Linear Regression Backkward Stepwise Selection  
        Author: Tara Gaddis
        """  
        included = list(self.xnames)
        X = self.df_copy[included]
        y = self.df_copy[self.yname]
        warnings.filterwarnings("once", category=UserWarning)
        while True:
            changed=False
            new_pval = pd.Series(index=included)
            if self.reg=="linear":
                model = sm.OLS(y, sm.add_constant(\
                                pd.DataFrame(X[included]))).fit()
            else:
                model = sm.Logit(y, sm.add_constant(\
                                pd.DataFrame(X[included]))).fit(disp=False)
                
            for new_column in included:
                new_pval[new_column] = model.pvalues.loc[new_column]
            worst_pval = new_pval.max()
            if worst_pval > self.crit_out:
                worst_feature = new_pval.idxmax()
                included.remove(worst_feature)
                changed=True
                if self.verbose:
                    print('Remove  {:30} with p-value {:.6}'.\
                          format(worst_feature, worst_pval))
            if not changed:
                break    
        return included
    
# **************************************************************************

    def fit_transform(self):
        if self.method=="stepwise":
            self.selected_ = self.stepwise_()
        else:
            if self.method=="forward":
                self.selected_ = self.forward_()
            else:
                self.selected_ = self.backward_()
        warnings.filterwarnings("always", category=UserWarning)
        return self.selected_
# **************************************************************************
 