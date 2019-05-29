#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:27:41 2019

@author: EJones
"""
import sys
import numpy  as np
from math import sqrt, log, pi
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import median_absolute_error

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
        ASEv = mean_squared_error(yt,predict_v)
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