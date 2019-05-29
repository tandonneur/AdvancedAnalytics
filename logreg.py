#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:30:54 2019

@author: EJones
"""

import sys
import numpy  as np
from math import sqrt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report 
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
