"""
@author: Edward R Jones
@version 1.34
@copyright 2020 - Edward R Jones, all rights reserved.
"""

import sys
import numpy  as np
import pandas as pd
from math import sqrt

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report 
from sklearn.neural_network import MLPClassifier

def _indicator(y, classes):
    # Nominal metrics below index targets by column; convert 1-D labels
    # to a one-hot array whose columns follow classes
    if isinstance(y, pd.DataFrame) or np.ndim(y) > 1:
        return y
    y = np.ravel(y)
    return (y[:, None] == np.asarray(classes)[None, :]).astype(int)

class nn_regressor(object):
            
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
        print("{:.<23s}{:15d}".format('Hidden Layers',\
                              nn.n_layers_-2))
        print("{:.<23s}{:15d}".format('Outputs', \
                              nn.n_outputs_))
        nl = nn.n_layers_-2
        n_neurons = sum(nn.coefs_[i].shape[1] for i in range(nl))
        print("{:.<23s}{:15d}".format('Neurons',\
                              n_neurons))
        print("{:.<23s}{:15d}".format('Weights', \
                             n_weights))
        print("{:.<23s}{:15d}".format('NIterations', \
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
        print("{:.<23s}{:15d}{:15d}".format('Hidden Layers',\
                              (nn.n_layers_-2),(nn.n_layers_-2)))
        nl = nn.n_layers_-2
        n_neurons = sum(nn.coefs_[i].shape[1] for i in range(nl))
        print("{:.<23s}{:15d}{:15d}".format('Neurons',\
                              n_neurons, n_neurons))
        print("{:.<23s}{:15d}{:15d}".format('Outputs', \
                              nn.n_outputs_, nn.n_outputs_))
        print("{:.<23s}{:15d}{:15d}".format('Weights', \
                              n_weights, n_weights))
        print("{:.<23s}{:15d}{:15d}".format('Iterations', \
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
        
class nn_classifier(object): 
    def display_metrics(nn, X, y):
        if len(nn.classes_) <= 2: #BINARY METRICS
            numpy_y = np.ravel(y)
            if isinstance(numpy_y[0], str):
                classes_ = nn.classes_
            else:
                classes_ = [str(int(nn.classes_[0])), str(int(nn.classes_[1]))]
            z = np.zeros(len(y))
            predictions = nn.predict(X) # get binary class predictions
            conf_mat = confusion_matrix(y_true=y, y_pred=predictions,
                                        labels=nn.classes_)
            tmisc = conf_mat[0][1]+conf_mat[1][0]
            misc = 100*(tmisc)/(len(y))
            for i in range(len(y)):
                if numpy_y[i] == nn.classes_[1]:
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
            print("{:.<27s}{:10d}".format('Hidden Layers',\
                                  nn.n_layers_-2))
            print("{:.<27s}{:10d}".format('Outputs', \
                                  nn.n_outputs_))
            nl = nn.n_layers_-2
            n_neurons = sum(nn.coefs_[i].shape[1] for i in range(nl))
            print("{:.<27s}{:10d}".format('Neurons',\
                                  n_neurons))
            print("{:.<27s}{:10d}".format('Weights', \
                                 n_weights))
            print("{:.<27s}{:10d}".format('Iterations', \
                                 nn.n_iter_))
            print("{:.<27s}{:>10s}".format('Hidden Layer Activation', \
                                 nn.activation))
            print("{:.<27s}{:>10s}".format('Target Activation', \
                                 nn.out_activation_))
            print("{:.<27s}{:10.4f}".format('Loss Function', \
                                 nn.loss_))
            print("{:.<27s}{:10.4f}".format('Mean Absolute Error', \
                          mean_absolute_error(z,probability[:, 1])))
            print("{:.<27s}{:10.4f}".format('Avg Squared Error', \
                          mean_squared_error(z,probability[:, 1])))
            acc = accuracy_score(y, predictions)
            print("{:.<27s}{:10.4f}".format('Accuracy', acc))
            pre = precision_score(y, predictions, pos_label=nn.classes_[1])
            tpr = recall_score(y, predictions, pos_label=nn.classes_[1])
            f1  = f1_score(y, predictions, pos_label=nn.classes_[1])
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
        else: #NOMINAL METRICS
            n_classes = len(nn.classes_)
            n_obs = y.shape[0]
            if n_classes < 2:
                raise RuntimeError("\n  Call to display_metrics invalid"+\
                        "\n  Target does not appear to be nominal.\n")
                sys.exit()
            y        = _indicator(y, nn.classes_)
            predict_ = _indicator(nn.predict(X), nn.classes_)
            prob_ = nn.predict_proba(X)
            ase_sum  = 0
            mase_sum = 0
            misc_ = 0
            misc  = []
            n_    = []
            conf_mat = []
            for i in range(n_classes):
                conf_mat.append(np.zeros(n_classes))
            for i in range(n_classes):
                misc.append(0)
                n_.append(0)
            for i in range(n_obs):
                if type(y) == pd.DataFrame:
                   ky = y.iloc[i].argmax()
                else:
                   ky = y[i].argmax()
                for j in range(n_classes):
                    if ky == j:
                        ase_sum += (1-prob_[i,j])*(1-prob_[i,j])
                        mase_sum += 1-prob_[i,j]
                        idx = j
                        n_[j] += 1
                    else:
                        ase_sum  += prob_[i,j]*prob_[i,j]
                        mase_sum += prob_[i,j]
                
                kp = predict_[i].argmax()
                conf_mat[idx][kp] += 1
                if kp != idx:
                    misc_     += 1
                    misc[idx] += 1
            tmisc = misc_
            misc_ = 100*misc_/n_obs
            ase   = ase_sum/(n_classes*n_obs)
            mase  = mase_sum/(n_classes*n_obs)
            #Calculate number of weights
            n_weights = 0
            if type(nn)==MLPClassifier:
                for i in range(nn.n_layers_ - 1):
                    n_weights += len(nn.intercepts_[i])
                    n_weights += nn.coefs_[i].shape[0]*nn.coefs_[i].shape[1]
            print("\nModel Metrics")
            print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
            print("{:.<27s}{:10d}".format('Features', X.shape[1]))
            if type(nn)==MLPClassifier:
                print("{:.<27s}{:10d}".format('Hidden Layers', nn.n_layers_-2))
                for i in range(nn.n_layers_-2):
                    print("{:<24s}{:.<3d}{:10d}".format('   Neurons Hidden Layer ',\
                                      i+1, nn.coefs_[i].shape[1]))
                print("{:.<27s}{:10d}".format('Outputs', \
                                      nn.n_outputs_))
                print("{:.<27s}{:10d}".format('Weights', \
                                     n_weights))
                print("{:.<27s}{:10d}".format('Iterations', \
                                     nn.n_iter_))
                print("{:.<27s}{:>10s}".format('Hidden Layer Activation', \
                                     nn.activation))
                print("{:.<27s}{:10.4f}".format('Loss Function', \
                                     nn.loss_))
            print("{:.<27s}{:10.4f}".format('Avg Squared Error', ase))
            print("{:.<27s}{:10.4f}".format('Root ASE', sqrt(ase)))
            print("{:.<27s}{:10.4f}".format('Mean Absolute Error', mase))
            acc = accuracy_score(y, predict_)
            print("{:.<27s}{:10.4f}".format('Accuracy', acc))
            pre = precision_score(y, predict_, average='macro')
            print("{:.<27s}{:10.4f}".format('Precision', pre))
            tpr = recall_score(y, predict_, average='macro')
            print("{:.<27s}{:10.4f}".format('Recall (Sensitivity)', tpr))
            f1 =  f1_score(y,predict_, average='macro')
            print("{:.<27s}{:10.4f}".format('F1-Score', f1))
            if type(nn)==MLPClassifier:
                print("{:.<27s}{:10.4f}".format('Loss', nn.loss_))
            print("{:.<27s}{:10d}".format(\
                    'Total Misclassifications', tmisc))
            print("{:.<27s}{:9.1f}{:s}".format(\
                    'MISC (Misclassification)', misc_, '%'))
            if isinstance(nn.classes_[0], str):
                fstr = "{:s}{:.<16s}{:>9.1f}{:<1s}"
            else:
                fstr = "{:s}{:.<16.0f}{:>9.1f}{:<1s}"
            for i in range(n_classes):
                if n_[i]>0:
                    misc[i] = 100*misc[i]/n_[i]
                print(fstr.format(\
                      '     class ', nn.classes_[i], misc[i], '%'))
            print("\n\n     Confusion")
            print("       Matrix    ", end="")
            
            if isinstance(nn.classes_[0], str):
                fstr1 = "{:>7s}{:<3s}"
                fstr2 = "{:s}{:.<6s}"
            else:
                fstr1 = "{:>7s}{:<3.0f}"
                fstr2 = "{:s}{:.<6.0f}"
            for i in range(n_classes):
                print(fstr1.format('Class ', nn.classes_[i]), end="")
            print("")
            for i in range(n_classes):
                print(fstr2.format('Class ', nn.classes_[i]), end="")
                for j in range(n_classes):
                    print("{:>10.0f}".format(conf_mat[i][j]), end="")
                print("")
    
            cr = classification_report(y, predict_, labels=range(n_classes),
                               target_names=[str(c) for c in nn.classes_])
            print("\n",cr)
        
    def display_split_metrics(nn, Xt, yt, Xv, yv, target_names=None):
        if len(nn.classes_) <=2:
            try:
                if len(nn.classes_) != 2:
                    raise RuntimeError("  Call to display_split_metrics "+\
                      "invalid.\n  Target does not have two classes.\n")
                    sys.exit()
            except:
                raise RuntimeError("  Call to display_split_metrics "+\
                      "invalid.\n  Target does not have two classes.\n")
                sys.exit()
            if isinstance(nn.classes_[0], str):
                classes_ = nn.classes_
            else:
                classes_ = [str(int(nn.classes_[0])), str(int(nn.classes_[1]))]
            #Calculate number of weights
            n_weights = 0
            if type(nn)==MLPClassifier:
                for i in range(nn.n_layers_ - 1):
                    n_weights += len(nn.intercepts_[i])
                    n_weights += nn.coefs_[i].shape[0]*nn.coefs_[i].shape[1]
            numpy_yt = np.ravel(yt)
            numpy_yv = np.ravel(yv)
            zt = np.zeros(len(yt))
            zv = np.zeros(len(yv))
            #zt = deepcopy(yt)
            for i in range(len(yt)):
                if numpy_yt[i] == nn.classes_[1]:
                    zt[i] = 1
            for i in range(len(yv)):
                if numpy_yv[i] == nn.classes_[1]:
                    zv[i] = 1
            predict_t = nn.predict(Xt)
            predict_v = nn.predict(Xv)
            conf_matt = confusion_matrix(y_true=yt, y_pred=predict_t,
                                         labels=nn.classes_)
            conf_matv = confusion_matrix(y_true=yv, y_pred=predict_v,
                                         labels=nn.classes_)
            prob_t = nn.predict_proba(Xt)
            prob_v = nn.predict_proba(Xv)
            print("\n")
            print("{:.<27s}{:>11s}{:>15s}".format('Model Metrics', 
                                          'Training', 'Validation'))
            print("{:.<27s}{:11d}{:15d}".format('Observations', 
                                              Xt.shape[0], Xv.shape[0]))
            
            print("{:.<27s}{:11d}{:15d}".format('Features', Xt.shape[1], 
                                                            Xv.shape[1]))
            if type(nn)==MLPClassifier:
                print("{:.<27s}{:11d}{:15d}".format('Hidden Layers',\
                                      nn.n_layers_-2, nn.n_layers_-2))
                print("{:.<27s}{:11d}{:15d}".format('Outputs', \
                                      nn.n_outputs_, nn.n_outputs_))
                nl = nn.n_layers_-2
                n_neurons = sum(nn.coefs_[i].shape[1] for i in range(nl))
                print("{:.<27s}{:11d}{:15d}".format('Neurons',
                                      n_neurons, n_neurons))
                print("{:.<27s}{:11d}{:15d}".format('Weights', 
                                     n_weights, n_weights))
                print("{:.<27s}{:11d}{:15d}".format('Iterations', 
                                      nn.n_iter_, nn.n_iter_))
                print("{:.<27s}{:>11s}{:>15s}".format('Hidden Layer Activation', 
                                      nn.activation, nn.activation))
                print("{:.<27s}{:11.4f}{:15.4f}".format('Loss', 
                              nn.loss_, nn.loss_))
            print("{:.<27s}{:11.4f}{:15.4f}".format('Mean Absolute Error', 
                          mean_absolute_error(zt,prob_t[:,1]), 
                          mean_absolute_error(zv,prob_v[:,1])))
            print("{:.<27s}{:11.4f}{:15.4f}".format('Avg Squared Error', 
                          mean_squared_error(zt,prob_t[:,1]), \
                          mean_squared_error(zv,prob_v[:,1])))
            
            acct = accuracy_score(yt, predict_t)
            accv = accuracy_score(yv, predict_v)
            print("{:.<27s}{:11.4f}{:15.4f}".format('Accuracy', acct, accv))
            pos = nn.classes_[1]
            pre_t = precision_score(yt, predict_t, pos_label=pos)
            tpr_t = recall_score(yt, predict_t, pos_label=pos)
            f1_t  = f1_score(yt,predict_t, pos_label=pos)
            pre_v = precision_score(yv, predict_v, pos_label=pos)
            tpr_v = recall_score(yv, predict_v, pos_label=pos)
            f1_v  = f1_score(yv,predict_v, pos_label=pos)
                
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
            # In the binary case, the classification report is incorrect
            cr = classification_report(yv, predict_v, labels=nn.classes_)
            print("\n",cr)
            
            print("\n\nValidation                Class     Class")
            print("{:<21s}{:>10s}{:>10s}".format("Confusion Matrix", 
                          classes_[0], classes_[1]) )
            for i in range(2):
                print("{:6s}{:.<15s}".format('Class ', classes_[i]), end="")
                for j in range(2):
                    print("{:>10d}".format(conf_matv[i][j]), end="")
                print("")
            # In the binary case, the classification report is incorrect
            #cr = classification_report(yv, predict_v, nn.classes_)
            #print("\n",cr)
   
        else:
            # NOMINAL TARGET
            try:
                if len(nn.classes_) == 2:
                    raise RuntimeError("  Call to display_split_metrics "+\
                      "invalid.\n  Target is Binary.")
                    sys.exit()
            except:
                raise RuntimeError("  Call to display_split_metrics "+\
                      "invalid.\n  Target is Binary.\n")
                
            try:        
                if len(nn.classes_) < 3:
                    raise RuntimeError("  Call to display_split_metrics "+\
                      "invalid.\n  Target has less than three classes.\n")
                    sys.exit()
            except:
                raise RuntimeError("  Call to display_split_metrics "+\
                      "invalid.\n  Target has less than three classes.\n")
                sys.exit()

            yt        = _indicator(yt, nn.classes_)
            yv        = _indicator(yv, nn.classes_)
            predict_t = _indicator(nn.predict(Xt), nn.classes_)
            predict_v = _indicator(nn.predict(Xv), nn.classes_)
            
            if type(yt) == pd.DataFrame or len(yt.shape) > 1:
                # Nominal Target
                n = yt.shape[0]
                m = yt.shape[1]
            else:
                # Binomial Target
                n = len(yt)
                m = 2
            print("\n******** Confusion Matrix **********")
            print("------------------------------------\n")
            print("********  Training Data   **********")
            print("------------------------------------")
            conf_mat_t= np.zeros((m, m), dtype='int32')
            misc = 0
            if type(yt) == pd.DataFrame or len(yt.shape) > 1:
                if type(yt) == pd.DataFrame:
                   for i in range(n):
                       kp = predict_t[i,].argmax()
                       ky = yt.iloc[i,].argmax()
                       conf_mat_t[ky, kp] += 1
                       if ky != kp:
                           misc += 1
                else:
                   for i in range(n):
                       kp = predict_t[i,].argmax()
                       ky = yt[i,].argmax()
                       conf_mat_t[ky, kp] += 1
                       if ky != kp:
                           misc += 1
                miscp = 100*misc/n
                for i in range(m):
                    for j in range(m):
                        print("{:>6d} ".format(conf_mat_t[i,j]), end="")
                    print("")
            else:
                for i in range(n):
                    if   yt[i] == 0 and predict_t[i]  < 0.5:
                          conf_mat_t[0,0] += 1
                    elif yt[i] == 0 and predict_t[i] >= 0.5:
                          conf_mat_t[0,1] += 1
                          misc += 1
                    elif yt[i] == 1 and predict_t[i] >= 0.5:
                          conf_mat_t[1,1] += 1
                    elif yt[i] == 1 and predict_t[i]  < 0.5:
                          conf_mat_t[1,0] += 1
                          misc += 1
                miscp = 100*misc/n
                for i in range(m):
                    print("{:>5d} {:>5d}".\
                          format(conf_mat_t[i,0], conf_mat_t[i,1]))
            print("------------------------------------")
            print("Training Misclassification: {}/{}={:>5.3f}%".\
                      format(misc, n, miscp))
                
            if type(yv) == pd.DataFrame or len(yv.shape) > 1:
                # Nominal Target
                n = yv.shape[0]
                m = yv.shape[1]
            else:
                # Binomial Target
                n = len(yv)
                m = 2
            print("\n------------------------------------")
            print("******** Validation Data  **********")
            print("------------------------------------")
            conf_mat_v = np.zeros((m, m), dtype='int32')
            misc = 0
            if type(yv) == pd.DataFrame or len(yv.shape) > 1:
                if type(yv) == pd.DataFrame:
                   for i in range(n):
                       kp = predict_v[i,].argmax()
                       ky = yv.iloc[i,].argmax()
                       conf_mat_v[ky, kp] += 1
                       if ky != kp:
                           misc += 1
                else:
                   for i in range(n):
                       kp = predict_v[i,].argmax()
                       ky = yv[i,].argmax()
                       conf_mat_v[ky, kp] += 1
                       if ky != kp:
                           misc += 1
                miscp = 100*misc/n
                for i in range(m):
                    for j in range(m):
                        print("{:>6d} ".format(conf_mat_v[i,j]), end="")
                    print("")
            else:
                for i in range(n):
                    if   yv[i] == 0 and predict_v[i]  < 0.5:
                          conf_mat_v[0,0] += 1
                    elif yv[i] == 0 and predict_v[i] >= 0.5:
                          conf_mat_v[0,1] += 1
                          misc += 1
                    elif yv[i] == 1 and predict_v[i] >= 0.5:
                          conf_mat_v[1,1] += 1
                    elif yv[i] == 1 and predict_v[i]  < 0.5:
                          conf_mat_v[1,0] += 1
                          misc += 1
                miscp = 100*misc/n
                for i in range(m):
                    print("{:>5d} {:>5d}".\
                          format(conf_mat_v[i,0], conf_mat_v[i,1]))
            print("------------------------------------")
            print("Validation Misclassification: {}/{}={:>5.3f}%".\
                      format(misc, n, miscp))
            """************************************************************"""
            prob_t = nn.predict_proba(Xt) # or is this nn._predict_proba_dt ?
            prob_v = nn.predict_proba(Xv)
            
            n_classes = len(nn.classes_)
            ase_sumt = 0
            ase_sumv = 0
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
            #y_t = np.ravel(yt) # necessary because yt is a df with row keys
            #y_v = np.ravel(yv) # likewise
            y_t = yt
            y_v = yv
            for i in range(n_classes):
                misct.append(0)
                n_t.append(0)
                miscv.append(0)
                n_v.append(0)
            for i in range(nt_obs):
                for j in range(n_classes):
                    if type(y_t) == pd.DataFrame:
                       if y_t.iloc[i].argmax() == j:
                           ase_sumt += (1-prob_t[i,j])*(1-prob_t[i,j])
                           mase_sumt += (1-prob_t[i,j])
                           idx = j
                       else:
                           ase_sumt += prob_t[i,j]*prob_t[i,j]
                           mase_sumt += prob_t[i,j]
                    else:
                       if y_t[i].argmax() == j:
                           ase_sumt += (1-prob_t[i,j])*(1-prob_t[i,j])
                           mase_sumt += (1-prob_t[i,j])
                           idx = j
                       else:
                           ase_sumt += prob_t[i,j]*prob_t[i,j]
                           mase_sumt += prob_t[i,j]
                      
                n_t[idx] += 1
                if type(y_t) == pd.DataFrame:
                   if predict_t[i].argmax() != y_t.iloc[i].argmax():
                       misc_t     += 1
                       misct[idx] += 1
                else:
                   if predict_t[i].argmax() != y_t[i].argmax():
                       misc_t     += 1
                       misct[idx] += 1
                    
            for i in range(nv_obs):
                if type(y_v) == pd.DataFrame:
                   for j in range(n_classes):
                       if y_v.iloc[i].argmax() == j:
                           ase_sumv += (1-prob_v[i,j])*(1-prob_v[i,j])
                           mase_sumv += (1-prob_v[i,j])
                           idx = j
                       else:
                           ase_sumv += prob_v[i,j]*prob_v[i,j]
                           mase_sumv += prob_v[i,j]
                else:
                   for j in range(n_classes):
                       if y_v[i].argmax() == j:
                              ase_sumv += (1-prob_v[i,j])*(1-prob_v[i,j])
                              mase_sumv += (1-prob_v[i,j])
                              idx = j
                       else:
                              ase_sumv += prob_v[i,j]*prob_v[i,j]
                              mase_sumv += prob_v[i,j]

                n_v[idx] += 1
                if type(y_v) == pd.DataFrame:
                   if predict_v[i].argmax() != y_v.iloc[i].argmax():
                       misc_v     += 1
                       miscv[idx] += 1
                else:
                   if predict_v[i].argmax() != y_v[i].argmax():
                       misc_v     += 1
                       miscv[idx] += 1
                      
            misct_ = misc_t
            miscv_ = misc_v
            misc_t = 100*misc_t/nt_obs
            misc_v = 100*misc_v/nv_obs
            aset   = ase_sumt/(n_classes*nt_obs)
            asev   = ase_sumv/(n_classes*nv_obs)
            maset  = mase_sumt/(n_classes*nt_obs)
            masev  = mase_sumv/(n_classes*nv_obs)
                    #Calculate number of weights
            n_weights = 0
            for i in range(nn.n_layers_ - 1):
                n_weights += len(nn.intercepts_[i])
                n_weights += nn.coefs_[i].shape[0]*nn.coefs_[i].shape[1]
            print("\n")
            print("{:.<27s}{:>15s}{:>15s}".format('Model Metrics', \
                                          'Training', 'Validation'))
            print("{:.<27s}{:15d}{:15d}".format('Observations', \
                                              Xt.shape[0], Xv.shape[0]))
            
            print("{:.<27s}{:15d}{:15d}".format('Features', Xt.shape[1], \
                                                              Xv.shape[1]))
    
            print("{:.<27s}{:15d}{:15d}".format('Hidden Layers',\
                                  nn.n_layers_-2, nn.n_layers_-2))
            for i in range(nn.n_layers_-2):
                neurons=nn.coefs_[i].shape[1]
                print("{:<20s}{:.<3d}{:15d}{:15d}".format('   Neurons Hidden Layer ',\
                                  i+1, neurons, neurons))
            print("{:.<27s}{:15d}{:15d}".format('Outputs', \
                                  nn.n_outputs_, nn.n_outputs_))
            print("{:.<27s}{:15d}{:15d}".format('Weights', \
                                  n_weights, n_weights))
            print("{:.<27s}{:15d}{:15d}".format('Iterations', \
                                  nn.n_iter_, nn.n_iter_))
            print("{:.<27s}{:>15s}{:>15s}".format('Hidden Layer Activation', \
                                  nn.activation, nn.activation))
            print("{:.<27s}{:>15s}{:>15s}".format('Target Activation', \
                                  nn.out_activation_, nn.out_activation_))
            print("{:.<27s}{:15.4f}{:15.4f}".format('Loss', \
                          nn.loss_, nn.loss_))
            print("{:.<27s}{:15.4f}{:15.4f}".format('Avg Squared Error', \
                          aset, asev))
            print("{:.<27s}{:15.4f}{:15.4f}".format(\
                                    'Root ASE', sqrt(aset), sqrt(asev)))
            print("{:.<27s}{:15.4f}{:15.4f}".format('Mean Absolute Error', \
                          maset, masev))
            
            acct = accuracy_score(yt, predict_t)
            accv = accuracy_score(yv, predict_v)
            print("{:.<27s}{:15.4f}{:15.4f}".format('Accuracy', acct, accv))
            
            print("{:.<27s}{:15.4f}{:15.4f}".format('Precision', \
                          precision_score(yt,predict_t, average='macro'), \
                          precision_score(yv,predict_v, average='macro')))
            print("{:.<27s}{:15.4f}{:15.4f}".format('Recall (Sensitivity)', \
                          recall_score(yt,predict_t, average='macro'), \
                          recall_score(yv,predict_v, average='macro')))
            print("{:.<27s}{:15.4f}{:15.4f}".format('F1-score', \
                          f1_score(yt,predict_t, average='macro'), \
                          f1_score(yv,predict_v, average='macro')))
            print("{:.<27s}{:15d}{:15d}".format(\
                    'Total Misclassifications', misct_, miscv_))
            print("{:.<27s}{:14.1f}{:s}{:14.1f}{:s}".format(\
                    'MISC (Misclassification)', misc_t, '%', misc_v, '%'))

            fstr0="{:s}{:.<16s}{:>14.1f}{:<1s}{:>14.1f}{:<1s}"
            fstr1="{:>7s}{:<3s}"
            fstr2="{:s}{:.<6s}"
            classes_ = []
            if isinstance(nn.classes_[0], str):
                classes_ = nn.classes_
            else:
                for i in range(n_classes):
                    classes_.append(str(int(nn.classes_[i])))
            if target_names is None:
                target_names = [str(c) for c in classes_]
            for i in range(n_classes):
               if n_t[i] > 0:
                  misct[i] = 100*misct[i]/n_t[i]
               else:
                  misct[i] = 0
               if n_v[i] > 0:
                  miscv[i] = 100*miscv[i]/n_v[i]
               else:
                  miscv[i] = 0
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
                
            ct = classification_report(yt, predict_t, labels=range(n_classes),
                                       target_names=target_names)
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
            cv = classification_report(yv, predict_v, labels=range(n_classes),
                                       target_names=target_names)
            print("\nValidation \nMetrics:\n",cv)
            
class nn_keras(object):

    def accuracy_plot(history_dic):
        loss_values     = history_dic['loss']
        val_loss_values = history_dic["val_loss"]
        acc_values      = history_dic['accuracy']
        val_acc_values  = history_dic['val_accuracy']
        
        epochs          = range(1, len(val_loss_values) + 1)
        plt.subplot(211)
        plt.plot(epochs, loss_values,    'ro', label='Training Loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
        plt.title("Loss vs. Accuracy")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.subplot(212)
        plt.plot(epochs, acc_values,    'ro', label='Training Accuracy')
        plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        
    def display_metrics(nn, X, y):
        if (len(y.shape) == 1 or y.shape[1]==1) \
                             and len(np.unique(y)) == 2: #BINARY METRICS
            numpy_y  = np.ravel(y)
            classes_ = np.unique(y)
            if type(numpy_y[0])!=str:
                classes_ = [str(int(classes_[0])), str(int(classes_[1]))]
            
            predictions = (nn.predict(X)>0.5).astype('int32')
            conf_mat = confusion_matrix(y_true=y, y_pred=predictions)
            tmisc = conf_mat[0][1]+conf_mat[1][0]
            misc = 100*(tmisc)/(len(y))
            
            z = np.zeros(len(y))
            for i in range(len(y)):
                if numpy_y[i] == 1:
                    z[i] = 1
            probability = nn.predict(X) # get binary probabilities
            #Calculate number of weights
            n_weights  = nn.count_params()
            n_layers_  = len(nn.layers)
            n_outputs_ = len(classes_)
            
            print("\nModel Metrics")
            print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
            print("{:.<27s}{:10d}".format('Features', X.shape[1]))
            print("{:.<27s}{:10d}".format('Hidden Layers',\
                                           n_layers_-1))
            print("{:.<27s}{:10d}".format('Outputs', \
                                           n_outputs_-1))
                
            n_neurons = 0
            config_dic = nn.get_config()
            l   = 0
            for dic in config_dic['layers']:
                if dic['class_name'] == 'Dense':
                    n_neurons += dic['config']['units']
                    if l == n_layers_-2:
                        hl_activation = dic['config']['activation']
                    if l == n_layers_-1:
                        out_activation = dic['config']['activation']
                    l += 1

            print("{:.<27s}{:10d}".format('Neurons',\
                                  n_neurons))
            print("{:.<27s}{:10d}".format('Weights', \
                                 n_weights))
            print("{:.<27s}{:>10s}".format('Hidden Layer Activation', \
                                 hl_activation))
            print("{:.<27s}{:>10s}".format('Output Layer Activation', \
                                 out_activation))
            print("{:.<27s}{:10.4f}".format('Mean Absolute Error', \
                          mean_absolute_error(z,probability[:, 0])))
            print("{:.<27s}{:10.4f}".format('Avg Squared Error', \
                          mean_squared_error(z,probability[:, 0])))
            acc = accuracy_score(y, predictions)
            print("{:.<27s}{:10.4f}".format('Accuracy', acc))
                
            if type(numpy_y[0]) == str:
                pre = precision_score(y, predictions, pos_label=classes_[1])
                tpr = recall_score(y, predictions,    pos_label=classes_[1])
                f1  =  f1_score(y,predictions,        pos_label=classes_[1])
                pre = precision_score(y, predictions, pos_label=classes_[1])
                tpr = recall_score(y, predictions,    pos_label=classes_[1])
                f1 =  f1_score(y,predictions,         pos_label=classes_[1])
            else:
                pre = precision_score(y, predictions)
                tpr = recall_score(y, predictions)
                f1  = f1_score(y,predictions)
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
        else: #NOMINAL METRICS
            n_classes = y.shape[1]
            n_obs     = y.shape[0]
            if n_classes < 2:
                raise RuntimeError("\n  Call to display_metrics invalid"+\
                        "\n  Target does not appear to be nominal.\n")
                sys.exit()
            prob_    = nn.predict(X)
            predict_ = np.argmax(prob_, axis=-1)
            y_       = np.argmax(y, axis=-1)
            classes_ = np.unique(y_)
            
            #Calculate number of weights
            n_weights  = nn.count_params()
            n_layers_  = len(nn.layers)
            n_outputs_ = n_classes
            ase_sum  = 0
            mase_sum = 0
            misc_ = 0
            misc  = [0]*n_classes
            n_    = [0]*n_classes
            conf_mat = []
            for i in range(n_classes):
                conf_mat.append(np.zeros(n_classes))

            for i in range(n_obs):
                for j in range(n_classes):
                    if y[i][j] == 1:
                        ase_sum  += (1-prob_[i,j])*(1-prob_[i,j])
                        mase_sum +=  1-prob_[i,j]
                        idx = j
                        n_[j] += 1
                    else:
                        ase_sum  += prob_[i,j]*prob_[i,j]
                        mase_sum += prob_[i,j]
                j = predict_[i]
                conf_mat[idx][j] += 1
                if j != idx:
                    misc_     += 1
                    misc[idx] += 1
            tmisc = misc_
            misc_ = 100*misc_/n_obs
            ase   = ase_sum/(n_classes*n_obs)
            mase  = mase_sum/(n_classes*n_obs)
        
            print("\nModel Metrics")
            print("{:.<27s}{:10d}".format('Observations', X.shape[0]))
            print("{:.<27s}{:10d}".format('Features',     X.shape[1]))
            print("{:.<27s}{:10d}".format('Hidden Layers', n_layers_-1))
            
            n_neurons = 0
            config_dic = nn.get_config()
            l   = 0
            for dic in config_dic['layers']:
                if dic['class_name'] == 'Dense':
                    n_neurons += dic['config']['units']
                    if l <= n_layers_-2:
                        hl_activation = dic['config']['activation']
                        print("{:<24s}{:.<3d}{:10d}".format(\
                            '   Neurons Hidden Layer ', l, 
                            dic['config']['units']))
                    if l == n_layers_-1:
                        out_activation = dic['config']['activation']
                    l += 1
            print("{:.<27s}{:10d}".format('Outputs', \
                                  n_outputs_))
            print("{:.<27s}{:10d}".format('Weights', \
                                 n_weights))
            print("{:.<27s}{:>10s}".format('Hidden Layer Activation', \
                                 hl_activation))
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
            if type(classes_[0]) == str:
                fstr = "{:s}{:.<16s}{:>9.1f}{:<1s}"
            else:
                fstr = "{:s}{:.<16.0f}{:>9.1f}{:<1s}"
            for i in range(n_classes):
                if n_[i]>0:
                    misc[i] = 100*misc[i]/n_[i]
                print(fstr.format(\
                      '     class ', classes_[i], misc[i], '%'))
            print("\n\n     Confusion")
            print("       Matrix    ", end="")
            
            if type(classes_[0]) == str:
                fstr1 = "{:>7s}{:<3s}"
                fstr2 = "{:s}{:.<6s}"
            else:
                fstr1 = "{:>7s}{:<3.0f}"
                fstr2 = "{:s}{:.<6.0f}"
            for i in range(n_classes):
                print(fstr1.format('Class ', classes_[i]), end="")
            print("")
            for i in range(n_classes):
                print(fstr2.format('Class ', classes_[i]), end="")
                for j in range(n_classes):
                    print("{:>10.0f}".format(conf_mat[i][j]), end="")
                print("")
    
            cr = classification_report(y_, predict_, labels=classes_, digits=4)
            print("\n",cr)
            