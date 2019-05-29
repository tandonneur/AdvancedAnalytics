#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:44:18 2019

@author: EJones
"""
import sys
import numpy  as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report 

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
