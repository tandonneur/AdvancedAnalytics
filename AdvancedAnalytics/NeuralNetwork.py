"""
@author: Edward R Jones
@version 1.14
@copyright 2020 - Edward R Jones, all rights reserved.
"""

import sys
import numpy  as np
from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report 
from sklearn.neural_network import MLPClassifier

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
        n_neurons = 0
        nl = nn.n_layers_-2
        if nl>1:
            for i in range(nl):
                n_neurons += nn.hidden_layer_sizes[i]
        else:
            n_neurons = nn.hidden_layer_sizes
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
        n_neurons = 0
        nl = nn.n_layers_-2
        if nl>1:
            for i in range(nl):
                n_neurons += nn.hidden_layer_sizes[i]
        else:
            n_neurons = nn.hidden_layer_sizes
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
            numpy_y = np.ravel(y)
            if type(numpy_y[0])==str:
                classes_ = nn.classes_
            else:
                classes_ = [str(int(nn.classes_[0])), str(int(nn.classes_[1]))]
            z = np.zeros(len(y))
            predictions = nn.predict(X) # get binary class predictions
            conf_mat = confusion_matrix(y_true=y, y_pred=predictions)
            tmisc = conf_mat[0][1]+conf_mat[1][0]
            misc = 100*(tmisc)/(len(y))
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
            print("{:.<27s}{:10d}".format('Hidden Layers',\
                                  nn.n_layers_-2))
            print("{:.<27s}{:10d}".format('Outputs', \
                                  nn.n_outputs_))
            n_neurons = 0
            nl = nn.n_layers_-2
            if nl>1:
                for i in range(nl):
                    n_neurons += nn.hidden_layer_sizes[i]
            else:
                n_neurons = nn.hidden_layer_sizes
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
            if type(numpy_y[0]) == str:
                pre = precision_score(y, predictions, pos_label=classes_[1])
                tpr = recall_score(y, predictions, pos_label=classes_[1])
                f1  =  f1_score(y,predictions, pos_label=classes_[1])
                pre = precision_score(y, predictions, pos_label=classes_[1])
                tpr = recall_score(y, predictions, pos_label=classes_[1])
                f1 =  f1_score(y,predictions, pos_label=classes_[1])
            else:
                pre = precision_score(y, predictions)
                tpr = recall_score(y, predictions)
                f1  =  f1_score(y,predictions)
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
            n_classes = len(nn.classes_)
            n_obs = y.shape[0]
            if n_classes < 2:
                raise RuntimeError("\n  Call to display_nominal_metrics invalid"+\
                        "\n  Target does not appear to be nominal.\n"+\
                        "  Try using NeuralNetwork.display_binary_metrics()"+\
                        " instead.\n")
                sys.exit()
            predict_ = nn.predict(X)
            """
            if len(predict_.shape) !=2:
                raise RuntimeError("\n  Call to display_nominal_metrics invalid"+\
                   "\n  Appears the target is not"+\
                   " encoded into multiple binary columns\n" + \
                   "  Try using ReplaceImputeEncode to encode target\n")
                sys.exit()
            """
            prob_ = nn.predict_proba(X)
            """
            for i in range(n_obs):
                max_prob   = 0
                prediction = 0
                for j in range(n_classes):
                    if prob_[i,j]>max_prob:
                        max_prob = prob_[i,j]
                        prediction = j
                predict_[i,prediction] = 1
            """
            ase_sum  = 0
            mase_sum = 0
            misc_ = 0
            misc  = []
            n_    = []
            conf_mat = []
            for i in range(n_classes):
                conf_mat.append(np.zeros(n_classes))
            y_ = np.ravel(y) # necessary because yt is a df with row keys
            for i in range(n_classes):
                misc.append(0)
                n_.append(0)
            for i in range(n_obs):
                for j in range(n_classes):
                    if y_[i] == nn.classes_[j]:
                        ase_sum += (1-prob_[i,j])*(1-prob_[i,j])
                        mase_sum += 1-prob_[i,j]
                        idx = j
                        n_[j] += 1
                    else:
                        ase_sum  += prob_[i,j]*prob_[i,j]
                        mase_sum += prob_[i,j]
                for j in range(n_classes):
                    if predict_[i] == nn.classes_[j]:
                            conf_mat[idx][j] += 1
                            continue
                if predict_[i] != nn.classes_[idx]:
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
                print("{:.<27s}{:>10s}".format('Target Activation', \
                                     nn.out_activation_))
                print("{:.<27s}{:10.4f}".format('Loss Function', \
                                     nn.loss_))
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
            if type(nn)==MLPClassifier:
                print("{:.<27s}{:10.4f}".format('Loss', nn.loss_))
            print("{:.<27s}{:10d}".format(\
                    'Total Misclassifications', tmisc))
            print("{:.<27s}{:9.1f}{:s}".format(\
                    'MISC (Misclassification)', misc_, '%'))
            if type(nn.classes_[0]) == str:
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
            
            if type(nn.classes_[0]) == str:
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
    
            cr = classification_report(y_, predict_, nn.classes_)
            print("\n",cr)
        
    def display_split_metrics(nn, Xt, yt, Xv, yv, target_names=None):
        if len(nn.classes_) <=2:
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
            if type(nn.classes_[0])==np.str_:
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
                n_neurons = 0
                nl = nn.n_layers_-2
                if nl>1:
                    for i in range(nl):
                        n_neurons += nn.hidden_layer_sizes[i]
                else:
                    n_neurons = nn.hidden_layer_sizes
                print("{:.<27s}{:11d}{:15d}".format('Neurons',
                                      n_neurons, n_neurons))
                print("{:.<27s}{:11d}{:15d}".format('Weights', 
                                     n_weights, n_weights))
                print("{:.<27s}{:11d}{:15d}".format('Iterations', 
                                      nn.n_iter_, nn.n_iter_))
                print("{:.<27s}{:>11s}{:>15s}".format('Hidden Layer Activation', 
                                      nn.activation, nn.activation))
                print("{:.<27s}{:>11s}{:>15s}".format('Target Activation', 
                                      nn.out_activation_, nn.out_activation_))
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
            # In the binary case, the classification report is incorrect
            #cr = classification_report(yv, predict_v, nn.classes_)
            #print("\n",cr)
            
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
            try:
                if len(nn.classes_) == 2:
                    raise RuntimeError("  Call to display_nominal_split_metrics "+\
                      "invalid.\n  Target is Binary.  "+\
                      "Use display_binary_split_metrics.\n")
                    sys.exit()
            except:
                raise RuntimeError("  Call to display_nominal_split_metrics "+\
                      "invalid.\n  Target is Binary.\n")
                
            try:        
                if len(nn.classes_) < 3:
                    raise RuntimeError("  Call to display_nominal_split_metrics "+\
                      "invalid.\n  Target has less than three classes.\n")
                    sys.exit()
            except:
                raise RuntimeError("  Call to display_nominal_split_metrics "+\
                      "invalid.\n  Target has less than three classes.\n")
                sys.exit()
            np_yt = np.ravel(yt)
            np_yv = np.ravel(yv)
            predict_t = nn.predict(Xt)
            predict_v = nn.predict(Xv)
            conf_mat_t = confusion_matrix(y_true=yt, y_pred=predict_t)
            conf_mat_v = confusion_matrix(y_true=yv, y_pred=predict_v)
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
                    if y_t[i] == nn.classes_[j]:
                        ase_sumt += (1-prob_t[i,j])*(1-prob_t[i,j])
                        mase_sumt += (1-prob_t[i,j])
                        idx = j
                    else:
                        ase_sumt += prob_t[i,j]*prob_t[i,j]
                        mase_sumt += prob_t[i,j]
                for j in range(n_classes):
                    if predict_t[i] == nn.classes_[j]:
                        conf_matt[idx][j] += 1
                        break
                n_t[idx] += 1
                if predict_t[i] != y_t[i]:
                    misc_t     += 1
                    misct[idx] += 1
                    
            for i in range(nv_obs):
                for j in range(n_classes):
                    if y_v[i] == nn.classes_[j]:
                        ase_sumv += (1-prob_v[i,j])*(1-prob_v[i,j])
                        mase_sumv += (1-prob_v[i,j])
                        idx = j
                    else:
                        ase_sumv += prob_v[i,j]*prob_v[i,j]
                        mase_sumv += prob_v[i,j]
                for j in range(n_classes):
                    if predict_v[i] == nn.classes_[j]:
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
            maset  = mase_sumt/(n_classes*nt_obs)
            masev  = mase_sumv/(n_classes*nt_obs)
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
            if type(nn.classes_[0])==str:
                classes_ = nn.classes_
            else:
                for i in range(n_classes):
                    classes_.append(str(int(nn.classes_[i])))
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
                print("{:>7s}{:<3.0f}".format('Class ', classes_[i]), end="")
            print("")
            for i in range(n_classes):
                print("{:s}{:.<6.0f}".format('Class ', classes_[i]), end="")
                for j in range(n_classes):
                    print("{:>10d}".format(conf_mat_v[i][j]), end="")
                print("")
            cv = classification_report(yv, predict_v, target_names)
            print("\nValidation \nMetrics:\n",cv)