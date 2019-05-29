#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:10:13 2019

@author: EJones
"""

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