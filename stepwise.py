#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:24:46 2019

@author: EJones
"""

import pandas as pd
import statsmodels.api as sm

def forward_regression(X, y,
                       pval_in=0.1,
                       verbose=False):
    initial_list = []
    attributes = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(attributes))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(\
                    pd.DataFrame(X[attributes+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < pval_in:
            best_feature = new_pval.idxmin()
            attributes.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(\
                      best_feature, best_pval))

        if not changed:
            break

    return attributes

def backward_regression(X, y,
                           pval_out=0.2,
                           verbose=False):
    attributes=list(X.columns)
    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(\
                          pd.DataFrame(X[attributes]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            attributes.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(\
                      worst_feature, worst_pval))
        if not changed:
            break
    return attributes
