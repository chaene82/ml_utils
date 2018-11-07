# -*- coding: utf-8 -*-

"""Main module."""

import numpy as np

def print_features_importances(model, labels):
    """
    This function is printing a feature list sorted by there importance. 
    
    Args:
        model       : a scikit-learn model
        lables      : a list of column-lables (X.columns)
     
    Returns:
        -             
    
    """
    importances = model.feature_importances_

    print("Feature ranking:")
    indices = np.argsort(importances)[::-1]

    for f in range(len(importances)):
        print("%d. %s (%f)" % (f + 1, labels[f],  importances[indices[f]]))