#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ml_utils` package."""

import pytest


from ml_utils.features import print_features_importances
from sklearn.externals import joblib

model = joblib.load('C:/Users/haenec/python/ds_early_vip_machinelearning/models/EarlyVIP_Predicitions_RFClassifier.sav')


@pytest.fixture
def test_features_importance():
    model = joblib.load('model/test_model.sav')
    labels = ['one','two','tree','four','five', 'six']
    print_features_importances(model, labels)
