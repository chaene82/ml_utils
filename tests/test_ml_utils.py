#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ml_utils` package."""

import pytest


from ml_utils import features
from ml_utils.features import print_features_importances
from sklearn.externals import joblib

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')

def test_features_importance():
    model = joblib.load('model/test_model.sav')
    labels = ['one','two','tree','four','five', 'six']
    print_features_importances(model, labels)
    assert 2 >= 1
