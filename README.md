# Bias Wrappers

Wrappers for standard multioutput machine learning regressors that apply regularization to training to produce better testing results, with a bias factor. Used mainly to combat bias on seemingly random/biased data. Default models are Linear Regression, however, you can input your own machine learning models with the model param.

BiasRegressorC1 uses a progressive regularization method to add a penalty to data, to prevent overfitting or underfitting due to noise via bias (explicit regularization).

BiasRegressorC2 uses another regression model to generate features that prevent overfitting or underfitting (implicit regularization).

## Fixes

### 0.4.0
Made many fixes to original BiasRegressor, now BiasRegressorC1, and added a second one incorporating machine learning regularization through generated features.

### 0.3.1
Fixed Array/List Contradiction in regression, removed classifier for code compatibility, and removed a few print statements.

Removed classifier because the formula used only benefits regression problems.

## Instructions

1. Install the package with pip:
```
pip install biaswrappers
```

2. Python Quickstart:
```python

# Import one of the regressors from the package, regressor
from biaswrappers import regressor
from biaswrappers.baseline_tests import test_regression

# Initialize cregressor and...
# Specify a model class (or multiple, for C2) with a fit and predict method as a param.
my_regressor = regressor.BiasRegressorC1()

# Look at the baseline_tests module for easy tests
test_regression(model=my_regressor) # No return values, just prints results

```