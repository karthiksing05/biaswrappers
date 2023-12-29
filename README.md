# Bias Wrappers

Wrappers for standard multioutput machine learning regressors that apply regularization to training to produce better testing results, with a bias factor. Used mainly to combat bias on seemingly random/biased data. Default models are Linear Regression, however, you can input your own machine learning models with the model param.

BiasRegressorC1 uses a progressive regularization method to calculate a penalty to add to data, to prevent overfitting or underfitting due to noise via bias (explicit regularization).

BiasRegressorC2 uses another regression model to fit on incorrect predictions and correct answers, to identify patterns of overfitting or underfitting and arrive at a more correct answer (implicit regularization).

## Fixes

### 0.5.1
Redid a lot more of the C1 method to allow for more subtle changes in the right direction: it now closely matches the FakeWrapper but is a lil different. Also, removed features from the postModel of C2 to allow for more pure error learning. Also, renamed FakeWrapper to RandomWrapper.

### 0.5.0
Redid most of the C1 method to allow multi-output and specialized penalties through regularization. Also, created a "Fake Wrapper" that applies random penalties from 0 to 1 to test the C1 method fairly, proving the use of the specific penalties. Also, improved compatibility with native sklearn commands for metrics and model selection.

### 0.4.1
Small fixes regarding integration with some data, should work for all dimensions in the event of layered array for y_preds. Also, changed the dataset to use the default sklearn diabetes dataset instead of a Friedman problem, and rewrote some commands for clarity.

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