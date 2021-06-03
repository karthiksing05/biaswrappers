# Bias Wrappers

Wrappers for standard multioutput machine learning models that apply progressive calibration to training to produce better testing results, with a bias factor. Used mainly to combat bias on seemingly random/biased data. Default models are Linear Regression with Gradient Descent (for regression) and a standard Naive Bayes (for classification), however, you can input your own machine learning models with the model param.

## Fixes

Fixed an issue that happened when importing the package.

## Instructions

1. Install the package with pip:
```
pip install biaswrappers
```

2. Python Quickstart:
```python

# Import Classifier/Regressor
from biaswrappers import classifier, regressor
from biaswrappers.baseline_tests import test_classification, test_regression

# Initialize classifier/regressor and...
# Specify a model class with a fit and predict method as a param.
my_clf = classifier.BiasClassifier() 
my_regressor = regressor.BiasRegressor()

# Use the baseline_tests module for comparable results
test_classification(model=my_regressor) # No return values, just prints results
test_regression(model=my_regressor) # No return values, just prints results

```