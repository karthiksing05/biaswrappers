from wrappers.regressor import PGCRegressor
from wrappers.classifier import PGCClassifier
from baseline_tests import test_regression, test_classification

pcgr = PGCRegressor()

test_regression(pcgr)

pgcc = PGCClassifier()

test_classification(pgcc)
