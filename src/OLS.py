import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer


class SMOLSWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """

    def __init__(self, formula=None):
        self.formula = formula
        self.target_name = formula.split('~')[0].strip()

    def fit(self, X, y):
        data = X.reset_index(drop=True)
        data[self.target_name] = y.reset_index(drop=True)
        self.model_ = sm.OLS.from_formula(formula=self.formula, data=data)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X):
        return self.results_.predict(exog=X)

    def get_summary(self):
        return self.results_.summary()


class PandasSimpleImputer(SimpleImputer):
    """ A wrapper around `SimpleImputer` to return data frames with columns """

    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X)

    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)