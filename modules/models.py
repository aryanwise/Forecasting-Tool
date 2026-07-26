import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Linear-Tree Residual Hybrid and candidate models

class ResidualHybridModel:
    """Combines a Linear Baseline for extrapolation with a Tree Model for residual fitting."""
    def __init__(self, tree_model):
        self.base_model = LinearRegression()
        self.tree_model = tree_model

    def fit(self, X, y):
        # 1. Fit Linear Baseline
        self.base_model.fit(X, y)
        base_preds = self.base_model.predict(X)

        # 2. Fit Tree on Residuals
        residuals = y - base_preds
        self.tree_model.fit(X, residuals)
        return self

    def predict(self, X):
        base_preds = self.base_model.predict(X)
        residual_preds = self.tree_model.predict(X)
        return np.maximum(0, base_preds + residual_preds)


class ModelSuite:
    """Manages training and prediction for all candidate algorithms."""
    def __init__(self):
        self.models = {
            "Random Forest Res": ResidualHybridModel(RandomForestRegressor(n_estimators=100, random_state=42)),
            "Gradient Boosting Res": ResidualHybridModel(GradientBoostingRegressor(n_estimators=100, random_state=42)),
            "XGBoost Res": ResidualHybridModel(XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)),
            "LightGBM Res": ResidualHybridModel(LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1)),
            "Ridge Baseline": Ridge(alpha=10.0)
        }

    def train_and_predict(self, X_train, y_train, X_test):
        test_predictions = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            test_predictions[name] = model.predict(X_test)
        return test_predictions
