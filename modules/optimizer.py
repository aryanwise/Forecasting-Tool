import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Optimus ensemble blender (WMAPE)

class OptimusEnsemble:
    def __init__(self, zero_threshold=1e-3):
        self.zero_threshold = zero_threshold

    def optimize_weights(self, pred_dict, y_true):
        model_names = list(pred_dict.keys())
        pred_matrix = np.array([pred_dict[m] for m in model_names]).T
        y_true_arr = np.array(y_true, dtype=float)
        n_models = len(model_names)

        def wmape_loss(w):
            ensemble = pred_matrix.dot(w)
            total_actual = np.sum(np.abs(y_true_arr))
            if total_actual == 0:
                return 0.0
            return (np.sum(np.abs(y_true_arr - ensemble)) / total_actual) * 100.0

        x0 = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0)] * n_models
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

        res = minimize(wmape_loss, x0, method="SLSQP", bounds=bounds, constraints=cons)

        if not res.success or np.isnan(res.x).any():
            w = x0
        else:
            w = res.x

        w = np.where(w < self.zero_threshold, 0.0, w)
        w = w / w.sum() if w.sum() > 0 else x0

        return dict(zip(model_names, np.round(w, 4)))

    @staticmethod
    def evaluate(y_true, pred_dict, ensemble_pred):
        def calc_metrics(yt, yp):
            wmape = (np.sum(np.abs(yt - yp)) / np.sum(yt)) * 100.0
            rmse = np.sqrt(mean_squared_error(yt, yp))
            mape = mean_absolute_percentage_error(yt, yp) * 100.0
            return round(wmape, 2), round(rmse, 2), round(mape, 2)

        leaderboard = []
        for name, preds in pred_dict.items():
            wm, rm, mp = calc_metrics(y_true, preds)
            leaderboard.append({"Model": name, "WMAPE (%)": wm, "RMSE": rm, "Standard MAPE (%)": mp})

        e_wm, e_rm, e_mp = calc_metrics(y_true, ensemble_pred)
        leaderboard.append({"Model": "⭐ Optimized Ensemble", "WMAPE (%)": e_wm, "RMSE": e_rm, "Standard MAPE (%)": e_mp})

        return pd.DataFrame(leaderboard).sort_values(by="WMAPE (%)")
