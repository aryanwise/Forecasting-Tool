import pandas as pd
import numpy as np

# Lags, rolling stats, & spike momentum

class FeatureBuilder:
    def __init__(self, target_col='Incoming Calls'):
        self.target_col = target_col
        self.feature_cols = []

    def create_features(self, df):
        df_feat = df.copy()
        target = self.target_col

        # Standard Lags & Rolling Means
        df_feat['lag_1'] = df_feat[target].shift(1)
        df_feat['lag_2'] = df_feat[target].shift(2)
        df_feat['lag_7'] = df_feat[target].shift(7)
        df_feat['rolling_mean_3'] = df_feat[target].shift(1).rolling(3).mean()
        df_feat['rolling_mean_7'] = df_feat[target].shift(1).rolling(7).mean()

        # Spike Momentum Indicators
        df_feat['rolling_max_3'] = df_feat[target].shift(1).rolling(3).max()
        df_feat['rolling_std_7'] = df_feat[target].shift(1).rolling(7).std()

        # Surge Multipliers
        df_feat['spike_intensity'] = df_feat['lag_1'] / (df_feat['rolling_mean_7'] + 1e-5)
        df_feat['is_spike_regime'] = (df_feat['spike_intensity'] > 1.5).astype(int)

        # Drop NaNs created by lag operations
        df_clean = df_feat.dropna().reset_index(drop=True)

        # Save feature column names
        exog_cols = [c for c in ['Talk Duration (AVG)', 'Waiting Time (AVG)', 'Service Level (20 Seconds)'] if c in df_clean.columns]
        self.feature_cols = [
            'lag_1', 'lag_2', 'lag_7', 'rolling_mean_3', 'rolling_mean_7',
            'rolling_max_3', 'rolling_std_7', 'spike_intensity', 'is_spike_regime'
        ] + exog_cols

        return df_clean[self.feature_cols].astype(float), df_clean[target].astype(float)
