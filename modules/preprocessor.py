import pandas as pd
import numpy as np

# Data cleaning & string parsing

class DataPreprocessor:
    def __init__(self, target_col='Incoming Calls'):
        self.target_col = target_col

    @staticmethod
    def parse_time_to_seconds(time_val):
        if pd.isna(time_val): 
            return 0.0
        val_str = str(time_val).strip()
        try:
            parts = val_str.split(':')
            if len(parts) == 3:
                return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2])))
            elif len(parts) == 2:
                return float(int(parts[0]) * 60 + int(float(parts[1])))
        except Exception:
            pass
        return 0.0

    def fit_transform(self, df):
        df_clean = df.copy()

        # Clean percentage columns
        for col in ['Answer Rate', 'Service Level (20 Seconds)']:
            if col in df_clean.columns:
                df_clean[col] = (
                    df_clean[col]
                    .astype(str)
                    .str.rstrip('%')
                    .apply(lambda x: pd.to_numeric(x, errors='coerce'))
                )

        # Clean duration columns
        for col in ['Answer Speed (AVG)', 'Talk Duration (AVG)', 'Waiting Time (AVG)']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.parse_time_to_seconds)

        # Clean numeric counts
        for col in [self.target_col, 'Answered Calls', 'Abandoned Calls']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        return df_clean.fillna(0)
