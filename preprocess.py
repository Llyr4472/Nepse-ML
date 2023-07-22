import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

if 'businessDate' in data.columns:
    # Convert 'date' column to datetime type
    data['businessDate'] = pd.to_datetime(data['businessDate'])

    # Extract date features
    data['year'] = data['businessDate'].dt.year
    data['month'] = data['businessDate'].dt.month
    data['day'] = data['businessDate'].dt.day
    data['day_of_week'] = data['businessDate'].dt.dayofweek  # 0 for Monday, 6 for Sunday

    # Handle cyclical features (month and day of the week)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

    # Drop the original date column
    data = data.drop(columns=['businessDate'])

data.to_csv('data.csv')