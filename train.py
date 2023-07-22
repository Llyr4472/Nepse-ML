import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Step 1: Data Preprocessing
data = pd.read_csv('data.csv')

# Step 2: Feature Engineering
company = 'AKJCL'
data = data[data['symbol'] == company]
selected_features = ['openPrice', 'highPrice', 'lowPrice', 'closePrice','securityId',
                     'highPrice','lowPrice','closePrice','totalTradedQuantity','totalTradedValue',
                     'fiftyTwoWeekHigh','fiftyTwoWeekLow','day_of_week_cos','previousDayClosePrice',
                     'lastUpdatedPrice','totalTrades','averageTradedPrice','marketCapitalization',
                     'year','month','day','day_of_week','month_sin','month_cos','day_of_week_sin',
                     ]
data = data[selected_features]

# Step 3: Train/Test Split
train_size = int(0.8 * len(data))
train_data, test_data = data[:train_size], data[train_size:]

# Step 4: Data Scaling
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_data['closePrice'].values.reshape(-1,1))
test_scaled = scaler.transform(test_data['closePrice'].values.reshape(-1,1))

# Step 5: Model Architecture
X_train, y_train = [], []
prediction_days = 60

for i in range(prediction_days, len(train_scaled)):
    X_train.append(train_scaled[i - prediction_days:i,0])
    y_train.append(train_scaled[i, 0])  # Closing price is at index 9

X_train, y_train = np.array(X_train), np.array(y_train)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)

# Step 6: Model Training
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Step 7: Model Evaluation
X_test, y_test = [], []

for i in range(prediction_days, len(test_scaled)):
    X_test.append(test_scaled[i - prediction_days:i])
    y_test.append(test_scaled[i, 3])  # Closing price is at index 3

X_test, y_test = np.array(X_test), np.array(y_test)

predicted_prices_scaled = model.predict(X_test)
predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((prediction_days, 1)), predicted_prices_scaled), axis=0))[:, 0]

# Step 8: Visualization
plt.plot(test_data['closePrice'], label='Actual')
plt.plot(predicted_prices, label='Predicted', color='green')
plt.title(f"{company} Share Prices Prediction")
plt.xlabel('Time')
plt.ylabel("Prices")
plt.legend()
plt.show()
