import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


#load data
company = 'AKJCL'
data = pd.read_csv('data.csv')
data = data[data['symbol']==company]

#prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data =  scaler.fit_transform(data['closePrice'].values.reshape(-1,1))

prediction_days = 60
x_train = []
y_train = []

for x in range(prediction_days,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Build the model
model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(prediction_days,1)))
model.add(Dropout(0.2)) 
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2)) 
model.add(LSTM(units=50))
model.add(Dropout(0.2)) 
model.add(Dense(units=1)) #Prediction

model.compile(
    optimizer='adam',
    loss = 'mean_squared_error',
    )
model.fit(x_train,y_train,epochs=25,batch_size=32)

'''Test the model accuracy on existing data'''
test_data = pd.read_csv('test_data.csv')
test_data = test_data[test_data['symbol']==company]
actual_prices = test_data['closePrice'].values

total_dataset =  pd.concat((data['closePrice'],test_data['closePrice']),axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs= scaler.transform(model_inputs)

#Make Prediction on test data
x_test = []

for x in range(prediction_days,len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_prices = model.predict(x_test)
predicted_prices = scalar.inverse_transform(predicted_prices)

#plot the predictions
plt.plot(actual_prices,color='black',lable="Actual")
plt.plot(actual_prices,color='green',lable="Predicted")
plt.title(f"{company} Share Prices")
plt.xlabel('Time')
plt.ylabel("Prices")
plt.legend()
plt.show