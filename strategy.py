import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
data=yf.download("AAPL",period='1y')
data=data[['Close']]

for i in range(1,6):
    data[f'Close_lag_{i}'] = data['Close'].shift(i)

data = data.dropna()

x=data[[f'Close_lag_{i}'for i in range(1,6)]]
y=data['Close']

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=False,test_size=0.2)

model= LinearRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

mse= mean_squared_error( y_test, y_pred)
print(f"mean squared error:{mse:.2f}")

plt.figure(figsize=(10,5))
plt.gca().set_facecolor("black")

plt.plot( y_test.index, y_test.values, label='Actual Price')
plt.plot( y_test.index, y_pred, label='Predicted Price')
plt.xlabel("Date")
plt.ylabel("Stock Price in USD")
plt.title("AAPLE Stock Price Prediction")
plt.legend()
plt.grid(True)
plt.show()