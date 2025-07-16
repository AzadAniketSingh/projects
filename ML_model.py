import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

df = pd.read_csv("hy.txt")
print(df)


X_train, X_test, y_train, y_test = train_test_split(df[['Experience']], df[['Salary']], test_size=0.3, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 ): {r2:.2f}")

plt.figure(figsize=(10,6))
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.scatter(X_test, y_pred, color='red', label='Predictions (Test)')
x_line = np.linspace(df['Experience'].min(), df['Experience'].max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='black', linewidth=2, label='Regression Line')

plt.xlabel('Experience (years)')
plt.ylabel('Salary ($)')
plt.title('Linear Regression with Train/Test Split')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)


plt.show()