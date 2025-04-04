import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


cali = fetch_california_housing()
X = pd.DataFrame(cali.data, columns = cali.feature_names)
y = pd.Series(cali.target)

X_train, X_test , y_train, y_test = train_test_split( X ,y, test_size = 0.2, random_state = 42)

model = LinearRegression() 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.scatter(y_test, y_pred)
plt.xlabel (' Giá trị thực tế')
plt.ylabel('Giá trị dự đoán ')
plt.title("Biểu đồ dự đoán giá nhà tại Cali")
plt.show()
