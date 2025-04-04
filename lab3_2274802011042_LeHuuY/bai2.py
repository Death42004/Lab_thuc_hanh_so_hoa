import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

file_path = "1.xlsx"  # Đường dẫn file Excel
df = pd.read_excel(file_path, engine="openpyxl")  # Đọc file Excel


print(" Dữ liệu ban đầu:")
print(df.head())  # Hiển thị 5 dòng đầu tiên
print("\n🔍 Kích thước dữ liệu:", df.shape)
print("\n🔍 Tên các cột:", df.columns)


if df.shape[1] == 1:  
    df = df.iloc[:, 0].str.split(",", expand=True)  # Tách dữ liệu thành 2 cột
    df.columns = ["X", "y"]  # Đặt tên cột
    df = df.astype(float)  # Chuyển về kiểu số


print("\n🔍 Số lượng giá trị NaN trước khi xử lý:")
print(df.isnull().sum())  # Kiểm tra số lượng NaN

df = df.dropna()  # Xóa hàng chứa giá trị NaN


print("\n Dữ liệu sau khi xử lý NaN:")
print(df.head())


X = df.iloc[:, 0].values.reshape(-1, 1)  
y = df.iloc[:, 1].values  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\n MSE (Mean Squared Error): {mse:.4f}')
print(f' R² Score: {r2:.4f}')

#  Bước 10: Vẽ biểu đồ
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')  # Chấm xanh: dữ liệu thực tế
plt.plot(X_test, y_pred, color='green', linewidth=2, label='Prediction')  # Đường hồi quy
plt.xlabel('Study Hours')  # Nhãn trục X
plt.ylabel('Exam Score')  # Nhãn trục Y
plt.title("Linear Regression - Predicting Exam Scores")  # Tiêu đề biểu đồ
plt.legend()
plt.grid(True)
plt.show()
