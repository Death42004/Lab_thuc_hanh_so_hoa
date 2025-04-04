import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.preprocessing import StandardScaler

# 1. Đọc dữ liệu
df = pd.read_csv("winequality-red.csv", sep=';', encoding='utf-8')

# Hiển thị thông tin
print(df.head())
print(df.info())
print(df.describe())

# 2.1 Kiểm tra giá trị thiếu
print(df.isnull().sum())  

# 2.2 Vẽ histogram cho tất cả các biến
df.hist(bins=20, figsize=(20, 25))
plt.tight_layout()
plt.show()

# 2.3 Vẽ boxplot cho từng biến
plt.figure(figsize=(12, 66))
for i, col in enumerate(df.columns):
    plt.subplot(len(df.columns), 1, i + 1)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot của {col}")
plt.tight_layout()
plt.show()

# 2.4 Vẽ Scatterplot cho "alcohol" và "quality"
sns.scatterplot(x="alcohol", y="quality", data=df)
plt.title("Mối quan hệ giữa Alcohol và Quality")
plt.show()

# 3. Vẽ Heatmap hệ số tương quan
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.show()

# 4.1 Xáo trộn dữ liệu
np.random.seed(42)
tron = np.random.permutation(len(df))
df_shuffled = df.iloc[tron].reset_index(drop=True)

# 4.2 Chia dữ liệu với 80% huấn luyện
test_size = int(0.2 * len(df))  
train_indices = tron[:-test_size]
test_indices = tron[-test_size:]

# 4.3 Tách dữ liệu
train_df = df.iloc[train_indices]
test_df = df.iloc[test_indices]

# 4.4 Tách đặc trưng và nhãn
X_train = train_df.drop(columns=["quality"])
y_train = train_df["quality"]

X_test = test_df.drop(columns=["quality"])
y_test = test_df["quality"]

print("Train size:", X_train.shape, y_train.shape)
print("Test size:", X_test.shape, y_test.shape)

# 5. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 6. Ứng dụng mô hình hồi quy tuyến tính
from sklearn.linear_model import LinearRegression

# 6.1 Khởi tạo mô hình
model = LinearRegression()

# 6.2 Huấn luyện mô hình
model.fit(X_train_scaled, y_train)

# 6.3 Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_scaled)

# In ra 5 giá trị dự đoán đầu tiên
print("Dự đoán:", y_pred[:5])
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 6.1 Khởi tạo mô hình
model = LinearRegression()

# 6.2 Huấn luyện mô hình
model.fit(X_train_scaled, y_train)

# 6.3 Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_scaled)

# 7. Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)  # Tính MSE
r2 = r2_score(y_test, y_pred)  # Tính R² Score

# In kết quả
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")


