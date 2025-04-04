import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# 1. Tạo DataFrame từ dữ liệu cung cấp
data = pd.DataFrame([
    [1, 39, 4, 0, 0, 0, 0, 0, 0, 195, 106, 70, 26.97, 80, 77, 0],
    [0, 46, 2, 0, 0, 0, 0, 0, 0, 250, 121, 81, 28.73, 95, 76, 0],
    [1, 48, 1, 1, 20, 0, 0, 0, 0, 245, 127.5, 80, 25.34, 75, 70, 0],
    [0, 61, 3, 1, 30, 0, 0, 1, 0, 225, 150, 95, 28.58, 65, 103, 1],
    [0, 46, 3, 1, 23, 0, 0, 0, 0, 285, 130, 84, 23.1, 85, 85, 0],
    [0, 43, 2, 0, 0, 0, 0, 1, 0, 228, 180, 110, 30.3, 77, 99, 0],
    [0, 63, 1, 0, 0, 0, 0, 0, 0, 205, 138, 71, 33.11, 60, 85, 1],
    [0, 45, 2, 1, 20, 0, 0, 0, 0, 313, 100, 71, 21.68, 79, 78, 0],
    [1, 52, 1, 0, 0, 0, 0, 1, 0, 260, 141.5, 89, 26.36, 76, 79, 0],
    [1, 43, 1, 1, 30, 0, 0, 1, 0, 225, 162, 107, 23.61, 93, 88, 0],
    # Thêm các dòng khác... (đã bị cắt trong tài liệu, chỉ lấy 10 dòng đầu để minh họa)
    # Nếu muốn dùng toàn bộ dữ liệu, bạn cần nhập đầy đủ hoặc tải từ file
], columns=['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 
            'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD'])

# 2. Xử lý giá trị NA (nếu có)
data = data.replace('NA', np.nan).dropna()

# 3. Tách đặc trưng (X) và nhãn (y)
X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5. Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Tạo và huấn luyện mô hình Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Dự đoán
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Xác suất cho lớp 1 (có nguy cơ)

# 8. Đánh giá mô hình
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 9. Trực quan hóa
plt.figure(figsize=(15, 5))

# Ma trận nhầm lẫn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Không nguy cơ', 'Có nguy cơ'], 
            yticklabels=['Không nguy cơ', 'Có nguy cơ'])
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn")

# Đường cong ROC
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--')  # Đường chéo ngẫu nhiên
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Đường cong ROC")
plt.legend()

plt.tight_layout()
plt.show()

# 10. In báo cáo
print("Classification Report:\n", report)