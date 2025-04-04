import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

model = LogisticRegression(max_iter=5000)  # Tăng số vòng lặp để tránh lỗi hội tụ
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = confusion_matrix(y_test, y_pred)
r2 = classification_report(y_test, y_pred)
y_prob = model.predict_proba(X_test)[:, 1]  # Xác suất cho lớp 1

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(y_train, bins=2, alpha=0.7, label='Tập huấn luyện')
plt.hist(y_test, bins=2, alpha=0.7, label='Tập kiểm tra')
plt.xticks([0, 1], ['Lành tính', 'Ác tính'])
plt.xlabel("Loại Khối U")
plt.ylabel("Số Lượng Mẫu")
plt.legend()
plt.title("Phân bố dữ liệu theo hai lớp")

# 4.2 Ma trận nhầm lẫn
plt.subplot(1, 3, 2)
sns.heatmap(mse, annot=True, fmt='d', cmap='Blues', xticklabels=['Lành tính', 'Ác tính'], yticklabels=['Lành tính', 'Ác tính'])
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn")

# 4.3 Đường cong ROC
plt.subplot(1, 3, 3)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--')  # Đường chéo ngẫu nhiên
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.show()

# 5. In báo cáo đánh giá
print("Classification Report:\n", r2)