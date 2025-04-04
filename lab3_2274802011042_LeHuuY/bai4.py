from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


iris = datasets.load_iris()

X = iris.data  

y = iris.target  

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

pca = PCA(n_components=2)#Giảm số chiều từ 4 xuống 2, giúp dễ dàng trực quan hóa.
#PCA tìm ra 2 hướng quan trọng nhất trong dữ liệu giúp giữ lại thông tin nhiều nhất.
X_pca = pca.fit_transform(X)

labels = kmeans.predict(X)
import matplotlib.pyplot as plt

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')#Vẽ dữ liệu trên không gian 2D sau khi giảm chiều.
#Mỗi điểm dữ liệu được tô màu theo nhãn cụm (labels).
plt.title('Phân cụm dữ liệu Iris bằng KMeans')
plt.xlabel('Thành phần chính 1')
plt.ylabel('Thành phần chính 2')
plt.colorbar(label='Nhãn phân cụm')
plt.show()

#PCA là một kỹ thuật giảm chiều dữ liệu giúp giữ lại thông tin quan trọng nhất của tập dữ liệu bằng cách biến đổi các đặc trưng gốc thành một tập hợp các thành phần chính (Principal Components).
#Mục tiêu chính: Giảm số chiều của dữ liệu nhưng vẫn giữ được phần lớn thông tin.
#PCA giúp loại bỏ nhiễu và tối ưu hóa hiệu suất của mô hình học máy.