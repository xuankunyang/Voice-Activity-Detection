import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from GMM import GaussianMixtureModel
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

# 使用sklearn生成数据集，生成两类数据点，6个维度
X, y = make_classification(n_samples=5000, n_features=6, n_classes=2, random_state=42, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gmm = GaussianMixtureModel(n_compoents=2, max_iter=2000, tol=1e-12, init_method="kmeans", reg_covar=1e-8)
gmm_1 = GaussianMixture(n_components=2, max_iter=2000, init_params='kmeans', tol=1e-4, covariance_type='full', reg_covar=1e-8)

gmm.fit(X_train)
gmm_1.fit(X_train)

predictions = gmm.predict(X_test)
predictions_1 = gmm_1.predict(X_test)

log_likelihood = gmm.score(X_test)
print(f"Log Likelihood on test set: {log_likelihood}")

log_likelihood_1 = gmm_1.score(X_test)
print(f"Log Likelihood on test set: {log_likelihood_1}")

acc = accuracy_score(y_test, predictions)
print(acc)

acc_1 = accuracy_score(y_test, predictions_1)
print(acc_1)
