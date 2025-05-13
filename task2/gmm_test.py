import numpy as np
from sklearn.mixture import GaussianMixture
from evaluate import get_metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = np.load('task2/training_data_features_and_labels/length_2048_step_1024.npz')

X = data['features']
y = data['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

gmm = GaussianMixture(n_components=16, max_iter=100, verbose=1, tol=1e-6, init_params='kmeans', reg_covar=1e-6)
gmm.fit(X_train)

preds = gmm.predict(X_test)

auc, eer = get_metrics(prediction=preds, label=y_test)
print(auc)
print(eer)
acc = accuracy_score(y_true=y_test, y_pred=preds)
print(acc)

fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=preds, pos_label=1)
auc = metrics.auc(fpr, tpr)

print("False Positive Rate:", fpr)
print("True Positive Rate:", tpr)
print("Thresholds:", thresholds)

# 绘制 ROC 曲线
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
