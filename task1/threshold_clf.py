import numpy as np
from evaluate import get_metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import metrics

data = np.load('task1/features_and_labels/length_2048_step_512.npz')

features = data['features']
labels = data['labels']
num_subbands = data['num_subbands']

if num_subbands != 6:
    raise TypeError("Please check the number of subbands!")


class thresholds_clf:
    def __init__(self, thresholds: np.array):
        self.thresholds = thresholds

    def predict(self, X: np.array):
        if X.shape[1] != self.thresholds.shape[0]:
            raise IndexError("The number of features and the dimension of thresholds differs.")

        preds = np.all(X >= self.thresholds, axis=1).astype(int)

        return preds
    

ths = np.array([5.67380910e-06, 3.71428571e-01, 7.96785543e-03, 6.77966102e+01, 
                7.34346086e-03, 7.74812137e-03, 5.17774856e-03, 1.93879365e-03, 
                4.46081674e-04, 2.29604094e-04])

bigger_ones = [0, 2, 4, 5, 6, 7, 8, 9]

clf = thresholds_clf(ths)
preds = clf.predict(features)

auc, eer = get_metrics(preds, labels)

acc = accuracy_score(labels, preds)

print("AUC: ", auc)
print("ACC:", acc)
print("EER:", eer)

fpr, tpr, thresholds = metrics.roc_curve(y_true=labels, y_score=preds, pos_label=1)
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

