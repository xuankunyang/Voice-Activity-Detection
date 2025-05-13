import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from evaluate import get_metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


data = np.load('task1/features_and_labels/length_512_step_128.npz')

X = data['features']
y = data['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(probability=True, verbose=1, kernel='linear',  C=0.5, max_iter=1000)

# model = LinearSVC(C=0.5, penalty='l2', verbose=1, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# y_probs = model._predict_proba_lr(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

auc, eer = get_metrics(prediction=y_probs[:, 1], label=y_test)

print('ACC: ', accuracy)

print('AUC: ', auc)
print('EER: ', eer)

