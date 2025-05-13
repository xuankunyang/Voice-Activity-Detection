import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from evaluate import get_metrics


data = np.load('task1/features_and_labels/length_2048_step_1024.npz')

X = data['features']
y = data['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(C=1, max_iter=500, penalty='l2')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
outputs = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)

auc, eer = get_metrics(prediction=outputs[:, 1], label=y_test)

print('ACC: ', accuracy)

print('AUC: ', auc)
print('EER: ', eer)

