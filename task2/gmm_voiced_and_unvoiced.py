from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from evaluate import get_metrics
from sklearn.preprocessing import StandardScaler


data = np.load('task2/dev_features_and_labels/length_4096_step_2048.npz')

X = data['features']
y = data['labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_speech = np.sum(y_train == 1)
n_nonspeech = np.sum(y_train == 0)
total = n_speech + n_nonspeech
prior_speech = n_speech / total if total > 0 else 0.5  # 默认0.5
prior_nonspeech = n_nonspeech / total if total > 0 else 0.5
# 确保先验非零
prior_speech = max(prior_speech, 1e-10)
prior_nonspeech = max(prior_nonspeech, 1e-10)

X_speech = X_train[y_train == 1]
X_nonspeech = X_train[y_train == 0]

gmm_speech = GaussianMixture(n_components=64, random_state=42, covariance_type='diag', max_iter=5000, verbose=1, tol=1e-12)
gmm_speech.fit(X_speech)
gmm_nonspeech = GaussianMixture(n_components=64, random_state=42, covariance_type='diag', max_iter=5000, verbose=1, tol=1e-12)
gmm_nonspeech.fit(X_nonspeech)

# 计算对数似然
loglik_speech = gmm_speech.score_samples(X_test)
loglik_nonspeech = gmm_nonspeech.score_samples(X_test)

print("对数似然（语音）最小值:", np.min(loglik_speech))
print("对数似然（非语音）最小值:", np.min(loglik_nonspeech))

# 对数域计算后验概率，增强数值稳定性
def logsumexp(a, b):
    """稳定计算 log(exp(a) + exp(b))"""
    max_val = np.maximum(a, b)
    return max_val + np.log(np.exp(a - max_val) + np.exp(b - max_val))

# 计算对数证据：log(p(x)) = log(p(x|speech)P(speech) + p(x|nonspeech)P(nonspeech))
log_prior_speech = np.log(prior_speech)
log_prior_nonspeech = np.log(prior_nonspeech)
log_evidence = logsumexp(loglik_speech + log_prior_speech, loglik_nonspeech + log_prior_nonspeech)

# 计算对数后验概率
log_prob_speech = loglik_speech + log_prior_speech - log_evidence
log_prob_nonspeech = loglik_nonspeech + log_prior_nonspeech - log_evidence

# 转换为概率
prob_speech = np.exp(log_prob_speech)
prob_nonspeech = np.exp(log_prob_nonspeech)

# 处理NaN和无效值
prob_speech = np.nan_to_num(prob_speech, nan=0.5, posinf=1.0, neginf=0.0)
prob_nonspeech = np.nan_to_num(prob_nonspeech, nan=0.5, posinf=0.0, neginf=1.0)

# 归一化确保概率和为1
sum_probs = prob_speech + prob_nonspeech
sum_probs = np.where(sum_probs == 0, 1.0, sum_probs)  # 避免除零
prob_speech = prob_speech / sum_probs
prob_nonspeech = prob_nonspeech / sum_probs

# 步骤8：平滑概率（参考论文M/N规则）
def smooth_probabilities(probs, window_size=24):
    smoothed = probs.copy()
    half_window = window_size // 2
    for i in range(half_window, len(probs) - half_window):
        window = probs[i - half_window:i + half_window]
        valid_window = window[~np.isnan(window)]
        smoothed[i] = np.mean(valid_window) if len(valid_window) > 0 else 0.5
    return smoothed

smoothed_prob_speech = smooth_probabilities(prob_speech)
smoothed_prob_nonspeech = 1 - smoothed_prob_speech  # 确保和为1

if np.any(np.isnan(smoothed_prob_speech)) or np.any(np.isnan(smoothed_prob_nonspeech)):
    raise ValueError("平滑后的概率包含NaN")

threshold = 0.5
binary_labels = (smoothed_prob_speech > threshold).astype(int)
print("预测标签:", binary_labels[:5])

auc, eer = get_metrics(prediction=smoothed_prob_speech, label=y_test)

print(auc)
print(eer)

acc = accuracy_score(y_test, binary_labels)
print(acc)
