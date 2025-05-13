from dnn_testing import dnn_testing
from dnn_training import dnn_training
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


frame_size_list = [float(0.020), float(0.032), float(0.064), float(0.128), float(0.256), float(0.020), float(0.032), float(0.064), float(0.128), float(0.256)]
frame_shift_list = [float(0.005), float(0.008), float(0.016), float(0.032), float(0.064), float(0.010), float(0.016), float(0.032), float(0.064), float(0.128)]

# frame_size_list = [float(0.128)]
# frame_shift_list = [float(0.064)]

# frame_size_list = [float(0.032)]
# frame_shift_list = [float(0.008)]

# frame_size_list = [float(0.020), float(0.064), float(0.128), float(0.256), float(0.020), float(0.032), float(0.064), float(0.128)]
# frame_shift_list = [float(0.005), float(0.016), float(0.032), float(0.064), float(0.010), float(0.016), float(0.032), float(0.064)]

all_prob = []
all_label = []

for frame_size, frame_shift in tqdm(zip(frame_size_list, frame_shift_list), desc="Processing different pairs", unit="pair"):
    frame_length = int(frame_size * 16000)
    frame_step = int(frame_shift * 16000)
    # dnn_training(frame_length=frame_length, frame_step=frame_step, batch_size=32, lr=0.001, num_epochs=10)
    label, prob = dnn_testing(frame_length=frame_length, frame_step=frame_step)
    all_label.append(label)
    all_prob.append(prob)

fig, axes = plt.subplots(5, 2, figsize=(6, 25))  # 设置总图大小
axes = axes.flatten()

for i, prob in enumerate(all_prob):
    y_test = all_label[i]

    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=prob, pos_label=1)

    axes[i].set_aspect('equal')
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(0, 1)

    axes[i].plot(fpr, tpr, color='blue', lw=0.5, label='ROC curve')
    axes[i].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random classifier', linewidth=0.5)

    axes[i].set_xlabel('False Positive Rate')
    axes[i].set_ylabel('True Positive Rate')
    axes[i].set_title(f'Frame Length:{int(16000 * frame_size_list[i])} Frame Step:{int(16000 * frame_shift_list[i])}', fontsize=10)

handles, labels = axes[0].get_legend_handles_labels()

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.subplots_adjust(hspace=0.1, wspace=0.1) 

plt.show()