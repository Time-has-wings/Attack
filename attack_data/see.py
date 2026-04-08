import sys
import os
sys.path.append(os.path.abspath(".."))

import pickle
import torch
from codes.model import CNN

# ---------------------
# 1. 加载数据
# ---------------------
with open("correct_1k.pkl", "rb") as f:
    data = pickle.load(f)

images = data[0]  # (N,1,28,28)
labels = data[1]  # (N,10)

# ---------------------
# 2. 加载训练好的模型
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load("../model/ep_1_devacc_87.63_.pt", map_location=device))  # 这个存在7.2%的错误
# model.load_state_dict(torch.load("../model/ep_20_devacc_90.39_.pt", map_location=device))  # 这个存在3.6%的错误
# model.load_state_dict(torch.load("../model/cnn.ckpt", map_location=device)) # 这个是全对的

model.to(device)
model.eval()

# ---------------------
# Fashion MNIST 标签名
# ---------------------
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# =====================
# 统计1000张对抗样本的预测准确率
# =====================
correct = 0
total = len(images)

for i in range(total):
    img = images[i]
    true_label = labels[i]

    img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    if pred == true_label:
        correct += 1

print(f"总样本数: {total}")
print(f"预测正确数: {correct}")
print(f"预测错误数: {total - correct}")
print(f"准确率: {correct / total * 100:.2f}%")