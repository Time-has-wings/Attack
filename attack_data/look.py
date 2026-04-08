import pickle
import matplotlib.pyplot as plt

# 加载文件
with open("correct_1k.pkl", "rb") as f:
    data = pickle.load(f)

images = data[0]  # 所有图片 (N, 1, 28, 28)
labels = data[1]  # 所有标签 (N, 10)

# 取第一张图
img = images[0]
label = labels[0]

print(img.shape)   # 输出 (1, 28, 28)
print(label.shape) # 输出 (1, 10)
print(label)
plt.imshow(img.reshape([28, 28]))
plt.show()

print(len(images))