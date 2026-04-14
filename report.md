# 信息
- 姓名：林光明
- 学号：2501210620
- 仓库：https://github.com/Time-has-wings/Attack.git

# 白盒攻击实验结果
# 自己训练的模型
- 自己训练的模型在test的准确率为90.53%
- 模型架构：未修改原先的模型架构，在epoch为15下训练，保存best epoch下的模型，代码文件为codes/train_better.py
- 模型文件见model/cnn_best.ckpt
- 代码运行, `python train_better.py`

# 白盒攻击
- 在test集下的攻击率为59.4%
- 10组图片见images/whitebox/cnn_best
- 代码运行 `python whitebox_attack.py --ckpt_tag cnn_best`

# 黑盒攻击实验结果
## 提供的模型实验结果
- 成功率为
- 10组图片见images/blackbox/provided
## 自己训练的模型实验结果
- 成功率为
- 10组图片见images/blackbox/own
## 代码运行
- `python blackbox_attack.py`

# 对抗训练
## 结果
- 新分类器在test集的准确率89.9%，旧分类器在test集的准确率为90.53%
- 新分类器白盒攻击准确率为52.2%，旧分类器白盒攻击准确率为59.4%
- 新分类器黑盒攻击准确率为，旧分类器黑盒攻击准确率为
- 新分类器白盒攻击对抗样本图像见images/whitebox/cnn_adv，旧分类器白盒攻击对抗样本图像见images/whitebox/cnn_best
- 新分类器黑盒攻击对抗样本图像见附件images/blackbox/，旧分类器黑盒攻击对抗样本图像见images/blackbox/own
## 代码运行
- `python whitebox_attackpy --ckpt cnn_best_train`
- `python train_adv.py`
- `python whitebox_attack.py --ckpt cnn_best`
- `python whitebox_attack.py --ckpt cnn_adv`
- `python blackbox_attack.py --target adv`