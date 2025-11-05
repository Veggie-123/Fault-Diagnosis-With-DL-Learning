# 行星齿轮箱故障诊断 - 时频融合 + BiGRU + 注意力 教学文档（完整新版）

本教程覆盖从数据预处理（小波包分解）到模型训练/测试的完整流程，重点解释小波包分解、双向GRU与注意力机制，并给出逐步可操作指引。

---

## 1. 总览：本项目做了什么？

- 从振动信号（CSV）出发，先用“小波包分解（WPT）”把一维信号变成“频带×时间”的系数矩阵（类似时频图，但更适合非平稳信号）。
- 用一维CNN在频带维上做融合（提取频带间关联），保留时间维度 → 用双向GRU建模时序 → 用注意力对时序动态加权 → 分类。
- 训练/测试阶段仅“读取”预计算的 WPT 结果（.npy + index.csv），不再做分解，快速稳定。

---

## 2. 一句话直觉

- 小波包分解：把原始信号“切成 2^L 条不同频段的时间曲线”，堆叠成矩阵（行=频带，列=时间）。
- 一维CNN：把这些频带“揉在一起提特征”，但仍然保留时间轴，用于后续时序建模。
- BiGRU：在时间轴上“看前看后”，提取上下文依赖，得到每个时间步的表示。
- 注意力：自动学出“哪些时间步更重要”，对 GRU 输出按权重加权求和，得到全局特征。

---

## 3. 跑起来：最短路径

1) 预计算 WPT（一次性）
```bash
cd My_Code_10_Classes/TimeFreqAttention_Model
python make_wpt_dataset.py
```
- 读取 `configs/config.yaml`：`data.csv_file_path`, `data.split_ratios`, `wavelet.*`, `data.wpt_output_dir`
- 生成 `data/WPTarrays/{train,val,test}/index.csv + *.npy`

2) 训练
```bash
python train.py
```
- 直接从 `data/WPTarrays/train|val` 读取 `.npy` 与 `index.csv`
- 保存：`checkpoints/run1/best_model.pth` 与 `training_curves.png`

3) 测试
```bash
python test.py
```
- 读取 `data/WPTarrays/test`，输出 `classification_report.txt` 与 `confusion_matrix.png`

---

## 4. 小波包分解（WPT）详解

为什么不是 STFT？
- STFT 用固定窗，时频分辨率“均匀”；WPT 是多尺度，低频/高频都能细分，更适合非平稳振动信号（如齿轮箱冲击等局部特征）。

WPT 直觉：
- 第 1 层：把信号分成低频(L)/高频(H)
- 第 2 层：LL、LH、HL、HH（四个窄频带）
- 第 L 层：2^L 个频带；每个频带都有一条“时间系数曲线”
- 把 2^L 条曲线按行堆叠 → (2^L × 时间点)

本项目的 WPT 代码（已内联在 `make_wpt_dataset.py` 中）：
```python
wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
leaf_nodes = [node.path for node in wp.get_level(max_level, order='freq')]
coeffs_list = [wp[path].data for path in leaf_nodes]
max_len = max(len(c) for c in coeffs_list)
coeff_matrix = np.zeros((len(coeffs_list), max_len), dtype=np.float32)
for i, c in enumerate(coeffs_list):
    coeff_matrix[i, :min(len(c), max_len)] = c[:min(len(c), max_len)]
# 结果：coeff_matrix 形状 (2^L, T)
```

参数建议：
- `wavelet_type='db4'`：常见可靠；可试 'db8'、'bior2.2'。
- `max_level=4`：16 个频带；可尝试 3（8 个）、5（32 个）平衡速度。

---

## 5. 模型结构（CNN + BiGRU + Attention）

数据形状流：
```
输入：WPT (batch, bands, time)  例：(32, 16, 1024)
CNN1d：沿时间做卷积（融合频带特征） → (32, C, 1024)
Pool：时间减半 → (32, C, 512)
转置：给 GRU 用 → (32, 512, C)
BiGRU：时间上下文 → (32, 512, 2H)
Attention：沿时间步加权和 → (32, 2H)
FC：分类 → (32, num_classes)
```

关键实现（`models/TimeFreqAttention.py`）：
- CNN1d 两层 + BN + ReLU：频带融合，保持时间轴
- BiGRU（2 层、双向）：时序上下文
- 注意力：对每个时间步算分数（fc），softmax 得到权重，按权重加权求和

注意力核心：
```python
scores = Linear(2H→1)(gru_output)             # (B, T, 1)
weights = softmax(scores.squeeze(-1), dim=1)  # (B, T) 每个样本时间步权重和=1
weighted = sum(gru_output * weights.unsqueeze(-1), dim=1)  # (B, 2H)
```

---

## 6. 代码入口文件说明

1) `make_wpt_dataset.py`
- 功能：从 CSV 读信号 → 划分 → 对每条样本做 WPT → 保存 `.npy` + `index.csv`
- 形状：`.npy` 内为 `(2^L, time_steps)` 的 float32 矩阵
- 输出：固定目录 `data/WPTarrays/{train,val,test}`

2) `train.py`
- 功能：读取 `train/val` 的 `.npy` 与 `index.csv`，训练模型
- 保存：最佳权重与训练曲线图
- 不做任何 WPT 计算

3) `test.py`
- 功能：读取 `test` 的 `.npy` 与 `index.csv`，评估与绘图
- 加载训练阶段保存的 `best_model.pth`

---

## 7. 配置键说明（configs/config.yaml）

- `data.csv_file_path`：CSV 路径（`mat_to_csv.py` 生成）
- `data.split_ratios`：数据集划分比例（如 [0.7, 0.2, 0.1]）
- `data.wpt_output_dir`：WPT .npy 输出根目录（默认 `../data/WPTarrays`）
- `wavelet.wavelet_type`：小波基，如 'db4'
- `wavelet.max_level`：分解层数 L（频带数=2^L）
- `model.cnn_out_channels`：CNN 输出通道数（64/128 可试）
- `model.gru_hidden_dim`：GRU 隐藏维（单向），双向=2×此值
- `model.dropout_rate`：0.1~0.5 试探
- `train.*`：常规训练超参与输出目录
- `test.batch_size`：测试批大小

---

## 8. 常见问题（FAQ）

- Q：每次训练会不会重复做 WPT？
  - A：不会。训练/测试阶段只读 `.npy`，WPT 仅在 `make_wpt_dataset.py` 里做一次。

- Q：为什么时间长度 T 可能不是严格等于 time_steps？
  - A：边界处理（mode='symmetric'）可能引入 1~2 点差异；项目里统一按最长对齐（不足补零）。

- Q：如何可视化 WPT 结果？
  - A：运行 `viz_wpt_demo.py`（合成信号）或加载 `.npy` 后用 `imshow` 做热力图。

- Q：如何加速/减少显存？
  - A：减小 `batch_size`、降低 `max_level`（频带更少）、降低 `time_steps`（CSV 窗口更短）、使用较小的 `cnn_out_channels` 与 `gru_hidden_dim`。

---

## 9. 建议的调参顺序

1) `wavelet.max_level`：3/4/5 试探 → 频带数 8/16/32
2) `model.gru_hidden_dim`：64/128/256 → 时序表达能力
3) `model.cnn_out_channels`：32/64/128 → 频带融合能力
4) `train.learning_rate`：1e-3/5e-4/1e-4 → 收敛稳定
5) `train.batch_size`：显存允许尽量大

---

## 10. 结果解读

- 训练曲线：两条 Loss 逐步下降、Val Acc 稳中有升 → 一般良好
- 分类报告：F1 值均衡且高 → 类别间可分性好；若某类偏低，考虑该类样本量与频带关注（可用注意力可视化定位时间段）
- 混淆矩阵：观察易混类别，针对性扩充数据或调整滤波/窗口/模型容量

---

如需：
- 增加注意力权重可视化到测试报告
- 支持多文件 CSV 合并或更灵活的数据划分
- 加入早停、学习率调度器、混合精度
请告诉我，我可以直接在现有代码上扩展。
