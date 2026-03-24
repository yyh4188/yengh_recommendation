# 🎬 Yengh Recommendation System

<div align="center">
  <img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=modern%20recommendation%20system%20logo%20with%20movie%20film%20strip%20and%20neural%20network%20visualization%2C%20blue%20and%20purple%20color%20scheme%2C%20clean%20minimalist%20design&image_size=square_hd" alt="Yengh Recommendation System" width="300" height="300">
</div>

## 项目简介

Yengh Recommendation System 是一个基于深度强化学习（DDPG算法）的智能推荐系统，专为个性化推荐场景设计。该系统采用连续动作空间的策略网络，能够动态学习用户偏好，提供精准的个性化推荐。

## 核心特色

- **深度强化学习**：采用DDPG算法，实现端到端的推荐策略学习
- **连续动作空间**：使用电影嵌入向量作为连续动作，提供更丰富的推荐表达
- **对比学习**：集成对比学习损失，增强模型对用户偏好的理解
- **注意力机制**：考虑用户历史行为的不同重要性，提升推荐质量
- **自适应学习**：动态调整学习率，优化训练效果
- **完整评估体系**：包含NDCG、Precision等多种推荐评估指标

## 项目结构

```
yengh_recommendation/
├── src/
│   ├── models/          # 神经网络模型
│   │   ├── models.py    # Actor, Critic等模型定义
│   │   └── __init__.py
│   ├── algorithms/      # 算法实现
│   │   ├── ddpg.py      # DDPG算法
│   │   ├── misc.py      # 辅助函数
│   │   └── __init__.py
│   ├── data/            # 数据处理
│   │   ├── env.py       # 环境和数据加载
│   │   ├── utils.py     # 数据处理工具
│   │   └── __init__.py
│   └── utils/           # 通用工具
│       ├── evaluation.py # 评估指标
│       ├── misc.py      # 工具函数
│       └── __init__.py
├── data/               # 数据目录
├── main.py             # 主运行文件
├── requirements.txt    # 依赖包
└── README.md          # 项目说明
```

## 技术栈

- **框架**：PyTorch 1.7.0+
- **算法**：DDPG (Deep Deterministic Policy Gradient)
- **优化器**：Ranger
- **评估**：NDCG@10, Precision@10
- **可视化**：TensorBoard
- **数据处理**：Pandas, NumPy

## 安装指南

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/yengh_recommendation.git
cd yengh_recommendation
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 数据准备

1. 下载MovieLens 20M数据集：https://grouplens.org/datasets/movielens/20m/
2. 下载电影嵌入文件：https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL

将数据文件放置在以下目录结构：

```
data/
├── ml-20m/
│   └── ratings.csv
├── embeddings/
│   └── ml20_pca128.pkl
└── cache/
    └── frame_env.pkl  # 自动生成
```

## 快速开始

### 基础训练

```bash
python main.py
```

### 自定义训练参数

编辑 `main.py` 中的参数：

```python
frame_size = 10      # 序列长度
batch_size = 25      # 批次大小
n_epochs = 100       # 训练轮数
hidden_size = 256    # 隐藏层大小
```

### 模型架构

**Actor网络**：
- 输入：用户历史（10部电影的嵌入 + 评分）
- 隐藏层：256维 × 2层
- 输出：128维电影嵌入向量

**Critic网络**：
- 输入：状态 + 动作
- 隐藏层：256维 × 2层
- 输出：状态-动作对的价值估计

## 监控与评估

### TensorBoard监控

```bash
tensorboard --logdir=./runs
```

### 评估指标

- **NDCG@10**：归一化折损累积增益
- **Precision@10**：前10个推荐的精确率

## 模型保存与加载

- 模型会自动保存到 `models/best-clattention.pth`
- 加载模型：

```python
checkpoint = torch.load('models/best-clattention.pth')
policy_net.load_state_dict(checkpoint['policy_net'])
value_net.load_state_dict(checkpoint['value_net'])
```

## 扩展与定制

### 添加新算法

在 `src/algorithms/` 目录下创建新的算法文件，参考 `ddpg.py` 的实现。

### 自定义数据处理

修改 `src/data/env.py` 中的 `prepare_dataset` 方法来适配你的数据格式。

### 调整网络架构

在 `src/models/models.py` 中修改或添加新的网络类。

## 性能优化

1. **使用GPU**：确保在支持CUDA的环境中运行
2. **调整批量大小**：根据显存大小调整 `batch_size`
3. **学习率调度**：系统会自动根据验证损失调整学习率
4. **早停机制**：当性能不再提升时自动停止训练

## 常见问题

**Q: 如何使用自己的数据集？**

A: 修改 `src/data/env.py` 中的数据加载逻辑，将你的数据转换为标准格式。

**Q: 如何提高推荐质量？**

A: 尝试调整以下参数：
- 增加 `frame_size` 以捕获更长的用户历史
- 调整 `hidden_size` 以增加模型容量
- 修改对比学习权重 `contrastive_weight`

**Q: 训练时间过长怎么办？**

A: 减少 `n_epochs` 或增加 `batch_size`，或使用更强大的GPU。

## 项目亮点

1. **创新的连续动作空间设计**：使用电影嵌入作为动作，提供更丰富的推荐表达
2. **对比学习增强**：通过对比损失提升模型对用户偏好的理解
3. **完整的评估体系**：包含多种推荐系统评估指标
4. **自适应训练**：动态调整学习率和早停机制
5. **模块化设计**：易于扩展和定制

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

**Yengh Recommendation System** - 智能推荐，精准触达