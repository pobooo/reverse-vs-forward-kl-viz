# Reverse KL vs Forward KL · 多模态分布拟合可视化

[English README](./README.md)

一个交互式 demo，直观展示 **reverse KL(q‖p)** 和 **forward KL(p‖q)** 在拟合多模态分布时的**典型倾向**：**mode-seeking** vs **mode-covering**。

后端 PyTorch 做梯度下降，前端 Canvas 实时绘制密度曲线与 loss 曲线，通过 WebSocket 每步推送状态。

> **注意**：本 demo 用于建立"倾向性直觉"。真实场景下的最终拟合行为**远比一句 "mode-seeking / mode-covering" 复杂**，取决于超参数、模型容量、数据分布的多种因素。详见后文 [复杂性说明](#-复杂性说明最终行为并非只由-kl-方向决定)。

## 为什么写这个

理解 mode-seeking / mode-covering 的一手参考：
- Eric Jang, *A Beginner's Guide to Variational Methods: Mean-Field Approximation* — <https://blog.evjang.com/2016/08/variational-bayes.html>
- Thinking Machines, *On-Policy Distillation* — <https://thinkingmachines.ai/blog/on-policy-distillation/>

两篇文章反复出现同一个结论：**KL 的方向选错，行为会截然不同**。这个 demo 就是把这个结论"看得见摸得着"。

### 和 On-Policy Distillation 的关系

Thinking Machines 那篇博客的核心论点是：LLM 蒸馏该用**在策略 (on-policy) + reverse KL**，而不是传统的 off-policy SFT (等价于 forward KL 的 MLE)：

| 视角 | Forward KL / SFT | Reverse KL / On-policy distillation |
|---|---|---|
| 采样来源 | 教师轨迹（离线） | 学生自己 rollout（在线） |
| 数学形式 | $\mathbb{E}_{x\sim p_{teacher}}[-\log q_{student}(x)]$ | $\mathbb{E}_{x\sim q_{student}}[\log q - \log p_{teacher}]$ |
| 行为倾向 | **mode-covering** — 覆盖教师所有 mode，产出"平均化"的策略 | **mode-seeking** — 聚焦教师某一个最优 mode，避免多策略混合 |
| 分布漂移 | 严重（学生会走到教师从未见过的状态） | 无（数据本来就来自学生） |

引用他们博文（意译）：
> "reverse KL 是 'mode-seeking' —— 它学习一种特定行为（教师的），而不是将其分布分散在多个次优选项上。"

**这个 demo 里 K_p=3, K_q=1 的场景**，就是那个思想的一维标量玩具版：
- 教师 = p（三峰混合，其中每个峰可以想象成一种"合理的行为策略"）
- 学生 = q（容量不够，只能挑一个）
- Reverse KL 让 q 果断挑一个峰 → 得到一个**清晰、一致**的策略
- Forward KL 让 q 摊平覆盖三峰 → 得到一个**含糊、平均**的策略

> **但请注意**：这只是"loss 形式偏好"层面的类比。真实 LLM 蒸馏中 reverse KL 好不好用还取决于教师质量、学生容量、rollout 长度、KL 权重系数、其他正则项等。**不能把这个一维玩具的结果直接外推**——它是"帮你建立直觉"，不是"证明 reverse KL 就是万能药"。

## 展示效果

推荐配置来直观感受差异：

| 场景 | K_p | K_q | 现象 |
|---|---|---|---|
| ★ 经典对比 | 3 | 1 | Reverse KL 收缩到某一个峰；Forward KL 拉宽一个大高斯盖住三个峰 |
| 部分覆盖 | 5 | 2 | Reverse KL 精准落在两个峰上；Forward KL 试图摊平所有五峰 |
| 充足容量 | 4 | 4 | 两种 KL 均能较好拟合（作为对照组） |

## ⚠ 复杂性说明：最终行为并非只由 KL 方向决定

上文的 "mode-seeking / mode-covering" 是**倾向性表述**，是把复杂问题**极端化**得到的直觉。真实实验里，你会看到很多不符合这个"教科书图示"的现象。**最终拟合形态由以下因素共同决定**：

### 1. 模型容量 (K_q)
- 当 q 的容量足够 (K_q ≥ K_p)，**两种 KL 都能拟合准确**，"mode-seeking vs mode-covering" 的差异会消失。这时 reverse KL 也覆盖所有 mode，因为最优解就是 q = p。
- 差异只在 **q 容量不足 (K_q < K_p)** 时才明显。

### 2. 目标分布的形状
- 直观上：**mode 之间距离越远 / 权重越均匀** → mode-seeking 和 mode-covering 的视觉差异越大
- 反之，如果各 mode 严重重叠或 σ 相对间距很大，p 本身就"糊成一团"，两种 KL 的最优解看起来会更相似
- **这些是从 KL 定义出发的定性推理，具体现象请在 UI 上自行验证**（比如把代码里 `TargetMixture` 的 σ 从 0.6 改成 2.0 试试）

### 3. 优化超参数
- **学习率**：太大 → 直接跳过某个 mode；太小 → 停在初始位置附近的局部最优
- **初始化**：q 分量的初始 μ 分布决定了它们各自"被吸引"到哪个 p 峰。同一组超参数不同随机种子，reverse KL 可能挑不同的 mode
- **迭代步数**：太少可能停在中间状态，收敛后才呈现出典型的 mode-seeking / mode-covering 形态

### 4. 局部最优
- KL loss 在混合模型下**极度非凸**。两个 q 分量重合到同一个 p 峰上（另外的 p 峰无人问津）是一个非常常见的**合法局部最优**——不是 bug，是数学
- 增加 q 容量并不保证解决：K_q = K_p = 5 时也常见"3 个分量挤同一个峰、2 个 mode 被漏掉"
- 传统 GMM-EM 里的 K-means 初始化、多次重启选最优等技巧，本质上都在对抗这个问题

### 5. 采样方式（真实 VI 中）
本 demo 用**数值积分**求 KL，梯度精确无偏。真实 VI/RL 里 KL 只能靠**蒙特卡洛**估计，会引入：
- 高方差（reverse KL 尤甚，因为要在 q 的样本上算 log(q/p)，q 尾部样本落到 p 极低密度区就会爆炸）
- 估计量的偏差与不稳定，这也是很多 RL/VI 算法需要额外正则或约束（如 PPO clip、TRPO trust region）的动机之一

### 一句话总结
> **"Reverse KL 是 mode-seeking" 是关于"损失函数的偏好方向"的定性表述，不是"实际优化结果"的定量保证**。
> 实际能观测到什么行为，是 loss × 模型 × 数据 × 优化器 × 随机性 五者的联合结果。
> 这个 demo 展示的是"最容易复现的极端案例"，不是"永远会发生的规律"。

**建议动手实验**：把 K_p=5, K_q=3 跑几十次（每次刷新一下重启训练），你会看到 reverse KL 有时挑 3 个左边 mode、有时挑 3 个右边、偶尔还会有分量重合到同一个峰。这种随机性正是"复杂性"的一部分。



```
kl-viz/
├── backend/
│   ├── app.py           # FastAPI + WebSocket + PyTorch，数值积分求 KL
│   └── requirements.txt
├── frontend/
│   └── index.html       # 纯 HTML/Canvas + MathJax，无构建
├── run.sh               # 一键启动脚本
└── README.md
```

## 数学

**目标分布** p(x) — 固定的高斯混合：

$$p(x) = \sum_{k=1}^{K_p} w_k \, \mathcal{N}(x \mid \mu_k, \sigma_k^2)$$

代码中 μ_k 在 [-4·(K_p−1)/2, +4·(K_p−1)/2] 等距取值，σ_k = 0.6，w_k = 1/K_p。

**变分分布** q(x; θ) — 可训练的高斯混合，参数 θ = {m_j, s_j, π_j}：

$$q(x;\theta) = \sum_{j=1}^{K_q} \pi_j \, \mathcal{N}(x \mid m_j, s_j^2)$$

**两种 KL** 都通过**数值积分**（梯形法则，2000 点密网格）在 PyTorch 中求得——由于一维、密度都是解析形式，无需蒙特卡洛采样，梯度对 θ 自动回传：

$$D_\text{KL}(q\|p) = \int q(x) \log \tfrac{q(x)}{p(x)} \, dx \qquad D_\text{KL}(p\|q) = \int p(x) \log \tfrac{p(x)}{q(x)} \, dx$$

**为什么不采样？** 真实 VI 里 p 通常只有未归一化形式，必须靠 MC；这里我们两个都是自定 GMM，数值积分更稳定，`loss ≥ 0` 严格成立，也避免了 log(0) 溢出。README 结尾有关于这个选择的讨论。

## 启动

```bash
git clone <this-repo>
cd kl-viz
./run.sh
```

首次运行会自动创建 venv 并安装 torch/fastapi（大约几分钟）。启动后打开 <http://127.0.0.1:8000>（如果端口被占用，改 `run.sh` 里的 `--port`）。

## 用法

1. 顶部 slider 选 K_p (1–6) 和 K_q (1–6)
2. 按钮：
   - **Reverse KL** — 单独训一个 reverse KL 拟合
   - **Forward KL** — 单独训一个 forward KL 拟合
   - **同时对比** — 两个 WebSocket 并行运行，在同一张图上叠画红/绿两条 q 曲线
3. 观察上图（密度曲线）和下图（loss 下降）

## 实现要点

- **数值积分 vs MC 采样**：早期版本用 Gumbel-Softmax reparam 从 q 采样估计 reverse KL，遇到过 `log q(x_soft) → -∞` 的数值陷阱（soft 采样点 ≠ 真实混合密度点）。改成数值积分后 loss 严格非负、光滑无噪
- **σ clamp**：训练中 σ 被 clamp 到 [0.05, 20]，防止梯度爆炸
- **q 初始化**：分量均值初始范围跟随 p 的展宽，避免所有分量挤在 [-3, 3]
- **前端零依赖**：Canvas 手绘，MathJax 从 CDN 加载公式，无 webpack/npm

## 参考

- Eric Jang 的博文 — <https://blog.evjang.com/2016/08/variational-bayes.html>
- Thinking Machines · On-Policy Distillation — <https://thinkingmachines.ai/blog/on-policy-distillation/>
- Bishop, *PRML*, 第 10 章（变分推断，其中包含 reverse KL vs forward KL 的经典对比图）
