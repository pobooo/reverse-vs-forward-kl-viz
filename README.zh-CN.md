# Reverse KL vs Forward KL · 多模态分布拟合可视化

[English README](./README.md)

🎮 **在线体验**：<https://qiangliu.net/reverse-vs-forward-kl-viz/>

一个交互式 demo，直观展示 **reverse KL(q‖p)** 和 **forward KL(p‖q)** 在拟合多模态分布时的**典型倾向**：**mode-seeking** vs **mode-covering**。

**零依赖、纯前端**——单个 `index.html` 文件在浏览器里完成梯度下降、数值积分求 KL、Canvas 绘制密度与 loss 曲线。无后端、无 npm、无构建工具，双击 HTML 即可运行，也可以直接部署到 GitHub Pages 等静态托管。

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

推荐配置来直观感受差异（结果依赖随机种子，下表为**典型倾向**而非确定结果）：

| 场景 | K_p | K_q | 常见现象 |
|---|---|---|---|
| ★ 经典对比 | 3 | 1 | Reverse KL 收缩到其中一个峰；Forward KL 拉宽成一个大高斯，试图覆盖三峰 |
| 容量不足 | 5 | 2 | Reverse KL 通常把两个分量分别放到不同的峰上；Forward KL 常出现"一宽一窄"或两个都很宽的形态，具体分布依赖初始化 |
| 中等容量 | 6 | 5 | 有趣情形：两种 KL 都常常**落到局部最优**——比如漏掉中间某个峰，或多个分量重合。见下面 [复杂性说明](#-复杂性说明最终行为并非只由-kl-方向决定) |
| 充足容量 | 4 | 4 | 两种 KL 均能较好拟合，差异基本消失（作为对照组） |

**建议**：给种子填 `42` 复现下表；再换成 `7`、`123` 等，看看同一配置下的多样性。这本身就是这个 demo 想传达的核心信息。

## 📚 教科书里的典型例子 vs 真实情况

绝大多数教材和博客讲 mode-seeking vs mode-covering 时，用的都是**同一个极简例子**：目标 `p` 是两个分得很开的高斯（双峰），拟合分布 `q` 只有**一个**高斯（单峰）。在这个理想化的设定下：

- Reverse KL `q‖p` → `q` 收缩到两个峰中的某一个上（mode-seeking）
- Forward KL `p‖q` → `q` 落在两峰之间、拉宽方差把两个峰都盖住（mode-covering）

这两张图太经典，以至于在大多数读者心里已经**成了两种行为的定义**。这个 demo 可以复现（选 `K_p=2, K_q=1`）——这些描述本身没错。

**但真实问题很少长这样。** 一旦离开"1 个 q 分量拟合 2 个 p 峰"的极简设定，画面会立刻复杂得多：

- **更多目标 mode**（`K_p=5`）：单分量 q 在 reverse KL 下依然会挑一个峰，但**挑哪个**严重依赖初始化；forward KL 的"大团高斯"解也早就不长得像教科书上那张两峰对比图了
- **更多 q 分量**（`K_q > 1`）：两种 KL 都可能产出**同时混合了 mode-seeking 和 mode-covering 行为**的解——一部分 q 分量锁在某个峰上，另一部分横跨两个峰
- **匹配容量**（`K_q = K_p`）：理论上两种 KL 都应该给出 `q = p`，但实际上常常掉进**局部最优**——两个 q 分量重合到同一个 p 峰上，另有几个 mode 被完全漏掉
- **mode 之间重叠**：那种"一个塌缩、一个铺开"的鲜明对比会消失，两种 KL 都给出偏宽的近似，视觉上很难区分

换句话说，**经典的"2 峰对 1 分量"图是一个教学用的极端案例，不是通用预测**。本 demo 支持 K_p、K_q 各自最高 6 个、加上随机种子控制，就是为了让你去探索教科书跳过的、更混乱的中间地带。你会发现："mode-seeking vs mode-covering" 是对**两种 KL 损失偏好方向**的有用总结——但最终优化器给出的形态，受**容量、初始化、局部最优**的影响不亚于 KL 方向本身。

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

## 目录结构

```
kl-viz/
├── frontend/
│   └── index.html       # 全部实现：trainer + Canvas 绘图 + MathJax 公式
├── README.md            # 英文版
└── README.zh-CN.md      # 本文件（中文）
```

就一个 HTML 文件。里面 `<script>` 大约 250 行原生 JS 实现了：高斯混合、手推解析梯度、Adam 优化器、梯形积分 KL、可复现的伪随机数发生器。

## 数学

**目标分布** p(x) — 固定的高斯混合：

$$p(x) = \sum_{k=1}^{K_p} w_k \, \mathcal{N}(x \mid \mu_k, \sigma_k^2)$$

代码中 μ_k 在 [-4·(K_p−1)/2, +4·(K_p−1)/2] 等距取值，σ_k = 0.6，w_k = 1/K_p。

**变分分布** q(x; θ) — 可训练的高斯混合，参数 θ = {m_j, s_j, π_j}：

$$q(x;\theta) = \sum_{j=1}^{K_q} \pi_j \, \mathcal{N}(x \mid m_j, s_j^2)$$

**两种 KL** 都通过**数值积分**（梯形法则，2000 点密网格）计算——由于一维、密度都是解析形式，无需蒙特卡洛采样；梯度对 θ 的公式**手工推导**，用有限差分做过一致性验证：

$$D_\text{KL}(q\|p) = \int q(x) \log \tfrac{q(x)}{p(x)} \, dx \qquad D_\text{KL}(p\|q) = \int p(x) \log \tfrac{p(x)}{q(x)} \, dx$$

**为什么不采样？** 真实 VI 里 p 通常只有未归一化形式，必须靠 MC；这里我们两个都是自定 GMM，数值积分更稳定，`loss ≥ 0` 严格成立，也避免了 log(0) 溢出。

## 启动

```bash
git clone <this-repo>
cd kl-viz
open frontend/index.html         # macOS
# 或: xdg-open frontend/index.html   （Linux）
# 或: start frontend/index.html      （Windows）
```

也可以起个静态服务：
```bash
cd kl-viz/frontend && python3 -m http.server 8765
# 然后访问 http://127.0.0.1:8765/
```

不需要 pip、venv、torch——一切在浏览器里跑。

## 用法

1. 顶部 slider 选 K_p (1–6) 和 K_q (1–6)
2. 可选：填一个**随机种子**做可复现的实验（留空则每次真随机）
3. 按钮：
   - **Reverse KL** — 单独训一个 reverse KL 拟合
   - **Forward KL** — 单独训一个 forward KL 拟合
   - **同时对比** — 两个 runner 通过 `requestAnimationFrame` 并行运行，在同一张图上叠画红/绿两条 q 曲线
4. 观察上图（密度曲线）和下图（loss 下降）

## 实现要点

- **手工推导的解析梯度**：不依赖任何 autograd 框架。`∂q/∂m_j`、`∂q/∂log s_j`、`∂q/∂logit_j` 都是闭式解。在约 100 组配置下用**有限差分**做过验证，只要积分网格宽到能容纳 q 的尾部，相对误差 ≤ 1e-8
- **梯形积分**：2000 点密网格覆盖 p 范围外扩 ±8——精度足够让 `loss ≥ 0` 严格成立，无 MC 噪声
- **σ clamp**：每次前向计算时 σ 被 clamp 到 [0.05, 20]，防止数值爆炸
- **可播种 RNG**：内置 mulberry32 伪随机数生成器，填种子可精确复现，留空则每次探索不同的局部最优
- **零依赖**：无构建、无 npm、无框架。MathJax 从 CDN 加载**仅用于渲染公式**——删掉它 demo 一样能用

## 参考

- Eric Jang 的博文 — <https://blog.evjang.com/2016/08/variational-bayes.html>
- Thinking Machines · On-Policy Distillation — <https://thinkingmachines.ai/blog/on-policy-distillation/>
- Bishop, *PRML*, 第 10 章（变分推断，其中包含 reverse KL vs forward KL 的经典对比图）
