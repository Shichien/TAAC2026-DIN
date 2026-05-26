# 别人提到的提分点

1. 语义 Group + 调节超参
2. 序列长度和时间、用户 Embedding，还有 User Pair（Int-Dense），即存在几组 Dense 和 Int 映射特征
3. Tokenization 的设计，简单做了一些修复，使其更加合理化
4. 序列编码器选型
5. 模型容量调整
6. 时间特征，加的位置在 User Dense 里
7. Dense 特征筛选
8. LR Schedule
9. 缺失值处理
10. Batch Size 调优
11. 加门控
12. 增加对于用户和历史序列 Token 的 Dropout
13. Tokenization Scheme 上，核心思路让所有模态在进入 Backbone 之前完成语义对齐，而不是把对齐的工作留给后面的 Attention 层去隐式完成
14. 序列时间特征、item 侧特征、优化器
15. User Dense Feats 将 UE 分离出来，其他和 int pair 加权
16. 建模为 raw embedding + 序列 merge + fafe + din + mlp

其实挺迷的。目前看，用 Transformer + Cross Attention 提取序列特征的模块，全换成 DIN 的提升很大。DIN 的 Target 是只用 Item，而不是所有 NS Token，序列也没有使用 Merge，换成 DIN 之后 Transformer 也不在了。

## 已确认的重要约束

### 训练与测试时间窗口

- 线上 Eval 测试集样本数是 `310000`。
- `UTC+8` 时间窗口是 `2026-03-23 07:40:35` 到 `2026-03-23 09:13:44`，主要落在周一早晨 `7`、`8`、`9` 点。
- 测试集最小时间戳正好等于训练集最大时间戳 `1774222835`，说明线上测试与公开训练集在时间轴上是无缝接续的。
- 公开训练集共 `1010000` 条，时间范围是 `2026-03-18 17:00:28` 到 `2026-03-23 07:40:35`，跨度约 `4 d 14 h 40 m 7 s`。
- 这意味着验证切分必须按真实 timestamp 处理，weekend 和 night 这类粗时间标记不一定稳定有效。

### 已确认的数据事实

- `UID` 没有学习价值。每个 `UID` 基本只有一条样本，做 `UID Embedding` 是无效的。
- 特征异质性很强，序列、稀疏离散、用户统计量、Dense 数值不是一类信号，不能继续一刀切处理。
- `8`、`81`、`83`、`84`、`85` 存在高缺失率和大量 `0` 值。
- `83`、`84`、`85` 存在层级关系，不能按彼此独立的普通离散特征理解。
- 某几列 Dense 特征和对应的 Int 特征之间存在 element-wise 对应关系，如果投影方式不对，信息会在前面静默丢失。

### 建模前必须正视的坑

- `user_dense_feats_62~66` 不是普通 Embedding，而更像统计量加离散模板的组合，不能整块 Dense 糊过去。
- `Domain D` 的列顺序不统一，这不是小脏点，而是会直接影响序列语义。
- `domain_c_seq_47` 的 item ID 量级与顶层 `item_id` 相同，但它按普通序列特征处理时并不会自动得到正确建模。
- 结论不是这些信息以前没喂进模型，而是过去缺少精细化建模。

### 长序列是硬约束，不是可选项

- 长序列处理方式在建模前就已经被 latency 限制约束住了，不是一个可以自由选择的设计空间。
- `Domain D` 的序列长度很长，`max` 接近 `4000`，四个域合计的行为跨度可达 `232` 天。
- baseline 对各域做了基础编码，但没有真正的长序列专项策略，只是走普通 `seq_domains` 流水线。
- 如果继续用朴素全序列 Attention，推理延迟很难过限制；后续方案必须正面处理长序列压缩、检索或分层读取。

## 训练稳定性的处理

高基数特征的 Embedding 在样本量有限的情况下，第一个 epoch 基本是在拟合噪声。工业界的处理方式是对这部分参数在特定 epoch 之后做重初始化，同时保留低基数特征已经积累的优化器状态。效果是让高基数特征获得一次冷重启，用更干净的初始化重新学习第二遍数据。

- baseline 虽然有重初始化机制，但默认配置比常见的高基数冷重启更激进。
- 默认 `reinit_sparse_after_epoch=1`，也就是从第 `1` 个 epoch 结束后开始触发。
- 默认 `reinit_cardinality_threshold=0`，按代码真实语义会导致几乎所有正常的 Embedding 表都被重置，而不是只重置高基数表。
- 这意味着 baseline 实现更像粗暴的全量重启近似版，不是精细版高基数冷重启，后续必须重新审视这个默认设定。

## 其他计划

拉长序列做高效 Attention，辅助 Loss。

## 几个问题点

1. item id 线下 auc 虚高，线上掉，估计也是分布问题。id 特征在精排没用真的难绷，搞了好多版。
2. 随机种子复现：看起来固定了种子还是有随机性，问 opus 说是 cuda determine 没固定 + sdpa 算子 + iterable 的问题，还没确定。
3. 验证集划分：正常来说按 timestamp 划分选最佳参数，再全量训练会更好，也可以训完最近的数据。但目前先按随机划分实验。
4. pure model 建模：主流的 rankmixer、onetrans、hyformer、mixformer 都基本为了提 mfu 改成 pure model，没有太多精细的传统派小结构建模，但试了下都掉分，可能数据量不够。
5. 长序列：seqc feat 47 跟 item id 是一样的，做不做 match 特征都掉分。

## 可尝试的改进方向

1. 时间分桶边界重设计：按真实分布重画时间分桶边界，减少空桶与过度集中问题。
2. 序列长度截断优化：按各域真实长度 p99 设置截断，避免信息丢失与算力浪费。
3. 缺失值差异化处理：新增 missing 专用特征或二值标识，避免将缺失与真实 0 混淆。
4. 重新设计 ns_groups.json：按特征相关性或重要性分组，让 token 向量语义更干净。
5. 交叉特征增强：加显式交叉结构，如 DCN 或 FM、SENet 重标定，提升 AUC 与训练稳定性。
6. Target Attention（DIN）：让模型针对当前广告激活用户相关历史行为，而非通用 self-attention。
7. 时间侧信息二次利用：将时间差作为序列 Attention Bias，或用 TimeMixin 建模时间维度权重。
8. 序列编码器升级：长短序列用不同编码器，或开启 RoPE 位置编码、基于 index 的有参位置编码优化序列建模。
9. 优化器与学习率调度：用双优化器 + Warmup + 余弦退火，或换 Lion、Muon 等新优化器提升训练效率。
10. Embedding 冷启动策略升级：用渐进式冷启动或 SWA，避免高频 id 信息丢失。
11. 数据增强：序列 mask、特征 dropout、对比学习、对抗训练，低成本提升模型鲁棒性。
12. Loss 改进：用 Rank、InfoNCE、Poly、GHM Loss，分别优化排序、样本稀疏、置信度与难易样本权重。
13. 评估指标细化：用 GAUC、PCOC 及广告分群 AUC，定位模型短板问题所在。
14. 特征重要性计算：通过 Permutation Importance 识别高价值特征，指导缺失值填充与特征优化。
15. Bad Case 误差分析：分析预估与实际不符的样本及低基数特征下的 PCOC 差异，针对性优化与去偏。

### 6.4.5 2026 腾讯广告算法大赛改进 Tricks 总结（个人观点供参考）

改进 Tricks 汇总（持续更新中）：

1. 时间分桶边界重设计

- BUCKET_BOUNDARIES 是手工设的 64 个对数级边界，从 5 秒到 1 年。手工边界很可能在你的数据分布上有空桶或过度集中桶，浪费 Embedding 容量。
- 跑全量数据统计真实 time_diff 分布，按等频分位数（1/100、2/100 等）重画边界；或者在距离较近的时间段，比如 7 天内，进行密集等频分桶，比如直接按小时分桶，7 天后再按天进行分桶。
- 这里不同的 domain 序列特征，其 time_diff 分布也不同，建议各自分桶处理。

2. 序列长度截断优化

- a:256、b:256、c:512、d:512，c 和 d 域更长。但从 sample 看，d 域单字段近 2000 个 token。256 和 512 的截断在超长序列域上严重丢信息，且大量 0 填充浪费算力。
- 对每个域统计真实非零长度的 p99 或 p95，按每个 domain 的真实情况设置 seq_max_lens。
- 或配合用 LongerEncoder（Top-K 稀疏注意力）处理长序列。

3. 缺失值的差异化处理

- 所有 null、0、-1 统一映射为 0（padding），与真实 padding 没法区分。大量 null 出现在 86、94、99-104、108、109，如果这些字段缺失率与转化高度相关，比如登录用户才有这些字段，把缺失当 0 就丢了关键信号。
- 离散特征加一个 missing 专用 slot，放在 vocab_size 位置，区别于 padding = 0。
- 缺失指示位：把是否缺失作为额外的二值特征 concat 进模型。很多广告业务场景里，缺失本身就是信号。

4. 重新设计 ns_groups.json

- baseline 默认配置走 rankmixer 分桶，不使用 ns_groups.json。GroupNSTokenizer 里同组特征共享一个 token，相关性高的特征聚组能让 token 向量语义更干净。
- 特征重要性先跑一遍，把高 gain 的特征聚一组、低 gain 的特征聚一组，或者把高 gain 的 token 进行单独强化。
- 由于不知道每个特征的具体业务含义，也可部分参考 ns_groups 分组，建议跑一下不同特征之间的相关性数据，来指导分组。

5. 交叉特征增强

- 模型完全靠 Embedding + Attention 隐式学交叉，增加一些显式交叉结构在工业界 AUC 普遍有提升，且训练稳定。
- 加一个 FM 或 DCN-v2 分支：把所有 user_int 和 item_int embedding 送入 DCN 或 FM 交叉，与 Transformer 输出 concat 后过分类头。如果能满足推理要求，这样做应该还好。
- 加 SENet 特征的重标定：对 NS tokens 做 gate-style 的 channel-wise reweighting，SENet 的 weight 也可以用来指定特征重要性。

6. Target Attention（DIN）

- baseline 序列只在内部做 self-attention，没有显式与 target item 做 target attention。MultiSeqQueryGenerator 的 Q tokens 是从 NS + MeanPool 生成的，但 item-side 的具体 ID embedding 没有作为 attention key 进行强化。
- 在 MultiSeqHyFormerBlock 的 cross-attention 里，让 Q tokens 的初始值由 item embedding 引导，即 Q_init = MLP(item_emb)，而非 MLP(GlobalInfo)。
- 或单独加一个 DIN 分支，即 a_t = softmax(Q_target · K_seq)，结果与 Transformer 输出 fusion。
- DIN 的核心 insight 是，用户对广告 A 和广告 B 感兴趣的历史并不相同。target attention 能让模型针对当前广告激活相关历史。
