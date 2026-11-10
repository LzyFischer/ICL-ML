# ICL-ML: In-Context Learning 行为分析实验框架

这是一个面向 class project 的小型实验框架，用来系统研究**示例（demonstrations）的哪些属性决定了 in-context learning 的效果**。所有实验只需推理、不需要训练，可以用本地 HuggingFace 模型（0.5B–8B），也可以用任何 OpenAI 兼容的 API。

| 实验 `--exp` | 研究问题 | 参考文献 |
|---|---|---|
| `shots` | 性能随示例数 k 如何变化？何时饱和？ | Brown et al. 2020; Agarwal et al. 2024 (many-shot) |
| `labels` | 示例的标签重要吗？对比 gold / random / flipped / abstract | Min et al. 2022; Wei et al. 2023 |
| `order` | 同一组示例换个顺序，准确率会差多少？ | Lu et al. 2022 |
| `template` | 换一个 prompt 模板 / 标签词会怎样？ | Zhao et al. 2021 |
| `cot`（GSM8K） | CoT 示例中真正起作用的是推理过程还是答案？ | Wang et al. 2023 (Towards Understanding CoT) |

另外，框架包含一个**方法**：用 LLM 按"规则"为每个 query 挑选示例，并用反馈循环优化这份规则，详见下文[方法](#方法学习示例选择规则)一节。

另外，每个分类实验都会**同时报告 contextual calibration 之后的准确率**（`acc_cal`，Zhao et al. 2021），不需要额外设置，可以直接作为一个"能否缓解敏感性"的分析点。

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 0.5B 模型，在 SST-2 上画出 k-shot 曲线（GPU 上几分钟就能跑完）
python scripts/run_experiment.py --model Qwen/Qwen2.5-0.5B --task sst2 --exp shots

# 标签操控实验
python scripts/run_experiment.py --model Qwen/Qwen2.5-1.5B --task sst2 ag_news --exp labels --k 16

# 顺序敏感性：3 组示例 × 每组 10 种排列
python scripts/run_experiment.py --model Qwen/Qwen2.5-0.5B --task ag_news --exp order --k 8 --seeds 3 --perms 10

# GSM8K：CoT / 只给答案 / 答案错误 / 推理过程错配
python scripts/run_experiment.py --model Qwen/Qwen2.5-1.5B --task gsm8k --exp cot --k 4 --n-test 300

# 画图，并汇总成表格
python scripts/plot_results.py --results results --out figures
```

完整实验（3 个模型 × 全部实验）：`bash scripts/run_all.sh`，可以通过环境变量修改模型，例如 `MODELS="Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-1B" bash scripts/run_all.sh`。

### 使用 API / vLLM

```bash
# OpenAI
export OPENAI_API_KEY=...
python scripts/run_experiment.py --backend api --api-mode chat --model gpt-4o-mini --task gsm8k --exp cot

# 本地 vLLM server（比 HF generate 快很多，适合 GSM8K 和大模型）
vllm serve Qwen/Qwen2.5-7B --port 8000
python scripts/run_experiment.py --backend api --api-mode completion --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-7B --task gsm8k --exp shots
```

注意：API 后端拿不到 label 的 log-prob，所以分类任务改为"生成后匹配标签词"，`acc_cal` 为空，并额外报告 `invalid_frac`（无法解析的比例）。

## 任务

| `--task` | 类型 | 数据来源 | 说明 |
|---|---|---|---|
| `sst2` | 情感 2 分类 | `nyu-mll/glue` (sst2) | 用 validation (872) 作为测试集 |
| `ag_news` | 新闻 4 分类 | `fancyzhx/ag_news` | |
| `rte` | NLI 2 分类 | `nyu-mll/glue` (rte) | 对小模型来说很难，适合做对照 |
| `trec` | 问题类型 6 分类 | `CogComp/trec`（备选 `SetFit/TREC-QC`） | |
| `cr` | 商品评论情感 2 分类 | `SetFit/CR` | |
| `subj` | 主观 / 客观 2 分类 | `SetFit/subj` | |
| `emotion` | 情绪 6 分类 | `dair-ai/emotion`（备选 `SetFit/emotion`） | |
| `banking77` | 意图 77 分类 | `mteb/banking77`（备选 `PolyAI/banking77`） | 标签词由多个 token 组成，打分最慢，建议用较小的 `--n-test` |
| `synthetic2` | 合成规则 2 分类 | 本地生成 | 第二个合成任务，离线测试 meta-train / meta-test 划分时使用 |
| `synthetic` | 合成规则 2 分类 | 本地生成 | 无先验可用，只能从示例中学习，适合"违反先验"类分析；也用于离线测试 |
| `gsm8k` | 数学推理 | `openai/gsm8k` | 生成式评估，抽取 "The answer is X" |
| `synthetic_math` | 简单算术 | 本地生成 | 离线测试用 |

每个分类任务各有 3 个模板，定义在 `icl/tasks.py` 的 `TEMPLATES` 中。新增任务时只需要写一个 loader，并补一组模板。

新加的几个数据集会依次尝试多个 HuggingFace 镜像，并且**按标签名称**映射到固定的标签顺序。因此即使某个镜像的标签顺序和我们定义的不一样，也不会把标签词对错。

> 这些数据集的 ID 在编写时没有实际下载核对过（开发环境连不上 HuggingFace）。如果某个加载失败，报错信息会列出每个镜像的错误原因，在 `icl/tasks.py` 里改一下路径或列名就可以。

## 实验设计细节

- **测试集固定**：`--n-test` 条测试样本用固定 seed 抽取，因此所有条件、所有模型都在完全相同的样本上评估。
- **示例抽样**：每个 `demo_seed` 抽取一组独立的示例，默认按类别平衡抽样（`--no-balance` 关闭）。画图时的误差带 / 误差棒是不同 demo_seed 之间的标准差。
- **分类评估**：比较每个标签词的 log P(label | prompt)，取最大者（Brown et al. 2020 的标准做法），无需解析输出。
- **标签操控**（`labels`）：
  - `gold`：真实标签
  - `random`：每个示例的标签随机均匀抽取
  - `flipped`：系统性映射 y → (y+1) mod C（二分类即全部翻转）。同时报告 `acc`（相对真实标签）和 `acc_flip`（模型跟随翻转映射的比例）。`acc_flip` 越高，说明模型越能"覆盖"先验
  - `abstract`：标签词替换为 foo / bar / baz…（语义为空），测试模型能否从示例中学到新映射
  - 另附 `k=0` 的 zero-shot 作为参照
- **超长上下文**：prompt 超过模型最大长度的条件会被跳过，并记录 `status=too_long`。
- **断点续跑**：结果按条件逐行追加到 `summary.csv`，重新运行同一命令时已完成的条件会被跳过。

## 方法：学习示例选择规则

**问题**：对每个测试 query，应该选哪 k 条训练样本作为示例？

**做法：先检索，再由 LLM 挑选**
1. 用句向量模型（all-MiniLM）检索出与 query 最相似的 top-M 条候选（默认 M=20）。
2. 选择器 LLM **逐条打分**：每次只看"规则 + query（不含标签）+ 一条候选（含标签）"，回答 Yes 或 No，取 log P(Yes) − log P(No) 作为分数。每个 query 内对分数做 z-score 标准化后，与检索相似度加权融合（权重 `--alpha`），取分数最高的 k 条。
   - 逐条打分不需要模型生成编号，2B 规模的模型也能稳定使用；同一 query 的所有候选共享前缀，可以批量计算。
3. 所有方法选出的示例都按随机顺序排列（`order_seed`），所以比较的只是"选哪些"，不涉及顺序。

**规则优化**（`scripts/optimize_rules.py`）：被优化的是规则本身，而不是某一组具体示例。每一轮：
1. 选择器用当前规则，为 meta-train 任务的 dev query 选择示例。dev 集从训练集中划出，不使用测试集。
2. 目标模型在 dev 上评估。分数定义为"相对 kNN 的准确率提升"在各任务上的平均值。
3. optimizer LLM 看到历史上的规则及其分数（OPRO 风格），以及诊断信息：和 kNN 相比的准确率、所选示例的统计（与 query 同标签的比例、每个 prompt 中的标签种类数、示例长度）、目标模型的预测分布、具体的输赢案例。然后改写规则。
4. 新规则的分数更高就接受。所有尝试都会写进 `log.jsonl`，最好的一版保存在 `best_rules.txt`。

学到的规则会**冻结**，直接用在没见过的任务和目标模型上。

**比较的方法**（`--exp method --methods ...`）：

| 方法 | 说明 |
|---|---|
| `random` | 从候选池随机抽 k 条 |
| `knn` | 取检索最相似的 k 条，是**最关键的 baseline** |
| `topm_random` | 从 top-M 中随机抽 k 条，用来排除"只是候选池更好"这一解释 |
| `llm_norules` | 选择器不给规则，用来检验规则本身是否有用 |
| `llm_human` | 人写的规则，也是规则优化的初始值，用来检验学到的规则是否比常识规则更好 |
| `llm_learned` | 学到的规则（需要 `--rules best_rules.txt`），即**我们的方法** |
| `*_adaptive` | 如果 kNN 选出的 k 个邻居标签完全一致（比例 ≥ `--tau`），就直接用 kNN，跳过 LLM。结果里的 `llm_frac` 记录调用 LLM 的比例，`selector_prompts` 记录选择器的调用次数 |

所有分类结果额外报告 `acc_bc`，即 batch calibration（Zhou et al. 2024）之后的准确率。

**运行**

```bash
# 完整流程：学规则 → 在 meta-test 任务上比较 → 迁移到其他尺寸的模型 → 画图
bash scripts/run_method.sh

# 也可以分步运行
python scripts/optimize_rules.py --target Qwen/Qwen3.5-2B --optimizer Qwen/Qwen3.5-9B \
    --tasks sst2 ag_news trec --rounds 10 --dev-size 200 --out rule_opt/qwen2b
python scripts/run_experiment.py --model Qwen/Qwen3.5-2B --task cr subj emotion rte --exp method \
    --rules rule_opt/qwen2b/best_rules.txt --orders 5
python scripts/plot_results.py --results results --out figures --rule-log rule_opt/qwen2b/log.jsonl
```

- main table 只使用 order seed 0 的结果（`figures/method_acc.png` 和 `tables.md`）。
- 鲁棒性图 `method_orders_acc.png` 展示每个方法在 5 种示例顺序下的准确率分布。
- 规则优化曲线：`rule_opt_curve.png`。
- 选择器默认就是目标模型（"模型为自己挑示例"）。optimizer 建议用更大的模型（`--optimizer`），也可以用 `--optimizer-backend api` 调用 API。optimizer 的调用次数很少，大约每轮一次。
- 模型 ID 需要先在 Hub 上核对。如果 Qwen3.5 是图文多模态的 checkpoint，代码会自动改用 `AutoModelForImageTextToText` 加载，只输入文本。

## 输出

```
results/<exp>/<task>/<model>/summary.csv        # 每个条件一行：acc, acc_cal, macro_f1, acc_flip, ...
results/<exp>/<task>/<model>/predictions.jsonl  # 逐样本预测（分类含各标签得分；GSM8K 含完整输出）
figures/*.png, figures/tables.md                # plot_results.py 生成
```

`predictions.jsonl` 可以用于误差分析，比如看模型是否总是预测同一个类别（`majority_pred_frac` 也给出了这个指标）。

## 代码结构

```
icl/tasks.py      数据集加载、模板、合成任务
icl/prompts.py    示例抽样、排列、标签 / CoT 操控、prompt 构造
icl/models.py     HFModel（log-prob 打分 + 生成）、APIModel（OpenAI 兼容）
icl/metrics.py    答案抽取、calibration、macro-F1
icl/runner.py     实验 → 条件列表 → 评估循环 → summary.csv；run_method（方法比较）
icl/retrieval.py  句向量检索（附带缓存；`hash` 选项可离线使用）
icl/selection.py  各种选择器、按 query 评估、所选示例的统计
icl/rule_opt.py   规则优化循环
scripts/          run_experiment.py / optimize_rules.py / plot_results.py / run_all.sh / run_method.sh
tests/            离线端到端测试（随机初始化的微型 GPT-2 + hash 检索，不需要联网）
```

运行测试：`python -m pytest -q`

## 建议的报告结构

1. **k-shot 曲线**（`shots`）：3 个模型规模 × 2 个任务。观察是否单调、何时饱和，以及 calibration 是否改变曲线形状。
2. **标签是否重要**（`labels`）：random 相比 gold 掉多少？flipped 能否让大模型"跟着错"（`acc_flip` 随规模的变化）？abstract 标签下还能学会吗？
3. **敏感性**（`order` + `template`）：同一组示例在不同排列下的准确率极差，与不同 demo set 之间的差异相比如何？calibration 能否降低方差？
4. **推理任务上的对照**（GSM8K `cot`）：在推理任务中，示例的哪一部分在起作用？

## 算力参考

- Qwen2.5-0.5B / 1.5B：消费级 GPU（甚至 Colab T4，用 `--dtype float16`）就足够。
- 7B–8B：需要 24GB 显存（bf16），k=32 时可以把 `--batch-size` 调到 2–4。
- GSM8K 生成是最慢的部分。建议先用 `--n-test 100` 试跑，或者使用 vLLM 后端。
