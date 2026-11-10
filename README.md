# ICL-ML: In-Context Learning 行为分析实验框架

这是一个面向 class project 的小型实验框架，用来系统研究**示例（demonstrations）的哪些属性决定了 in-context learning 的效果**。所有实验只需推理、不需要训练，可以用本地 HuggingFace 模型（0.5B–8B），也可以用任何 OpenAI 兼容的 API。

| 实验 `--exp` | 研究问题 | 参考文献 |
|---|---|---|
| `shots` | 性能随示例数 k 如何变化？何时饱和？ | Brown et al. 2020; Agarwal et al. 2024 (many-shot) |
| `labels` | 示例的标签重要吗？对比 gold / random / flipped / abstract | Min et al. 2022; Wei et al. 2023 |
| `order` | 同一组示例换个顺序，准确率会差多少？ | Lu et al. 2022 |
| `template` | 换一个 prompt 模板 / 标签词会怎样？ | Zhao et al. 2021 |
| `cot`（GSM8K） | CoT 示例中真正起作用的是推理过程还是答案？ | Wang et al. 2023 (Towards Understanding CoT) |

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
| `synthetic` | 合成规则 2 分类 | 本地生成 | 无先验可用，只能从示例中学习，适合"违反先验"类分析；也用于离线测试 |
| `gsm8k` | 数学推理 | `openai/gsm8k` | 生成式评估，抽取 "The answer is X" |
| `synthetic_math` | 简单算术 | 本地生成 | 离线测试用 |

每个分类任务各有 3 个模板，定义在 `icl/tasks.py` 的 `TEMPLATES` 中。新增任务时只需要写一个 loader，并补一组模板。

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
icl/runner.py     实验 → 条件列表 → 评估循环 → summary.csv
scripts/          run_experiment.py / plot_results.py / run_all.sh
tests/            离线端到端测试（随机初始化的微型 GPT-2，不需要联网）
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
