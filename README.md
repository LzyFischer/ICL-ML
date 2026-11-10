# ICL-ML：In-Context Learning 示例选择

一个小型 class project 框架，分两部分：

1. **分析**：示例的数量、标签、顺序分别对 ICL 有多大影响？
2. **方法**：让 LLM 按一份"规则"为每个 query 挑选示例，并用反馈循环自动改进这份规则；学到的规则直接用在没见过的任务上。

所有实验只需推理，默认全部使用同一个模型（Qwen3.5-2B），单卡即可完成。

## 快速开始

```bash
pip install -r requirements.txt

# 先做冒烟测试：小规模、单独的输出目录，几分钟跑完
OUT=runs/smoke N_TEST=50 ROUNDS=2 DEV=50 bash scripts/run_all.sh

# 正式运行
bash scripts/run_all.sh
```

结果全部写到 `runs/main/` 下：

```
runs/main/results/<exp>/<task>/<model>/summary.csv   每个实验条件一行
runs/main/rule_opt/best_rules.txt                     学到的规则
runs/main/rule_opt/log.jsonl                          每一轮的规则、分数和 optimizer 的原始回复
runs/main/figures/*.png, tables.md                    图和表
```

可以通过环境变量修改的参数：`MODEL`（目标模型，同时也是选择器）、`OPTIMIZER`（写规则的模型，用更大的模型效果通常更好，例如 `OPTIMIZER=Qwen/Qwen3.5-9B`）、`N_TEST`、`ROUNDS`、`DEV`、`BS`（batch size）、`OUT`。

## 数据集

只用文本短、标签少的分类任务，跑得快：

| 任务 | 标签数 | 用途 |
|---|---|---|
| `sst2`（电影评论情感） | 2 | 分析 + meta-train |
| `ag_news`（新闻主题） | 4 | meta-train |
| `trec`（问题类型） | 6 | 分析 + meta-train |
| `cr`（商品评论情感） | 2 | meta-test |
| `subj`（主观 / 客观） | 2 | meta-test |
| `emotion`（情绪） | 6 | meta-test |

数据从 HuggingFace 自动下载。每个任务会依次尝试几个镜像，并且**按标签名称**映射到固定的标签顺序，避免镜像之间顺序不同导致标签词对错。数据集 ID 在编写时没有实际下载核对过：如果某个加载失败，报错会列出每个镜像的错误原因，在 `icl/tasks.py` 里改一下路径即可。`synthetic` / `synthetic2` 是本地生成的合成任务，只用于离线测试。

## 第一部分：分析实验

| `--exp` | 问题 |
|---|---|
| `shots` | 准确率随示例数 k（0–32）如何变化？ |
| `labels` | 示例标签换成 random / flipped / abstract（foo、bar）后会怎样？flipped 时还会报告模型"跟随翻转标签"的比例（`acc_flip`） |
| `order` | 同一组示例换 10 种顺序，准确率波动有多大？和换一组示例相比呢？ |
| `template` | 换 prompt 模板会怎样？（`run_all.sh` 默认不跑） |

分类通过比较每个标签词的 log-prob 得到预测，不需要解析输出。所有条件都在同一批固定的测试样本上评估。每次运行还会额外报告两种校准后的准确率：`acc_cal`（contextual calibration）和 `acc_bc`（batch calibration）。

## 第二部分：方法

**按 query 选示例，分两步：**
1. 用句向量模型检索出和 query 最相似的 20 条训练样本作为候选。
2. 选择器 LLM 逐条判断："按照这些规则，这条候选适合作为这个 query 的示例吗？"，取回答 Yes 的概率作为分数，并与相似度加权融合，最后取分数最高的 8 条。

**学习规则**（`scripts/optimize_rules.py`）：在 meta-train 任务上，每一轮：

1. 用当前规则为 dev query 选择示例（dev 集从训练集中划出，不使用测试集）；
2. 用目标模型在 dev 上评估，分数定义为"相对 kNN 的准确率提升"；
3. 把历史规则及其分数、诊断信息（输赢案例、所选示例的统计、模型的预测分布）交给 optimizer，由它改写规则；
4. 分数更高就接受新规则。

最后把学到的规则**冻结**，在没见过的 meta-test 任务上评估。

**比较的方法：**

| 方法 | 作用 |
|---|---|
| `random` | 随机选 |
| `knn` | 取最相似的 k 条，是**最关键的 baseline** |
| `topm_random` | 从 top-20 中随机取，用来排除"只是候选池更好"这一解释 |
| `llm_norules` | 选择器不给规则 |
| `llm_human` | 人写的规则（也是规则优化的初始值） |
| `llm_learned` | 学到的规则，即**我们的方法** |
| `llm_learned_adaptive` | 当 kNN 选出的邻居标签完全一致时直接用 kNN，省去 LLM 调用（`llm_frac` 记录调用 LLM 的比例） |

main table 只看 order seed 0 的结果（`method_acc.png`、`tables.md`）；`method_orders_acc.png` 展示每个方法在 5 种示例顺序下的结果，用来说明结论不依赖顺序。规则优化的过程见 `rule_opt_curve.png`。

## 单独运行某一步

```bash
python scripts/run_experiment.py --model Qwen/Qwen3.5-2B --task sst2 trec --exp shots
python scripts/optimize_rules.py --target Qwen/Qwen3.5-2B --tasks sst2 ag_news trec --out runs/main/rule_opt
python scripts/run_experiment.py --model Qwen/Qwen3.5-2B --task cr subj emotion --exp method \
    --rules runs/main/rule_opt/best_rules.txt --orders 5
python scripts/plot_results.py --results results --out figures
```

optimizer 也可以用 API：`--optimizer-backend api --optimizer <模型名> --base-url ...`（需要设置 `OPENAI_API_KEY`）。

## 代码结构

```
icl/tasks.py       数据集和模板
icl/prompts.py     示例抽样、标签操控、prompt 构造
icl/models.py      HFModel（label 打分 + chat 生成）、APIModel（仅用于 optimizer）
icl/runner.py      分析实验 + 方法比较的评估循环，结果写入 summary.csv（支持断点续跑）
icl/retrieval.py   句向量检索
icl/selection.py   各种选择器
icl/rule_opt.py    规则优化循环
scripts/           run_all.sh / run_experiment.py / optimize_rules.py / plot_results.py
tests/             离线测试（随机初始化的微型 GPT-2，不需要联网）：python -m pytest -q
```

## 注意事项

- 模型 ID 请先在 HuggingFace 上核对。如果 Qwen3.5 是图文多模态的 checkpoint，代码会自动改用对应的类加载，只输入文本；Qwen3.5 需要较新版本的 transformers。
- 如果 `rule_opt_curve.png` 里每一轮都显示没有生成新规则，打开 `log.jsonl` 看 optimizer 的原始回复：通常是模型没有按 `<rules>...</rules>` 格式输出，这时换一个更大的 `OPTIMIZER` 即可。
- 选择器每次只看一条候选，所以没法保证一组示例之间的多样性。这一点可以作为局限性写进报告。
