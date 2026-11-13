# SOSR 复现仓库

论文 *Efficiently Answering Semantic Optimal Sequenced Route Queries with Dynamic Candidates* 的代码与实验材料。

HSM 将自然语言需求映射到一个 POI 类别，FSSP 在对应候选集合上规划路线。

## 模块说明

| 模块 | 内容 | 文档 |
|---|---|---|
| HSM | 模型训练、检查点评测、三项消融 | [HSM README](hsm/README.md) |
| 路由 | FSSP、路由基线、编译和运行 | [Routing README](routing/README.md) |
| 语义基线 | XLM-R、Sentence-T、Qwen、Qwen-City | [Baselines README](baselines/README.md) |
| 数据 | 数据划分、格式、路网文件和来源 | [Data README](data/README.md) |
| 结果 | 保存的预测、测量结果、表格和绘图 | [Results README](results/README.md) |

## 快速复算结果

在仓库根目录运行：

```bash
pip install -r requirements.txt
python scripts/check_data.py
python scripts/reproduce_results.py
python scripts/plot_results.py
```

输出位于 `runs/reproduced/` 和 `runs/figures/`。完整训练与路由运行命令见各模块文档。

语义数据已包含在仓库中。[下载路网数据包](https://github.com/islandonwei/SOSR/releases/download/reproducibility-2026-11-13/SOSR-road-data-20261113.zip)，并解压到仓库根目录。模型权重可通过训练生成或作为独立模型附件提供。

路由入口使用已校准几何下界的实现。历史论文绘图数据和新增四方法比较的版本差异见 [来源说明](docs/provenance.md)。

项目许可证待作者确定。引用信息见 [CITATION.cff](CITATION.cff)，模型和数据来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
