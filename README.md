# MTRB-QTA

## Data-Efficient Massive Tool Retrieval: A Reinforcement Learning Approach for Query-Tool Alignment with Language Models

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2410.03212-B31B1B)](https://arxiv.org/abs/2410.03212)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3673791.3698429-2F5D8A)](https://doi.org/10.1145/3673791.3698429)
[![Conference](https://img.shields.io/badge/SIGIR--AP-2024-8B5CF6)](https://doi.org/10.1145/3673791.3698429)
[![Best Paper](https://img.shields.io/badge/Best%20Paper-Shortlisted%20Nominee-D4A017)](https://www.sigir-ap.org/sigir-ap-2024/bestpapers/)

Yuxiang Zhang<sup>*</sup>, <strong>Xin Fan<sup>*</sup></strong>, Junjie Wang<sup>*</sup>, Chongxian Chen, Fan Mo, Tetsuya Sakai, and Hayato Yamana<br>
<sup>*</sup> Equal contribution

*Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP '24)*, pp. 226-235, 2024.

> MTRB-QTA studies how to retrieve a small, sufficient set of tools from repositories containing dozens to thousands of candidates. It introduces the low-resource MTRB benchmark and a Query-Tool Alignment framework that learns to rewrite user queries from retrieval-derived preferences using Direct Preference Optimization.

## Why massive tool retrieval?

Tool-augmented language models must often choose from tool collections that are far too large to fit into a model's context window. A retrieval stage can narrow the collection, but conventional retrievers face three practical difficulties:

- **Complex queries imply multiple tools.** Retrieving only part of the required tool chain is insufficient for completing the downstream task.
- **Query and tool-document language are misaligned.** A natural user request may not resemble the descriptions or endpoint names used by retrieval systems.
- **Domain-specific annotations are scarce.** Fine-tuning a retriever for every new tool collection is expensive and transfers poorly across datasets.

This work formalizes **Massive Tool Retrieval (MTR)**: retrieve the smallest possible tool subset while preserving every tool needed to solve the user's task.

<p align="center">
  <img src="Images/Background.jpg" alt="Massive tool retrieval task and the motivation for MTRB and QTA" width="78%">
</p>

## MTRB benchmark

The **Massive Tool Retrieval Benchmark (MTRB)** reorganizes three established tool-use datasets into realistic, low-resource retrieval tasks. Each subset contains 10 training samples and 90 test samples in the benchmark design. The public files in this repository provide the 90 labeled test queries and curated tool documents for each subset.

<p align="center">
  <img src="Images/Statistics.jpg" alt="Statistics of the three MTRB benchmark subsets" width="80%">
</p>

| Subset | Tools | Tool-document length | Required tools per query | Released test set | Released tool documents |
|---|---:|---:|---:|---|---|
| MTRB-RestBench | 54 | 20-30 tokens | 1-4 | [`mtr_restbench_test.json`](restbench/mtr_restbench_test.json) | [`tmdb_tool_rewrite.json`](restbench/tmdb_tool_rewrite.json) |
| MTRB-MetaTool | 199 | 10-20 tokens | 1 | [`mtr_metatool_test.json`](metatool/mtr_metatool_test.json) | [`ts_metatool_api_list_rewrite.json`](metatool/ts_metatool_api_list_rewrite.json) |
| MTRB-ToolBench | 2,391 | 70-100 tokens | 2-3 | [`mtr_toolbench_test.json`](toolbench/mtr_toolbench_test.json) | [`toolbench_tool_instruction_rewrite_revised.json`](toolbench/toolbench_tool_instruction_rewrite_revised.json) |

Every released test set contains `query`, `relevant Tools`, and `relevant IDs` fields. The tool-document files contain the normalized descriptions used to construct the retrieval collections.

## QTA framework

<p align="center">
  <img src="Images/QTA.jpg" alt="Training and inference pipelines of the Query-Tool Alignment framework" width="100%">
</p>

QTA improves a frozen retrieval model by training an LLM to rewrite user queries so that they align more closely with tool documentation:

1. **Generate candidate rewrites.** Given a small number of annotated queries and their golden tools, an LLM produces multiple rewritten queries.
2. **Extract preferences from retrieval rankings.** A frozen retriever ranks the golden tools for the original and rewritten queries. A task-specific ranking function converts these results into chosen and rejected query pairs.
3. **Optimize with DPO.** Direct Preference Optimization teaches the LLM to favor rewrites that move all required tools toward the top of the ranking, without training a separate reward model.
4. **Retrieve at inference time.** The aligned LLM rewrites a new query, and the frozen retriever returns a compact tool subset for downstream planning and execution.

## Evaluation

MTRB evaluates whether retrieval provides a *complete* tool set, not merely some relevant tools:

- **Sufficiency@k (S@k):** 1 only when every required tool appears in the top-k results; otherwise 0.
- **NDCG@k (N@k):** rewards relevant tools while accounting for their positions in the ranked list.
- Results are reported at **k = 5** and **k = 10**.

<p align="center">
  <img src="Images/Result.jpg" alt="Main MTRB benchmark results" width="100%">
</p>

## Main findings

- **Best Paper Shortlisted Nominee:** this work was selected for the [SIGIR-AP 2024 Best Paper shortlist](https://www.sigir-ap.org/sigir-ap-2024/bestpapers/).
- **Up to 93.28% relative improvement:** on MTRB-RestBench, QTA raises S@5 from 16.67 to 32.22 compared with the fully fine-tuned all-MiniLM-L6-v2 retriever.
- **Effective with extremely limited supervision:** the paper reports an improvement of up to 78.53% in Sufficiency@5 using only one annotated sample.
- **Strong results across all three subsets:** QTA achieves the best or second-best result across every reported MTRB metric.
- **Cross-dataset generalization:** when trained only on MTRB-RestBench and evaluated on MTRB-ToolBench, QTA reaches S@5/S@10 of 30.00/52.22, compared with 24.44/40.00 for the strongest transfer baseline.
- **Works with different retrieval backbones:** QTA improves both BM25 and all-MiniLM-L6-v2, showing that query alignment complements the underlying retriever.

## What does QTA change?

The qualitative examples show that QTA turns short or underspecified requests into retrieval-oriented descriptions of the intermediate steps. This exposes the endpoint names and operations that a frozen retriever needs to locate the full tool chain.

<p align="center">
  <img src="Images/Visualization.jpg" alt="Examples of raw and QTA-rewritten queries on MTRB-RestBench" width="100%">
</p>

## Repository contents

```text
MTRB-QTA/
├── metatool/      # MetaTool test queries and curated tool documents
├── restbench/     # RestBench test queries and curated TMDB tools
├── toolbench/     # ToolBench test queries and curated tool instructions
└── Images/        # Figures reproduced from the paper
```

This repository is a **benchmark-data and project-page release**. Model training code, checkpoints, and end-to-end reproduction scripts are not currently included.

## JiuwenSwarm retrieval runners

The `scripts/` directory contains four retrieval-only runners: BM25 and Jev for
MTRB-MetaTool and MTRB-RestBench. They use JiuwenSwarm's configured agent model
and discovery settings, score the selected tool lists with Sufficiency@5/10 and
NDCG@5/10, and write a structured text report with per-query JSON records.
Tool execution and latency reporting are disabled. The Jev runner omits Jev's
no-tool option, so it measures forced tool ranking; reports record this setting.

Each runner disables system-operation and skill tools, enables deferred-tool
exposure, and checks the registered inventory after initialization. It stops
with an explicit error if any tool outside the benchmark catalog,
`tool_search`, and `tool_call` was registered. The catalog tools remain
deferred search candidates; only the two progressive wrappers are directly
visible.

Run commands from `MTRB-QTA` after setting
`progressive_tool_enabled: true` and the desired
`tool_discovery_backend` (`bm25` or `jev`) in JiuwenSwarm's config:

```powershell
# Smoke run, first three queries
..\jiuwenswarm\.venv\Scripts\python.exe scripts\run_restbench_bm25.py --num-queries 3
..\jiuwenswarm\.venv\Scripts\python.exe scripts\run_restbench_jev.py --num-queries 3

# Full 90-query runs
..\jiuwenswarm\.venv\Scripts\python.exe scripts\run_restbench_bm25.py --num-queries all
..\jiuwenswarm\.venv\Scripts\python.exe scripts\run_restbench_jev.py --num-queries all
```

The runners require the config backend to match the script. Default reports are
`mtrb_restbench_bm25_results.txt` and `mtrb_restbench_jev_results.txt`; each run
overwrites its corresponding report. For MetaTool, use `run_mtrb_bm25.py` and
`run_mtrb_jev.py` instead. RESTBench's released relevant IDs use a different
ordering from its rewritten tool catalog, so report IDs are normalized from
the labeled endpoint names to the rewritten catalog IDs.

## Citation

If you find MTRB or QTA useful, please cite:

```bibtex
@inproceedings{zhang2024dataefficient,
  title     = {Data-Efficient Massive Tool Retrieval: A Reinforcement Learning Approach for Query-Tool Alignment with Language Models},
  author    = {Zhang, Yuxiang and Fan, Xin and Wang, Junjie and Chen, Chongxian and Mo, Fan and Sakai, Tetsuya and Yamana, Hayato},
  booktitle = {Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region},
  series    = {SIGIR-AP '24},
  pages     = {226--235},
  publisher = {Association for Computing Machinery},
  year      = {2024},
  doi       = {10.1145/3673791.3698429}
}
```

## Data provenance and figure attribution

MTRB is derived from RestBench, MetaTool, and ToolBench; users should follow the terms of the corresponding source datasets. Figures in `Images/` are reproduced from the published paper for scholarly communication. The repository does not currently declare a blanket license for the benchmark data or paper figures.
