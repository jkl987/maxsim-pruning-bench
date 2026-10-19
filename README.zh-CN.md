# MaxSim 筛选：可复现审计与独立分箱基线

本仓库整理独立分箱实现、正确性审计、原始计时，以及条件摘要路线的负结果。
它是研究复现材料，不是生产检索引擎；不主张新的 SOTA 或论文级算法创新。

## 无需服务器即可运行

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/demo_cpu.py
python scripts/recompute.py
```

前三项验证可安装性和边界正确性；CPU 示例展示 8B、16B、32B 状态。
结果重算会核验 70 个归档文件的 SHA256，然后从逐查询原始计时恢复表格。

有 CUDA 环境时，可安装匹配机器的 PyTorch，再安装 `.[gpu]`，运行：

```sh
python scripts/demo_gpu.py --device 0 --output run-output/gpu-smoke.json
```

GPU 示例直接调用保留下来的独立状态实现，覆盖随机、负值、并列三类输入与
三种状态预算。这是小规模正确性与可运行性验证，不是百万文档性能复测。

## 我们得到什么

- 同流程独立分箱从 8 箱改为 4 箱，NQ/HotpotQA 的完整流程耗时分别降低
  8.92%/20.39%。每库固定 64 查询、5 遍配对、单进程会话，最终 Top100 一致。
- 24B 局部条件混合相对 24B 独立控制，在三次会话平均后分别变慢
  2.10%/17.26%。保留该负结果，并停止把当前条件摘要作为主研发路线。
- 上界只约束固定码本粗分。它不保证原始向量全库精确 Top100。
- 上述收益是与我们自己的同流程控制比较，不能直接当成对 LIK、FastPLAID
  或其他上游系统的提升。

## 可复现范围

公开包可以直接重算表格，运行 CPU 审计和 GPU 合成示例。完整大库实验的源代码
原样保存在 `archive/source/`，仍依赖原先的大库向量、索引和精算缓存；没有伪装成
外部用户无需数据就能一键完成的大库复测。详见
[复现边界](docs/REPRODUCIBILITY.md) 与 [英文说明](README.md)。

早期失败、无效 oracle 对照和 pilot 记录保留在 `evidence/failed/`，不会混入有效
性能统计。新机制还需要冻结实现后在未用于设计的数据上验证。
