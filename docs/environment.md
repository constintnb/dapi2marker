# 环境配置

用户提供的 `environment.yml` 是 Linux Conda `base` 环境导出：Python
3.13.11，包含 Conda 本身、Anaconda 管理组件、Linux 编译库和 Jupyter 内核。
文件中未列出 `torch`、`diffusers`、`accelerate`、`numpy`、`Pillow`、
`PyYAML`、`scikit-image`、`safetensors` 和 `huggingface-hub`；已有 `tqdm`。
因此它记录了基础环境，不能单独复现本项目的训练环境。这个结论基于导出
文件，不表示已经检查过训练服务器的实际安装情况。

原文件保留为本地环境记录。GitHub 发布使用 `environment.project.yml`，
保留 Python 3.13 系列，创建名为 `dapi2marker` 的独立环境，不带机器的
绝对 `prefix`、Linux 构建号或 Conda 管理组件。Conda 官方说明，分享环境
时应避免携带不必要的平台依赖；参见
[跨平台环境导出](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)。

## 创建项目环境

在解压或克隆后的项目根目录执行：

```bash
conda env create -f environment.project.yml
conda activate dapi2marker
```

这是 Python 和安装工具的引导环境。先安装适合训练硬件的 PyTorch，再安装
项目；训练依赖统一维护在 `pyproject.toml` 中。Python 3.13 的项目依赖要求
PyTorch >= 2.6，较早的 Python 版本仍可使用 >= 2.4。PyTorch 2.6 已支持
Python 3.13 的 `torch.compile`，见
[官方发布说明](https://pytorch.org/blog/pytorch2-6/)；项目默认仍关闭编译。

Linux **CPU** 环境可用：

```bash
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pip check
dapi2marker params --config configs/small.yaml
```

## RTX 4090 / CUDA 12.8

用户确认训练服务器使用 RTX 4090、CUDA 12.8。项目提供
`requirements-cu128.txt`，固定 **PyTorch 2.11.0 + cu128**；安装时只引入
项目需要的 `torch`。这个版本的 CUDA 12.8 安装方式见
[PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/#v2110)，
[官方 wheel 索引](https://download.pytorch.org/whl/cu128/torch/)也包含
Python 3.13 的 Linux x86_64 包。在已激活的项目环境中执行：

```bash
nvidia-smi
python -m pip install -r requirements-cu128.txt
python -m pip install -e '.[dev]'
python -m pip check
dapi2marker doctor --require-cuda
```

`requirements-cu128.txt` 固定的是 PyTorch CUDA 构建；其他依赖仍由
`pyproject.toml` 管理。安装项目时，已满足版本范围的 PyTorch 会保留。
如果服务器上已有经过验证的 cu128 PyTorch 环境，可以直接安装项目并
执行自检。其他显卡/驱动组合按
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装。

如果继续使用已经配置好的 Python 3.13.11 环境，也可以先确认其 PyTorch
构建，再通过 `python -m pip install -e '.[dev]'` 补齐本项目依赖；上述独立
环境方案便于维护和复现。

## 检查训练机器

自检无需数据或 VAE。它会输出 Python、依赖版本、PyTorch CUDA 构建、
`nvidia-smi` 驱动信息，以及每张可见显卡的名称、显存、计算能力和自动
选择的精度，并执行一次小型矩阵运算和反向传播。不能仅凭 CUDA 可用标志
就确认安装包能在显卡上运行。

```bash
dapi2marker doctor --require-cuda
```

通过条件是 `cuda_usable: true`，且每张可见 GPU 的
`kernel_check_passed: true`。`--require-cuda` 在 CUDA 不可用或内核检查
失败时返回非零退出码；省略该参数也可查看 CPU 环境报告。

RTX 4090 的标准显存为 24 GB、计算能力为 8.9，见
[NVIDIA 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)。
`configs/small_cuda.yaml` 和 `configs/smoke_cuda.yaml` 强制 CUDA，训练和
推理采用 `mixed_precision: auto`：检测原生 bf16 支持，支持时选 bf16，
否则选 fp16。预计 RTX 4090 会选择 bf16，以服务器自检结果为准。
VAE 仍冻结并使用 FP32；自动选择只改变 UNet 和条件编码器的混合精度。
实际精度写入结果和权重元数据；断点恢复也检查自动选择结果是否一致。

## 验证后训练

确认两个数据目录和本地 VAE 已放到配置指定的位置，先执行参数统计、
划分预览和两步 GPU smoke：

```bash
dapi2marker params --config configs/small_cuda.yaml
dapi2marker cross-validate --config configs/small_cuda.yaml --include-full --dry-run
dapi2marker train --config configs/smoke_cuda.yaml
```

GPU smoke 使用实际小模型和本地预训练 VAE、4 个训练 tile、2 个验证
tile、batch size 2、2 个训练步和 2 个 DDIM 步，结果存放于
`runs/smoke_cuda/fold_0/`。CPU smoke 仍使用 `configs/smoke.yaml`，两者
输出目录独立。重复测试需选择新输出目录或使用已有断点恢复。

检查成功后，可启动三折和独立全量模型：

```bash
dapi2marker cross-validate --config configs/small_cuda.yaml --include-full
```

GPU 正式配置继承同一个小模型和固定 ROI 划分，默认 batch size 8、
每个模型 136,800 个更新步。已返回的三折和全量训练产物使用了这个预算，
batch size 为 8、精度为 bf16；产物未记录显存峰值或吞吐率。新机器仍需
先跑 smoke。三折统计可用于另定全量训练预算，具体流程见
[使用说明](usage.md)。

已完成的本地测试环境是 Python 3.12/macOS CPU，见
[验证记录](validation.md)。CI 已增加 Linux/Python 3.13 测试项。2026-10-20
已检查返回的三折和全量 bf16 训练产物，参考指标见
[README](../README.md#reference-results)。产物未附完整软件版本清单，
不能把推荐安装版本当作该次训练的实际环境记录。
