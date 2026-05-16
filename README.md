# X-VLA 当前代码说明

这份仓库当前更适合作为本地训练、LoRA 微调和部署工作副本使用。根 README 只保留和当前代码直接相关的说明；上游论文式长 README 已归档到 [doc/0_上游README.md](doc/0_上游README.md)。

## 当前包含什么

- `deploy.py`：启动 XVLA 推理服务，支持 `--model_path`、`--processor_path`、`--LoRA_path`、`--host`、`--port`，并在 `--output_dir` 下写入 `info.json`。
- `train.py`：常规训练入口，使用 `accelerate` 和 `datasets.create_dataloader(...)` 读取 meta 并保存 checkpoint。
- `peft_train.py`：LoRA 微调入口，在训练流程上包了一层 `peft`。
- `datasets/`：数据读取、domain 配置、handler 注册与动作/观测预处理。
- `models/`：`XVLA`、processor、transformer、action space 等核心实现。
- `evaluation/`：不同 benchmark 和机器人域的评测与参考客户端。
- `doc/`：本地补充文档、示例 meta、工具脚本和归档说明。

## 目录速览

```text
deploy.py                      推理服务入口
train.py                       常规训练入口
peft_train.py                  LoRA 微调入口
datasets/                      数据集封装、domain handler、注册与工具函数
models/                        XVLA 模型、processor、transformer、action 相关实现
evaluation/                    各任务/机器人域评测脚本
logs/                          推理服务输出目录（默认可写入 info.json）
runnings/                      训练输出目录（checkpoint、tensorboard 等）
doc/                           本地文档与辅助脚本
```

## 常用命令

### 1. 安装环境

```bash
conda env create -f environment.yml
conda activate xvla-stable
```

或

```bash
pip install -r requirements.txt
```

### 2. 启动推理服务

```bash
python deploy.py --model_path /path/to/model --output_dir ./logs
```

如果 processor 或 LoRA 不和模型放在一起，可以额外传：

```bash
python deploy.py --model_path /path/to/model --processor_path /path/to/processor --LoRA_path /path/to/lora --port 8010
```

### 3. 常规训练

```bash
accelerate launch train.py --models /path/to/model --train_metas_path /path/to/meta.json --output_dir runnings
```

### 4. LoRA 微调

```bash
accelerate launch peft_train.py --models /path/to/model --train_metas_path /path/to/meta.json --output_dir runnings
```

## 数据接入位置

如果要接入新的机器人域或数据格式，主要看这几个位置：

- `datasets/domain_handler/`：新增或修改具体 handler。
- `datasets/domain_handler/registry.py`：注册 handler 名称。
- `datasets/domain_config.py`：补充 domain 配置。
- `--train_metas_path`：传入训练所需的 meta 文件。

## 文档索引

- [doc/0_上游README.md](doc/0_上游README.md)：上游原始 README 归档版
- [doc/1_robomind_ur_training.md](doc/1_robomind_ur_training.md)：RoboMind-UR 训练说明
- [doc/2_multi_hdf5_and_continue_training.md](doc/2_multi_hdf5_and_continue_training.md)：多 HDF5 与续训说明
- [doc/3_代码采样示例.md](doc/3_代码采样示例.md)：采样示例
- [doc/4_方案B_双meta与采样权重详细说明.md](doc/4_方案B_双meta与采样权重详细说明.md)：双 meta 与采样权重说明
- [doc/5_从HDF5文件夹生成meta_json说明.md](doc/5_从HDF5文件夹生成meta_json说明.md)：从 HDF5 文件夹生成 meta 的说明
- [doc/5_generate_meta_from_hdf5_folder.py](doc/5_generate_meta_from_hdf5_folder.py)：配套生成脚本
