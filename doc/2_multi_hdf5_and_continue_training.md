# 双 HDF5、采样比例、重新训练与继续训练说明

## 1. 先说结论

如果你要同时训练两个 UR HDF5 数据集，当前仓库有两种思路：

1. 不改源码：把两个数据集的 HDF5 路径都放进同一个 `meta.json`
2. 手动改配置后更规范：拆成两个 `meta.json`，再通过 `DATA_WEIGHTS` 控制采样权重

如果你只想让我写文档、不改代码，那么最稳妥的是先按第 1 种方式做。

## 2. 方案 A：不改源码，直接把两个 HDF5 数据集合并到同一个 `meta.json`

这是最省事的做法。

示例：

```json
{
  "dataset_name": "robomind-ur",
  "observation_key": [
    "observation/image0"
  ],
  "language_instruction_key": "language_instruction",
  "datalist": [
    "F:/data/ur_set_a/episode_000001.hdf5",
    "F:/data/ur_set_a/episode_000002.hdf5",
    "F:/data/ur_set_b/episode_000001.hdf5",
    "F:/data/ur_set_b/episode_000002.hdf5"
  ]
}
```

这样做的含义是：

- 两个数据集合并训练
- 训练时统一视作 `robomind-ur`
- 自动使用 `domain_id = 12`

### 这种方式的采样规律

这种方式本质上不是“数据集级别权重采样”，而是：

- 所有 episode 一起参与训练
- 采样频率主要由 episode 数量和轨迹长度共同决定

### 如果你想在不改源码的前提下近似控制采样比例

可以通过重复写某个数据集的路径来近似增大它的采样频率。

例如你想让 `set_a : set_b = 2 : 1`，可以把 `set_a` 的条目重复一遍：

```json
{
  "dataset_name": "robomind-ur",
  "observation_key": [
    "observation/image0"
  ],
  "language_instruction_key": "language_instruction",
  "datalist": [
    "F:/data/ur_set_a/episode_000001.hdf5",
    "F:/data/ur_set_a/episode_000002.hdf5",
    "F:/data/ur_set_a/episode_000001.hdf5",
    "F:/data/ur_set_a/episode_000002.hdf5",
    "F:/data/ur_set_b/episode_000001.hdf5",
    "F:/data/ur_set_b/episode_000002.hdf5"
  ]
}
```

注意：

- 这是近似控制，不是严格概率控制
- 因为每条轨迹能产出的训练样本数还和轨迹长度有关

## 3. 方案 B：两个 `meta.json` + 数据集级别采样权重

如果你希望“按数据集级别”更干净地控制采样概率，当前仓库的机制是：

- 读入多个 `meta.json`
- 每个 `meta.json` 对应一个数据集名
- 训练时按 `datasets/domain_config.py` 中的 `DATA_WEIGHTS[dataset_name]` 进行加权采样

### 这一方案有两个关键点

#### 关键点 1：两个 `meta.json` 的 `dataset_name` 不能都写成 `robomind-ur`

原因是当前加载逻辑会用 `dataset_name` 作为字典键。

如果两个文件都写：

```json
"dataset_name": "robomind-ur"
```

后一个会覆盖前一个。

#### 关键点 2：为了继续使用 UR 的 handler 和 `domain_id = 12`，要额外写 `robot_type`

推荐写法：

- `dataset_name` 负责区分两个数据集
- `robot_type` 固定写成 `robomind-ur`

例如第一个文件：

```json
{
  "dataset_name": "robomind-ur-set-a",
  "robot_type": "robomind-ur",
  "observation_key": [
    "observation/image0"
  ],
  "language_instruction_key": "language_instruction",
  "datalist": [
    "F:/data/ur_set_a/episode_000001.hdf5",
    "F:/data/ur_set_a/episode_000002.hdf5"
  ]
}
```

第二个文件：

```json
{
  "dataset_name": "robomind-ur-set-b",
  "robot_type": "robomind-ur",
  "observation_key": [
    "observation/image0"
  ],
  "language_instruction_key": "language_instruction",
  "datalist": [
    "F:/data/ur_set_b/episode_000001.hdf5",
    "F:/data/ur_set_b/episode_000002.hdf5"
  ]
}
```

### 然后你需要手动配置采样权重

当前仓库会读取：

- `datasets/domain_config.py` 里的 `DATA_WEIGHTS`

也就是说，如果你想要：

- `set-a` 采样概率 `0.7`
- `set-b` 采样概率 `0.3`

你需要手动把下面两项加进去：

```python
"robomind-ur-set-a": 0.7,
"robomind-ur-set-b": 0.3,
```

注意：

- 这里只需要加到 `DATA_WEIGHTS`
- 不需要给这两个别名再额外加 `DATA_DOMAIN_ID`
- 因为 `robot_type = "robomind-ur"` 时，训练仍然会自动使用 `domain_id = 12`

### 训练命令怎么写

把两个 `meta.json` 放进同一个目录，例如：

- `F:/research/vla_team/X-VLA/meta/multi_ur/robomind_ur_set_a.json`
- `F:/research/vla_team/X-VLA/meta/multi_ur/robomind_ur_set_b.json`

然后执行：

```powershell
accelerate launch --mixed_precision bf16 train.py --models 2toINF/X-VLA-Pt --train_metas_path F:/research/vla_team/X-VLA/meta/multi_ur --output_dir runnings/robomind_ur_multi_full --batch_size 16 --learning_rate 1e-4 --learning_coef 0.1 --iters 50000 --freeze_steps 1000 --warmup_steps 2000 --save_interval 5000
```

这里 `--train_metas_path` 指向目录即可，代码会自动读取其中所有 `.json` 文件。

## 4. 如何重新开始训练

“重新开始训练”最简单的理解就是：

- 不接着旧结果跑
- 直接从某个基座模型重新起一个新的训练目录

### 全量训练重新开始

如果你要从官方基座重新开始：

```powershell
accelerate launch --mixed_precision bf16 train.py --models 2toINF/X-VLA-Pt --train_metas_path F:/research/vla_team/X-VLA/meta/robomind_ur.json --output_dir runnings/robomind_ur_full_restart --batch_size 16 --learning_rate 1e-4 --learning_coef 0.1 --iters 50000 --freeze_steps 1000 --warmup_steps 2000
```

建议：

- 每次重新开始都换一个新的 `--output_dir`
- 这样最不容易和旧 checkpoint 混在一起

### LoRA 微调重新开始

当前仓库对应脚本是 `peft_train.py`：

```powershell
accelerate launch --mixed_precision bf16 peft_train.py --models 2toINF/X-VLA-Pt --train_metas_path F:/research/vla_team/X-VLA/meta/robomind_ur.json --output_dir runnings/robomind_ur_lora_restart --batch_size 16 --learning_rate 1e-4 --learning_coef 0.1 --iters 50000 --freeze_steps 1000 --warmup_steps 2000
```

## 5. 如何训练一个已经训练过的模型

这里要分成两类理解：

1. 全量继续训练
2. LoRA 继续训练

而且还要分清两种“继续”：

1. 只加载旧权重，然后开始一个新的训练阶段
2. 严格从上一次中断处恢复，包括优化器、学习率状态、全局步数

当前仓库对这两者的支持程度不一样。

## 6. 全量训练：继续训练一个已经训练过的模型

### 可以直接做的方式

当前 `train.py` 支持把 `--models` 指向一个已经训练出的 checkpoint 目录。

例如如果你之前训练出了：

- `runnings/robomind_ur_full/ckpt-50000`

那么你可以继续做第二阶段训练：

```powershell
accelerate launch --mixed_precision bf16 train.py --models runnings/robomind_ur_full/ckpt-50000 --train_metas_path F:/research/vla_team/X-VLA/meta/robomind_ur.json --output_dir runnings/robomind_ur_full_stage2 --batch_size 16 --learning_rate 5e-5 --learning_coef 0.1 --iters 30000 --freeze_steps 0 --warmup_steps 1000
```

这代表：

- 加载 `ckpt-50000` 的模型权重
- 在此基础上再训练一段

### 当前脚本的真实限制

当前 `train.py` 会：

- 加载模型权重
- 重新创建优化器
- 从新的 `global_step = 0` 开始计数

因此它是：

- “加载旧权重继续训练”

而不是：

- “严格断点续训”

也就是说，当前仓库默认不自动恢复：

- 优化器状态
- 学习率调度状态
- 上次训练的真实步数

## 7. LoRA 微调：如何训练一个已经训练过的模型

### 情况 A：基于某个已经训练过的全量模型，再做一轮新的 LoRA 微调

这是可以直接做的。

例如你已经有一个全量 checkpoint：

- `runnings/robomind_ur_full/ckpt-50000`

你可以把它当成 LoRA 的基座模型：

```powershell
accelerate launch --mixed_precision bf16 peft_train.py --models runnings/robomind_ur_full/ckpt-50000 --train_metas_path F:/research/vla_team/X-VLA/meta/robomind_ur.json --output_dir runnings/robomind_ur_lora_on_stage2 --batch_size 16 --learning_rate 1e-4 --learning_coef 0.1 --iters 30000 --freeze_steps 0 --warmup_steps 1000
```

这表示：

- 基座不再是 `2toINF/X-VLA-Pt`
- 而是你已经训练过的全量模型

### 情况 B：继续训练一个已经生成过的 LoRA adapter

当前 `peft_train.py` 默认流程是：

- `XVLA.from_pretrained(args.models)`
- 然后新建一套 LoRA adapter

也就是说，当前脚本没有提供一个明确的参数用于：

- 先加载旧的 LoRA adapter
- 再在其基础上继续训练

所以对“继续训练一个已经训练出的 LoRA adapter”这件事，当前仓库默认脚本并没有直接暴露出一个清晰的命令行入口。

换句话说：

- 新开一轮 LoRA 微调：支持
- 基于旧全量 checkpoint 再开一轮 LoRA：支持
- 严格接着某个旧 LoRA adapter 继续训练：当前默认脚本没有直接写好的入口

## 8. 推理时如何加载 LoRA

虽然训练脚本没有直接提供“继续旧 LoRA adapter 训练”的明确入口，但推理脚本已经支持加载 LoRA：

```powershell
python deploy.py --model_path runnings/robomind_ur_full/ckpt-50000 --LoRA_path runnings/robomind_ur_lora_restart/ckpt-30000 --output_dir logs/robomind_ur_server
```

含义是：

- `--model_path` 指向全量模型
- `--LoRA_path` 指向 LoRA adapter

## 9. 实操建议

如果你现在的目标是先把 UR 数据顺利训起来，建议顺序如下：

1. 先用单个 `meta.json` 跑通单数据集训练
2. 再把两个 HDF5 集合并到同一个 `meta.json`，先验证混合训练能正常启动
3. 如果你确实需要严格的数据集级别采样概率，再采用“双 `meta.json` + `DATA_WEIGHTS`”方式
4. 如果你要继续训练一个旧模型：
   - 全量模型：直接把 `--models` 指到旧 `ckpt-*`
   - LoRA：优先考虑“基于旧全量 checkpoint 重新开一轮新的 LoRA”

## 10. 最后再强调一次和你最相关的结论

对你这类数据：

- UR 单臂
- 6 维 TCP
- 1 维夹爪
- HDF5 已满足 `/puppet/end_effector` 和 `/puppet/joint_position`

当前最关键的配置只有两个：

1. `meta.json` 里把数据集标成 `robomind-ur`
2. 如果做双数据集并且想保持 `domain_id = 12`，双 `meta.json` 方案里一定要加：

```json
"robot_type": "robomind-ur"
```

这样训练时才会稳定走：

- UR 的 handler
- `domain_id = 12`
