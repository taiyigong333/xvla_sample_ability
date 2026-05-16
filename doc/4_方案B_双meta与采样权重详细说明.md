# 方案 B：两个 `meta.json` + 数据集级别采样权重

## 1. 这份说明解决什么问题

你现在希望做的是：

- 训练数据是 `robomind-ur`
- 同时使用两个 HDF5 数据集
- 不是简单把两个数据集混在同一个 `meta.json`
- 而是使用两个 `meta.json`
- 再通过数据集级别采样权重控制训练比例

这正是“方案 B”的目标。

## 2. 先和你的采集示例对齐

你给出的采集示例在 [3_代码采样示例.md](</F:/research/vla_team/X-VLA/doc/3_代码采样示例.md>) 中，核心 HDF5 结构是：

- `observations/images/cam_high`
- `observations/images/cam_wrist`（可选）
- `puppet/end_effector`
- `puppet/joint_position`
- `language_instruction`

这里要特别注意两个点。

### 注意 1：图像 key 要和 `meta.json` 完全一致

如果你的 HDF5 真的是按采集示例写出的，那么 `observation_key` 应该写成：

```json
[
  "observations/images/cam_high"
]
```

如果你的每一个 HDF5 文件都稳定包含腕部视角，也可以写成：

```json
[
  "observations/images/cam_high",
  "observations/images/cam_wrist"
]
```

但前提必须是：

- 同一个 `meta.json` 对应的所有 HDF5 文件都含有这两个 key

否则训练时会在读图像时直接报错。

### 注意 2：当前采集示例把 `language_instruction` 写成了 attribute

你的采集示例里这一行是：

```python
f.attrs["language_instruction"] = instruction
```

但当前训练代码读取语言时用的是：

- `f[key]`

也就是说，它读取的是 **HDF5 dataset**，不是 **HDF5 attribute**。

所以如果你完全按当前采集示例落盘，而不做额外处理，那么下面这种配置：

```json
"language_instruction_key": "language_instruction"
```

在训练时会失败，因为代码会尝试访问：

- `f["language_instruction"]`

但你实际存的是：

- `f.attrs["language_instruction"]`

这两个不是一回事。

## 3. 方案 B 在当前仓库里的一个关键限制

对 `robomind-ur` 来说，当前仓库里“方案 B”的概念是对的，但有一个实现层面的限制必须说清楚。

### 当前代码里，`dataset_name` 被同时拿来做了两件事

第一件事：

- 在 `datasets/dataset.py` 里，`dataset_name` 被用作多个 `meta.json` 的唯一键
- 所以两个 `meta.json` 不能都写成同一个 `dataset_name`

第二件事：

- 在 `datasets/domain_handler/robomind.py` 里，`dataset_name` 又被直接拿来判断是：
  - `robomind-ur`
  - `robomind-franka`
  - `robomind-agilex`
  - `robomind-franka-dual`

### 这会导致一个冲突

如果你为了方案 B 这样写：

- `robomind-ur-set-a`
- `robomind-ur-set-b`

它们在“多 meta 唯一标识”层面是正确的，但在 `RobomindHandler` 里又不再是精确的 `robomind-ur`，因此当前代码逻辑会识别失败。

### 所以你需要明确区分两件事

1. 数据集唯一名字
2. 机器人真实类型

这也是为什么我给你的示例 JSON 里会同时写：

- `dataset_name`
- `robot_type`

也就是：

- `dataset_name` 用来区分 set-a / set-b
- `robot_type` 用来表达它们本质上都是 `robomind-ur`

## 4. 这次给你的两个 `meta.json` 示例

我已经在 `doc/` 下放了两个示例文件：

- [4a_robomind_ur_set_a.meta.json](</F:/research/vla_team/X-VLA/doc/4a_robomind_ur_set_a.meta.json>)
- [4b_robomind_ur_set_b.meta.json](</F:/research/vla_team/X-VLA/doc/4b_robomind_ur_set_b.meta.json>)

### set-a 示例

```json
{
  "dataset_name": "robomind-ur-set-a",
  "robot_type": "robomind-ur",
  "observation_key": [
    "observations/images/cam_high"
  ],
  "language_instruction_key": "language_instruction",
  "datalist": [
    "F:/data/xvla/ur_set_a/demo_0000.hdf5",
    "F:/data/xvla/ur_set_a/demo_0001.hdf5",
    "F:/data/xvla/ur_set_a/demo_0002.hdf5"
  ]
}
```

### set-b 示例

```json
{
  "dataset_name": "robomind-ur-set-b",
  "robot_type": "robomind-ur",
  "observation_key": [
    "observations/images/cam_high"
  ],
  "language_instruction_key": "language_instruction",
  "datalist": [
    "F:/data/xvla/ur_set_b/demo_0000.hdf5",
    "F:/data/xvla/ur_set_b/demo_0001.hdf5",
    "F:/data/xvla/ur_set_b/demo_0002.hdf5"
  ]
}
```

## 5. 每个字段该怎么理解

### `dataset_name`

这里不是直接写 `robomind-ur`，而是写成：

- `robomind-ur-set-a`
- `robomind-ur-set-b`

目的只有一个：

- 让两个 `meta.json` 在同一次训练里不会互相覆盖

### `robot_type`

这个字段表达的是：

- 这两个数据集本质上都属于 `robomind-ur`

也就是说它们只是两个不同的数据源，不是两个不同的机器人类型。

### `observation_key`

这里我默认只写了主视角：

```json
[
  "observations/images/cam_high"
]
```

这是最稳妥的写法。

如果你确认：

- set-a 里所有 HDF5 都有 `cam_wrist`
- set-b 里所有 HDF5 也都有 `cam_wrist`

那么可以改成：

```json
[
  "observations/images/cam_high",
  "observations/images/cam_wrist"
]
```

### `language_instruction_key`

这里写的是：

```json
"language_instruction_key": "language_instruction"
```

但这成立的前提是：

- 你的 HDF5 里真的存在一个 dataset `/language_instruction`

如果你继续沿用采集示例里的 attribute 写法，那么这里只写对 key 名也没用，因为读取接口类型不匹配。

### `datalist`

这里直接列出每个数据集自己的 HDF5 文件路径。

推荐使用绝对路径，且统一用正斜杠：

- `F:/...`

这样最省心。

## 6. 采样权重怎么配

方案 B 的采样比例不是写在 `meta.json` 里的，而是写在：

- [datasets/domain_config.py](</F:/research/vla_team/X-VLA/datasets/domain_config.py>)

当前仓库读取的是：

- `DATA_WEIGHTS[dataset_name]`

所以如果你希望：

- set-a 采样概率约为 `0.7`
- set-b 采样概率约为 `0.3`

那你需要在 `DATA_WEIGHTS` 中增加：

```python
"robomind-ur-set-a": 0.7,
"robomind-ur-set-b": 0.3,
```

注意这里用的是：

- `dataset_name`

不是 `robot_type`。

## 7. 目录应该怎么组织

推荐你单独建一个目录放这两个 `meta.json`，例如：

```text
F:/research/vla_team/X-VLA/meta/robomind_ur_multi/
├── robomind_ur_set_a.json
└── robomind_ur_set_b.json
```

然后训练时把 `--train_metas_path` 指向这个目录。

## 8. 训练命令怎么写

如果两个 `meta.json` 都放在：

- `F:/research/vla_team/X-VLA/meta/robomind_ur_multi/`

那么训练命令可以写成：

```powershell
accelerate launch --mixed_precision bf16 train.py --models 2toINF/X-VLA-Pt --train_metas_path F:/research/vla_team/X-VLA/meta/robomind_ur_multi --output_dir runnings/robomind_ur_scheme_b_full --batch_size 16 --learning_rate 1e-4 --learning_coef 0.1 --iters 50000 --freeze_steps 1000 --warmup_steps 2000 --save_interval 5000
```

这里 `--train_metas_path` 指向的是目录，不是单个 json 文件。

## 9. 关于 `domain_id = 12`

你这次最关心的是 UR 对应的权重和域编号。

结论是：

- 你真正希望保持的是“UR 语义”
- 也就是训练样本仍然按 `robomind-ur` 处理
- 对应的域编号仍然应当是 `12`

所以从概念上说，方案 B 里虽然 `dataset_name` 会写成：

- `robomind-ur-set-a`
- `robomind-ur-set-b`

但它们的真实机器人类型都应该仍然是：

```json
"robot_type": "robomind-ur"
```

## 10. 你现在最容易踩的三个坑

### 坑 1：两个 `meta.json` 都写成 `dataset_name = "robomind-ur"`

这样会导致一个覆盖另一个，最终只训练到一个数据集。

### 坑 2：`observation_key` 写成了旧文档里的别名，而不是采集脚本真实输出的 key

基于你给的采集示例，应该优先写：

- `observations/images/cam_high`
- `observations/images/cam_wrist`

而不是：

- `observation/image0`
- `observation/image1`

除非你的实际 HDF5 就是后者。

### 坑 3：`language_instruction` 仍然是 HDF5 attribute

这是当前最现实的问题。

如果你的 HDF5 还在用：

```python
f.attrs["language_instruction"] = instruction
```

那么当前训练读取逻辑并不能直接用这个值。

## 11. 给你的建议顺序

如果你就是要走方案 B，我建议你按这个顺序检查：

1. 先确认 HDF5 中图像 key 是否真的是 `observations/images/cam_high`
2. 再确认语言指令是不是 dataset，而不是 attr
3. 再把两个 `meta.json` 分别写成 set-a / set-b
4. 再在 `DATA_WEIGHTS` 中给这两个 `dataset_name` 配权重
5. 最后再启动训练

## 12. 这份文档和旧文档的关系

之前的 [2_multi_hdf5_and_continue_training.md](</F:/research/vla_team/X-VLA/doc/2_multi_hdf5_and_continue_training.md>) 可以继续参考整体思路。

但如果你现在明确要走：

- 采集示例对应的 HDF5
- `robomind-ur`
- 两个 `meta.json`
- 数据集级别采样权重

那么请以这份文档为准，因为它补充了两个更贴近当前代码实际情况的关键点：

1. `language_instruction` 是 dataset 还是 attr
2. `dataset_name` 和 `robot_type` 在方案 B 中的职责分离
