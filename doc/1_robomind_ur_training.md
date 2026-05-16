# Robomind-UR 训练说明

## 1. 你的数据是否可以直接用于 X-VLA

可以。

你给出的 HDF5 结构已经符合当前仓库里 `robomind-ur` 的读取逻辑：

- `/puppet/end_effector` 的形状是 `[T, 6]`
  - 含义是 `xyz + euler_xyz`
- `/puppet/joint_position` 的形状是 `[T, ...]`
  - 最后一个元素会被当成夹爪值
  - 期望是 `1/0`，通常可理解为 `1=闭合，0=张开`

当前代码中：

- `robomind-ur` 会走 `datasets/domain_handler/robomind.py`
- 训练时会自动使用 `domain_id = 12`
- 你不需要在训练命令里手动传 `12`

是否会自动使用 `12`，取决于 `meta.json` 里的数据集标识是否正确。

## 2. `robomind-ur` 在当前仓库中的真实含义

对单臂 UR 数据，当前处理逻辑会把每一帧转成：

- 左臂动作: `3(xyz) + 6(rotate6d) + 1(gripper) = 10 维`
- 右臂动作: 全 0 占位 10 维
- 最终拼成 `20` 维动作

也就是说，你的原始数据虽然是：

- 6 个 TCP 数值
- 1 个夹爪关键值

但进入模型前会被自动转换为 X-VLA 统一使用的 `EE6D 20 维动作空间`。这一部分你不需要额外改命令。

## 3. 训练时最关键的不是命令，而是 `meta.json`

当前仓库读取 HDF5 数据时，除了固定要求：

- `/puppet/end_effector`
- `/puppet/joint_position`

之外，还需要你在 `meta.json` 中告诉它：

- 图像数据在 HDF5 里的哪个 key
- 语言指令在 HDF5 里的哪个 key
- 哪些 HDF5 文件要参与训练

## 4. 单数据集可直接使用的 `meta.json` 模板

下面给你一个最直接的模板。

请把下面几处改成你自己的真实路径和真实 key：

- `observation_key`
- `language_instruction_key`
- `datalist`

```json
{
  "dataset_name": "robomind-ur",
  "observation_key": [
    "observation/image0"
  ],
  "language_instruction_key": "language_instruction",
  "datalist": [
    "F:/your_dataset/ur_hdf5/episode_000001.hdf5",
    "F:/your_dataset/ur_hdf5/episode_000002.hdf5",
    "F:/your_dataset/ur_hdf5/episode_000003.hdf5"
  ]
}
```

## 5. `meta.json` 每个字段怎么写

### `dataset_name`

单个 UR 数据集最稳妥的写法就是：

```json
"dataset_name": "robomind-ur"
```

这样训练时会自动走：

- `robomind-ur` 的 handler
- `domain_id = 12`

### `observation_key`

这是一个数组，里面写 HDF5 里图像序列的 key。

例如只有一个相机：

```json
"observation_key": [
  "observation/image0"
]
```

如果你有两个或三个相机，也可以写成：

```json
"observation_key": [
  "observation/image0",
  "observation/image1",
  "observation/image2"
]
```

注意：

- 这里的每个 key 都必须真实存在于 HDF5 中
- 这些图像数据必须能按时间索引
- 也就是代码会按 `images[v][idx]` 的方式取第 `idx` 帧

### `language_instruction_key`

这是语言指令在 HDF5 中的 key。

例如：

```json
"language_instruction_key": "language_instruction"
```

当前代码期望这里读出来的是：

- 一个标量字符串
- 或一个长度为 1 的字符串数组

如果你的每个 episode 只有一条固定指令，最简单的做法就是给每个 HDF5 存一个字符串数据集，例如：

- `language_instruction = "pick up the bottle"`

### `datalist`

这里放要训练的所有 HDF5 文件绝对路径。

例如：

```json
"datalist": [
  "F:/your_dataset/ur_hdf5/episode_000001.hdf5",
  "F:/your_dataset/ur_hdf5/episode_000002.hdf5"
]
```

## 6. 训练命令怎么写

先准备一个 `meta.json`，例如：

- `F:/research/vla_team/X-VLA/meta/robomind_ur.json`

然后在 PowerShell 里执行：

```powershell
accelerate launch --mixed_precision bf16 train.py --models 2toINF/X-VLA-Pt --train_metas_path F:/research/vla_team/X-VLA/meta/robomind_ur.json --output_dir runnings/robomind_ur_full --batch_size 16 --learning_rate 1e-4 --learning_coef 0.1 --iters 50000 --freeze_steps 1000 --warmup_steps 2000 --save_interval 5000
```

## 7. 这条命令里你最需要关心的参数

- `--models`
  - 基座模型
  - 常用就是 `2toINF/X-VLA-Pt`
- `--train_metas_path`
  - 指向你的 `meta.json`
- `--output_dir`
  - 保存训练结果的目录
- `--batch_size`
  - 批大小
- `--learning_rate`
  - 主学习率
- `--learning_coef`
  - soft prompt 的学习率倍率
- `--iters`
  - 总训练步数
- `--freeze_steps`
  - 前多少步冻结骨干
- `--warmup_steps`
  - 学习率 warmup 步数

## 8. 如果你想确认训练时是否真的用了 `robomind-ur -> 12`

当前训练流程不是手动在命令行写 `12`，而是数据加载时自动注入：

- 当 `dataset_name = "robomind-ur"` 时
- 样本会自动带上 `domain_id = 12`

所以你真正要保证的是：

- `meta.json` 里写的是 `robomind-ur`

而不是去训练命令里找一个 `--domain_id` 参数。当前仓库没有这个命令行参数。

## 9. 一个最小可执行示例

如果你的 HDF5 结构是：

- `/puppet/end_effector`
- `/puppet/joint_position`
- `/observation/image0`
- `/language_instruction`

那么一个最小版本的 `meta.json` 可以直接写成：

```json
{
  "dataset_name": "robomind-ur",
  "observation_key": [
    "observation/image0"
  ],
  "language_instruction_key": "language_instruction",
  "datalist": [
    "F:/data/ur/episode_000001.hdf5",
    "F:/data/ur/episode_000002.hdf5"
  ]
}
```

对应训练命令：

```powershell
accelerate launch --mixed_precision bf16 train.py --models 2toINF/X-VLA-Pt --train_metas_path F:/research/vla_team/X-VLA/meta/robomind_ur.json --output_dir runnings/robomind_ur_full --batch_size 16 --learning_rate 1e-4 --learning_coef 0.1 --iters 50000 --freeze_steps 1000 --warmup_steps 2000
```

## 10. 补充说明

如果你后面要做部署或推理，请手动传：

```json
"domain_id": 12
```

但训练阶段不需要手动写这个值。
