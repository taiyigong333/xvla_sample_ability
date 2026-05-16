# 从 HDF5 文件夹直接生成 `meta.json`

## 1. 这份脚本是做什么的

我已经新增了一个脚本：

- [5_generate_meta_from_hdf5_folder.py](</F:/research/vla_team/X-VLA/doc/5_generate_meta_from_hdf5_folder.py>)

它的作用是：

- 扫描一个 HDF5 文件夹
- 自动收集其中所有 `.hdf5` / `.h5` 文件
- 自动生成你要的这种 `meta.json`

生成结果格式就是：

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

## 2. 脚本会顺手做哪些检查

脚本会验证每个 HDF5 是否包含：

- `puppet/end_effector`
- `puppet/joint_position`
- 你指定的图像 key，比如 `observations/images/cam_high`
- 你指定的语言 key，比如 `language_instruction`

如果缺失，它会打印提示。

## 3. 最常用的 set-a 生成命令

```powershell
python doc/5_generate_meta_from_hdf5_folder.py --input_dir F:/data/xvla/ur_set_a --output_path F:/research/vla_team/X-VLA/meta/robomind_ur_multi/robomind_ur_set_a.json --dataset_name robomind-ur-set-a --robot_type robomind-ur --observation_key observations/images/cam_high --language_instruction_key language_instruction
```

## 4. 最常用的 set-b 生成命令

```powershell
python doc/5_generate_meta_from_hdf5_folder.py --input_dir F:/data/xvla/ur_set_b --output_path F:/research/vla_team/X-VLA/meta/robomind_ur_multi/robomind_ur_set_b.json --dataset_name robomind-ur-set-b --robot_type robomind-ur --observation_key observations/images/cam_high --language_instruction_key language_instruction
```

## 5. 如果你有两个相机

如果你的 HDF5 同时稳定包含：

- `observations/images/cam_high`
- `observations/images/cam_wrist`

那么命令可以改成：

```powershell
python doc/5_generate_meta_from_hdf5_folder.py --input_dir F:/data/xvla/ur_set_b --output_path F:/research/vla_team/X-VLA/meta/robomind_ur_multi/robomind_ur_set_b.json --dataset_name robomind-ur-set-b --robot_type robomind-ur --observation_key observations/images/cam_high observations/images/cam_wrist --language_instruction_key language_instruction
```

## 6. 如果 HDF5 文件在子目录里

可以加 `--recursive`：

```powershell
python doc/5_generate_meta_from_hdf5_folder.py --input_dir F:/data/xvla/ur_set_b --output_path F:/research/vla_team/X-VLA/meta/robomind_ur_multi/robomind_ur_set_b.json --dataset_name robomind-ur-set-b --robot_type robomind-ur --observation_key observations/images/cam_high --language_instruction_key language_instruction --recursive
```

## 7. 关于 `language_instruction` 的特别提醒

这个脚本会按当前训练代码的预期检查：

- `language_instruction` 应该是 HDF5 dataset

如果你现在还是按 [3_代码采样示例.md](</F:/research/vla_team/X-VLA/doc/3_代码采样示例.md>) 里的写法，把它存成：

```python
f.attrs["language_instruction"] = instruction
```

那么脚本会提示它不是当前训练代码直接可读的 dataset。

你也可以临时加：

```powershell
--allow_language_attr
```

这样脚本在校验时会放宽检查，但它只是在生成 `meta.json` 时放宽，不代表训练代码已经能直接读取 attribute。

## 8. 方案 B 的完整使用顺序

1. 对 `ur_set_a` 跑一次脚本，生成 `robomind_ur_set_a.json`
2. 对 `ur_set_b` 再跑一次脚本，生成 `robomind_ur_set_b.json`
3. 在 [datasets/domain_config.py](</F:/research/vla_team/X-VLA/datasets/domain_config.py>) 里给：
   - `robomind-ur-set-a`
   - `robomind-ur-set-b`
   配采样权重
4. 训练时把 `--train_metas_path` 指向这两个 json 所在目录

## 9. 训练命令示例

```powershell
accelerate launch --mixed_precision bf16 train.py --models 2toINF/X-VLA-Pt --train_metas_path F:/research/vla_team/X-VLA/meta/robomind_ur_multi --output_dir runnings/robomind_ur_scheme_b_full --batch_size 16 --learning_rate 1e-4 --learning_coef 0.1 --iters 50000 --freeze_steps 1000 --warmup_steps 2000 --save_interval 5000
```
