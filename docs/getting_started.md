# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

## Train BEVFormer with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 8
```

Note 1: Our default setting requires at least **28000M** GPU memory.

Note 2: Base 使用 A100(80GB)*8 大约需要训练56个小时

## Eval BEVFormer with 8 GPUs
```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./path/to/ckpts.pth 8
```
Note: using 1 GPU to eval can obtain slightly higher performance because continuous video may be truncated with multiple GPUs. By default we report the score evaled with 8 GPUs.

### Eval Results
See `results_nusc.json` in `.../BEVFormer/test`

Structure of the folder:
```
test
└── bevformer_base
    └── Thu_Sep_28_09_35_31_2023
        └── pts_bbox
            ├── metrics_details.json
            ├── metrics_summary.json
            ├── plots
            └── results_nusc.json
```

## Common Errors
### 1. TypeError: FormatCode() got an unexpected keyword argument 'verify'
Fix: 降低yapf版本

```
pip install yapf==0.40.1
```
### 2. RuntimeError: CUDA out of memory
Fix 1: Train the model with a GPU which has at least 28000M memory

Fix 2: Change the config to `bevformer_small.py` or `bevformer_tiny.py`

Fix 3(?): Just skip the process of training, and use the ckpt that offered in README.md instead.

### 3. RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central
Fix: 验证和重新下载模型/数据文件。有时候在下载或上传到autoDL的过程中，文件可能会损坏。

# Using FP16 to train the model.
The above training script can not support FP16 training, 
and we provide another script to train BEVFormer with FP16.

```
./tools/fp16/dist_train.sh ./projects/configs/bevformer_fp16/bevformer_tiny_fp16.py 8
```


# Visualization 

see [visual.py](../tools/analysis_tools/visual.py)

modify:
```
# 修改1 替换主函数
import os

if __name__ == '__main__':
    # 数据集路径，使用mini就用v1.0-mini, 使用full就用v1.0-trainval
    nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=False)
    # results_nusc.json路径
    bevformer_results = mmcv.load('test/bevformer_base/Thu_Sep_28_09_35_31_2023/pts_bbox/results_nusc.json')
    # 添加result目录
    save_dir = "result"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sample_token_list = list(bevformer_results['results'].keys())
    
    for id in range(0, 10):
        render_sample_data(sample_token_list[id], pred_data=bevformer_results, out_path=os.path.join(save_dir, sample_token_list[id]))

# 修改2：将visual.py中的下面2句注释掉,就不用每次关闭当前显示窗口生成下一张图
# if verbose:
#     plt.show()
```

run:
```
python tools/analysis_tools/visual.py
```

