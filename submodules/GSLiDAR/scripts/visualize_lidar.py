import json
import pandas as pd
from submodules.GSLiDAR.utils.system_utils import save_ply
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

path = "./eval_output/nuscenes_reconstruction"

vfov = (-24.9, 2.0)  # (-31.96, 10.67)
hfov = (-180, 180)
rendered_gt = False

data = {
    'dir': [],
    'mix_train': [],
    'mean_train': [],
    'median_train': [],
    'mix_test': [],
    'mean_test': [],
    'median_test': [],
    'refine': [],
}

for dir in sorted(os.listdir(path)):
    # if "2350" not in dir or "debug" in dir:
    #     continue

    try:
        with open(os.path.join(path, dir, 'eval/train_30000_render/metrics.json'), "r") as f:
            F = json.load(f)
            mix_train = F['Point Cloud mix']['C-D']
            mean_train = F['Point Cloud mean']['C-D']
            median_train = F['Point Cloud median']['C-D']

            data['mix_train'].append(mix_train)
            data['mean_train'].append(mean_train)
            data['median_train'].append(median_train)

        with open(os.path.join(path, dir, 'eval/test_30000_render/metrics.json'), "r") as f:
            F = json.load(f)
            mix_test = F['Point Cloud mix']['C-D']
            mean_test = F['Point Cloud mean']['C-D']
            median_test = F['Point Cloud median']['C-D']

            data['mix_test'].append(mix_test)
            data['mean_test'].append(mean_test)
            data['median_test'].append(median_test)

        with open(os.path.join(path, dir, 'eval/test_refine_render/metrics.json'), "r") as f:
            C_D_refine = json.load(f)['Point Cloud']['C-D']

            data['refine'].append(C_D_refine)

        data["dir"].append(dir)
    except:
        continue

# 创建DataFrame
df = pd.DataFrame(data)

# 打印表格，使用tabulate格式化输出
print(df.to_string(index=False))