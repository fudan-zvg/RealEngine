import json
import pandas as pd
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

path = "./output/nuplan_scene"

vfov = (-24.9, 2.0)  # (-31.96, 10.67)
hfov = (-180, 180)
rendered_gt = False

data = {
    'dir': [],
    'psnr': [],
    'l1': [],
}

for dir in sorted(os.listdir(path)):
    # if "2350" not in dir or "debug" in dir:
    #     continue

    data['dir'].append(dir)
    if os.path.exists(os.path.join(path, dir, 'eval/train_50000_render/metrics.json')):
        with open(os.path.join(path, dir, 'eval/train_50000_render/metrics.json'), "r") as f:
            F = json.load(f)
            psnr = F['psnr']
            l1 = F['l1']

            data['psnr'].append(psnr)
            data['l1'].append(l1)
    else:
        data['psnr'].append(-1)
        data['l1'].append(-1)

# 创建DataFrame
df = pd.DataFrame(data)

# 打印表格，使用tabulate格式化输出
print(df.to_string(index=False))
