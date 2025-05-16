import json
import os
import glob

# 初始化所有指标的总和和计数器
metrics = {
    "Point Cloud": {"C-D": 0, "F-score": 0},
    "Depth": {"RMSE": 0, "MedAE": 0, "LPIPS": 0, "SSIM": 0, "PSNR": 0},
    "Intensity": {"RMSE": 0, "MedAE": 0, "LPIPS": 0, "SSIM": 0, "PSNR": 0}
}


def calculate_averages(directory):
    file_count = 0

    # 读取所有的JSON文件
    for filename in os.listdir(directory):  # ["1538-best", "1728-best", "1908-best", "3353-best"]:  # ["2350", "4950", "10750", "11400"]: #
        if '-wo-chamfer' not in filename:  # and filename not in ["1538-best", "1728-best", "1908-best", "3353-best"]:
            continue
        # print(filename)
        filepath = os.path.join(directory, filename, 'eval/test_refine_render/metrics.json')
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)

                # print(data['Point Cloud']['C-D'])

                # 累加各项指标
                for section, values in metrics.items():
                    for metric in values:
                        metrics[section][metric] += data[section][metric]
        except FileNotFoundError:
            print(filename)

        file_count += 1

    # 计算平均值
    average_metrics = {}
    for section, values in metrics.items():
        average_metrics[section] = {metric: value / file_count for metric, value in values.items()}

    # 打印结果
    for section, values in average_metrics.items():
        print(f"{section} Averages:")
        for metric, value in values.items():
            print(f"{metric}: {value}")
        print()


if __name__ == "__main__":
    directory = "./eval_output/kitti360_reconstruction/"
    calculate_averages(directory)
