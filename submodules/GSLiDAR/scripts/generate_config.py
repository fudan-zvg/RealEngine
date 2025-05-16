import shutil


def modify_yaml_sequence_id(input_file, type, seq):
    if type != 'nuScenes-mini':
        output_file = f"./configs/kitti360_nvs_{seq}.yaml"  # 输出YAML文件路径
    else:
        output_file = f"./configs/nuScenes_mini_nvs_{seq}.yaml"  # 输出YAML文件路径

    # 复制YAML文件到新的文件
    shutil.copy(input_file, output_file)

    with open(output_file, 'r') as file:
        lines = file.readlines()

        # 遍历文件的每一行并查找sequence_id
    with open(output_file, 'w') as file:
        for line in lines:
            if type == 'nuScenes-mini':
                if line.strip().startswith('scene_type:'):
                    line = f'scene_type: \"nuScenes-mini\"\n'
                elif line.strip().startswith('resolution_scales:'):
                    line = f'resolution_scales: [1, 2]\n'
                elif line.strip().startswith('vfov:'):
                    line = f'vfov: [-30.0, 10.0]\n'

            if line.strip().startswith('sequence_id:'):
                # 替换sequence_id的值
                line = f'sequence_id: \"{seq}\"\n'
            elif line.strip().startswith('dynamic:'):
                line = f"dynamic: {True if type == 'dynamic' or type == 'nuScenes-mini' else False}\n"
            # 写回文件
            file.write(line)


if __name__ == '__main__':
    # input_file = './configs/kitti360_nvs_2350.yaml'  # 输入YAML文件路径

    # dic = {  # 'static': [1538, 1728, 3353],
    #     'dynamic': [4950, 8120, 10200, 10750, 11400],}

    input_file = './configs/nuScenes_mini_nvs_3180.yaml'
    dic = {"nuScenes-mini": ["0450", "1250", "1600", "2200"]}

    for key in dic.keys():
        for seq in dic[key]:
            modify_yaml_sequence_id(input_file, key, str(seq))
