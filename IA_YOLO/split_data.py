import os
import shutil
import random
from natsort import natsorted

def search_name_move_suffix(target_path, file_ext='png'):
    """搜索指定扩展名的文件"""
    all_files = os.listdir(target_path)
    files = [file.split('.')[0] for file in all_files if file.lower().endswith(file_ext.lower())]
    return natsorted(files)


def split_and_reorganize_dataset(images_dir, labels_dir, output_dir, ratios=(0.7, 0.2, 0.1)):
    """
    按照比例分割数据集并重新组织目录结构

    Args:
        images_dir: 原始图像目录
        labels_dir: 原始标签目录
        output_dir: 输出根目录
        ratios: 训练集、验证集、测试集的比例
    """
    # 获取所有文件名（不带扩展名）
    image_files = search_name_move_suffix(images_dir, 'png')

    # 打乱文件列表
    random.shuffle(image_files)

    # 计算各集合的数量
    total_count = len(image_files)
    train_count = int(total_count * ratios[0])
    val_count = int(total_count * ratios[1])
    test_count = total_count - train_count - val_count

    print(f"数据集总数: {total_count}")
    print(f"训练集: {train_count} ({train_count / total_count * 100:.1f}%)")
    print(f"验证集: {val_count} ({val_count / total_count * 100:.1f}%)")
    print(f"测试集: {test_count} ({test_count / total_count * 100:.1f}%)")

    # 分割文件列表
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    # 创建输出目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # 复制文件到新目录
    def copy_files(file_list, split_name):
        copied_count = 0
        for file_name in file_list:
            # 源文件路径
            src_image = os.path.join(images_dir, file_name + '.png')
            src_label = os.path.join(labels_dir, file_name + '.txt')

            # 目标文件路径
            dst_image = os.path.join(output_dir, split_name, 'images', file_name + '.png')
            dst_label = os.path.join(output_dir, split_name, 'labels', file_name + '.txt')

            # 复制图像文件
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
                copied_count += 1
            else:
                print(f"警告: 图像文件不存在 {src_image}")

            # 复制标签文件
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告: 标签文件不存在 {src_label}")

        return copied_count

    # 复制各集合文件
    print("\n正在复制文件...")
    train_copied = copy_files(train_files, 'train')
    val_copied = copy_files(val_files, 'val')
    test_copied = copy_files(test_files, 'test')

    print(f"\n复制完成:")
    print(f"训练集: {train_copied} 个图像")
    print(f"验证集: {val_copied} 个图像")
    print(f"测试集: {test_copied} 个图像")

    return train_files, val_files, test_files


def create_data_yaml(output_dir, class_names, train_dir='train', val_dir='val', test_dir='test'):
    """
    创建YOLO格式的data.yaml配置文件
    """
    yaml_content = f"""# YOLO dataset configuration file
path: {os.path.abspath(output_dir)}  # dataset root dir
train: {train_dir}/images  # train images
val: {val_dir}/images  # val images
test: {test_dir}/images  # test images

# number of classes
nc: {len(class_names)}

# class names
names: {class_names}
"""

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"配置文件已创建: {yaml_path}")
    return yaml_path


if __name__=="__main__":
    rtts_images_dir = r'E:\PythonProject\target_detection\data\RTTS\JPEGImages'
    rtts_labels_dir = r'E:\PythonProject\target_detection\data\RTTS\anns'  # 之前转换的标签目录
    output_dir = r'E:\PythonProject\target_detection\data\RTTS_split'
    # RTTS数据集类别
    rtts_class_names = ['car', 'person', 'bus', 'motorbike', 'bicycle']
    random.seed(42)
    # 执行数据集分割和重组
    print("开始分割数据集...")
    train_files, val_files, test_files = split_and_reorganize_dataset(
        rtts_images_dir, rtts_labels_dir, output_dir
    )

    # 创建data.yaml配置文件
    create_data_yaml(output_dir, rtts_class_names)
    def save_file_lists(output_dir, train_files, val_files, test_files):
        """保存各集合的文件列表"""
        with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
            for file in train_files:
                f.write(f"./{file}.png\n")

        with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
            for file in val_files:
                f.write(f"./{file}.png\n")

        with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
            for file in test_files:
                f.write(f"./{file}.png\n")


    save_file_lists(output_dir, train_files, val_files, test_files)
    print("文件列表已保存")
