import cv2
import os
import xml.etree.ElementTree as ET
from natsort import natsorted

def search_name_move_suffix(target_path, file_ext='png'):
    all_files = os.listdir(target_path)
    png_files = [file.split('.')[0] for file in all_files if file.lower().endswith(file_ext)]
    return natsorted(png_files)

def xml_to_yolo_txt(xml_path, txt_path, img_width, img_height, class_names):
    """
    将XML标注文件转换为YOLO格式的TXT文件

    Args:
        xml_path: XML文件路径
        txt_path: 输出的TXT文件路径
        img_width: 图像宽度
        img_height: 图像高度
        class_names: 类别名称列表
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                # 获取类别名称
                class_name = obj.find('name').text
                if class_name not in class_names:
                    continue

                class_id = class_names.index(class_name)

                # 获取边界框坐标
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # 转换为YOLO格式 (中心点坐标和宽高，归一化到0-1)
                x_center = (xmin + xmax) / 2.0 / img_width
                y_center = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # 写入TXT文件
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    except Exception as e:
        print(f"Error processing {xml_path}: {e}")


if __name__ == "__main__":
    rtts_images_dir = r'E:\PythonProject\target_detection\data\RTTS\JPEGImages'
    rtts_anns_dir = r'E:\PythonProject\target_detection\data\RTTS\Annotations'
    rtts_save_dir = r'E:\PythonProject\target_detection\data\RTTS\anns'
    os.makedirs(rtts_save_dir, exist_ok=True)
    # RTTS数据集的类别名称
    rtts_class_names = ['car', 'person', 'bus', 'motorbike', 'bicycle']
    rtts_file_list = search_name_move_suffix(rtts_images_dir, 'png')
    print(f"找到 {len(rtts_file_list)} 个图像文件")
    for image_name in rtts_file_list:
        image_path = os.path.join(rtts_images_dir, image_name + '.png')
        xml_path = os.path.join(rtts_anns_dir, image_name + '.xml')
        txt_path = os.path.join(rtts_save_dir, image_name + '.txt')
        if not os.path.exists(xml_path):
            print(f"警告: {xml_path} 不存在，跳过")
            continue
        image = cv2.imread(image_path)
        h, w, c = image.shape
        # 转换XML为YOLO TXT格式
        xml_to_yolo_txt(xml_path, txt_path, w, h, rtts_class_names)
        print(f"已转换: {image_name}")
    print("转换完成！")
