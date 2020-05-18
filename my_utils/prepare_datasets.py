# 运行成功后会生成如下目录结构的文件夹：
# trainval/
#    -images
#        -0001.jpg
#        -0002.jpg
#        -0003.jpg
#    -labels
#        -0001.txt
#        -0002.txt
#        -0003.txt
# 将trainval文件夹打包并命名为trainval.zip, 上传到OBS中以备使用。
import os
import codecs
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
import argparse


def get_classes(classes_path):
    '''loads the classes'''
    with codecs.open(classes_path, 'r', 'utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def creat_label_txt(soure_datasets, new_datasets):
    annotations = os.path.join(soure_datasets, 'VOC2007\Annotations')
    txt_path = os.path.join(new_datasets, 'labels')
    class_names = get_classes(os.path.join(soure_datasets, 'train_classes.txt'))

    xmls = os.listdir(annotations)
    for xml in tqdm(xmls):
        txt_anno_path = os.path.join(txt_path, xml.replace('xml', 'txt'))
        xml = os.path.join(annotations, xml)
        tree = ET.parse(xml)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        line = ''
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in class_names:
                print('name error', xml)
                continue
            cls_id = class_names.index(cls)
            xmlbox = obj.find('bndbox')
            box = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                   int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
            width = round((box[2] - box[0]) / w, 6)
            height = round((box[3] - box[1]) / h, 6)
            x_center = round(((box[2] + box[0]) / 2) / w, 6)
            y_center = round(((box[3] + box[1]) / 2) / h, 6)
            line = line + str(cls_id) + ' ' + ' '.join(str(v) for v in [x_center, y_center, width, height])+'\n'
            if box[2] > w or box[3] > h:
                print('Image with annotation error:', xml)
            if box[0] < 0 or box[1] < 0:
                print('Image with annotation error:', xml)
        with open(txt_anno_path, 'w') as f:
            f.writelines(line)


def creat_new_datasets(source_datasets, new_datasets):
    if not os.path.exists(source_datasets):
        print('could find source datasets, please make sure if it is exist')
        return

    if new_datasets.endswith('trainval'):
        if not os.path.exists(new_datasets):
            os.makedirs(new_datasets)
        os.makedirs(new_datasets + '\labels')
        print('copying images......')
        shutil.copytree(source_datasets + '\VOC2007\JPEGImages', new_datasets + '\images')
    else:
        print('最后一级目录必须为trainval,且为空文件夹')
        return
    print('creating txt labels:')
    creat_label_txt(source_datasets, new_datasets)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--soure_datasets", "-sd", type=str, help="SODiC官方原始数据集解压后目录")
    parser.add_argument("--new_datasets", "-nd", type=str, help="新数据集路径，以trainval结尾且为空文件夹")
    opt = parser.parse_args()
    # creat_new_datasets(opt.soure_datasets, opt.new_datasets)

    soure_datasets = r'D:\trainval'
    new_datasets = r'D:\SODiC\trainval'
    creat_new_datasets(soure_datasets, new_datasets)
