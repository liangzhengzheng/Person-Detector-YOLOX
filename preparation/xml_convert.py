import xml.dom.minidom
import os
import xml.etree.ElementTree as ET

path = r'C:\Users\liangzheng\PycharmProjects\YOLOX-main\datasets\HIE\Annotations'  # xml文件存放路径
sv_path = r'C:\Users\liangzheng\PycharmProjects\YOLOX-main\datasets\VOCdevkit\VOC2007\anno_with_difficult'  # 修改后的xml文件存放路径
files = os.listdir(path)

for xmlFile in files:
    file_name, file_extension = os.path.splitext(xmlFile)
    video = file_name.split('_')[0]
    frame = file_name.split('_')[2]
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    object = root.getElementsByTagName('object')
    item = root.getElementsByTagName('path')  # 获取path这一node名字及相关属性值
    name = root.getElementsByTagName('filename')
    folder = root.getElementsByTagName('folder')
    source = root.getElementsByTagName('source')
    sub = root.getElementsByTagName('name')
    for so in source:
        annotation = dom.createElement('annotation')
        annotation_text = dom.createTextNode("HIE")
        annotation.appendChild(annotation_text)
        so.appendChild(annotation)
    for i in item:
        i.firstChild.data = f'C:\\Users\\liangzheng\\PycharmProjects\\YOLOX-main\\datasets\\VOCdevkit\\VOC2007\\JPEGImages\\' + f"{video}" + '_frame_' + f"{int(frame):06d}" + '.jpg'
    for n in name:
        n.firstChild.data = f"{video}" + '_frame_' + f"{int(frame):06d}" + '.jpg'
    for f in folder:
        f.firstChild.data = 'JPEGImages'
    for s in sub:
        s.firstChild.data = 'person'
    for obj in object:
        pose = obj.getElementsByTagName('pose')
        truncated = obj.getElementsByTagName('truncated')
        #difficult = obj.getElementsByTagName('difficult')
        action = obj.getElementsByTagName('action')
        if pose:
            for po in pose:
                obj.removeChild(po)
        if truncated:
            for tr in truncated:
                obj.removeChild(tr)
        #if difficult:
        #    for di in difficult:
        #       obj.removeChild(di)

        if action:
            for a in action:
                obj.removeChild(a)
        bndbox = obj.getElementsByTagName('bndbox')[0]
        for tag in ['Nose', 'Chest', 'Right-shoulder', 'Right-elbow', 'Right-wrist',
                'Left-shoulder', 'Left-elbow', 'Left-wrist', 'Right-hip', 'Right-knee',
                'Right-ankle', 'Left-hip', 'Left-knee', 'Left-ankle']:
            child = bndbox.getElementsByTagName(tag)
            if child:
                # 删除该子节点
                bndbox.removeChild(child[0])
    with open(os.path.join(sv_path, xmlFile), 'w', encoding='utf-8') as fh:
        dom.writexml(fh)



