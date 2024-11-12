import os
import random

train_seqs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
val_seqs = [15,16,17,18,19]
root_dir = 'C:\\Users\\liangzheng\\PycharmProjects\\YOLOX-main'
xml_file_path = os.path.join(root_dir,'./datasets/HIE/Annotations')
txt_save_path = os.path.join(root_dir,'./datasets/HIE/ImageSets/Main')
total_xml = os.listdir(xml_file_path)

train_file = os.path.join(txt_save_path,'train.txt')
val_file = os.path.join(txt_save_path,'val.txt')

train_list = []
val_list = []

for train_seq in train_seqs:
    for xml in total_xml:
        file_name, file_extension = os.path.splitext(xml)
        if  int(file_name.split('_')[0]) == train_seq:
            train_list.append(file_name)

for val_seq in val_seqs:
    for xml in total_xml:
        file_name, file_extension = os.path.splitext(xml)
        if int(file_name.split('_')[0]) == val_seq:
            val_list.append(file_name)

with open(train_file, 'w') as f_train:
    for item in train_list:
        f_train.write(f"{item}\n")

with open(val_file, 'w') as f_val:
    for item in val_list:
        f_val.write(f"{item}\n")









