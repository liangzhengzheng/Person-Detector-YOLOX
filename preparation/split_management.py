import os
import random
import sys
sys.path.append('.')

train_percent = 0.80
xmlfilepath = 'C:\\Users\\liangzheng\\PycharmProjects\\YOLOX-main\\datasets\\VOCdevkit\\VOC2007\\Annotations'
txtsavepath = 'C:\\Users\\liangzheng\\PycharmProjects\\YOLOX-main\\datasets\\VOCdevkit\\VOC2007\\ImageSets\\Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tr = int(num * train_percent)
train = random.sample(list, tr)

ftrain = open('C:\\Users\\liangzheng\\PycharmProjects\\YOLOX-main\\datasets\\VOCdevkit\\VOC2007\\ImageSets\\Main\\train.txt', 'w')
ftest = open('C:\\Users\\liangzheng\\PycharmProjects\\YOLOX-main\\datasets\\VOCdevkit\\VOC2007\\ImageSets\\Main\\test.txt', 'w')


for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in train:
        ftrain.write(name)
    else:
        ftest.write(name)


ftrain.close()
ftest.close()

