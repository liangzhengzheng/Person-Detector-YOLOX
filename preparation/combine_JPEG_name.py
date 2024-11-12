import os

file_seqs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# 设置目标文件夹路径
for file_seq in file_seqs:
    folder_path = fr'C:\Users\liangzheng\PycharmProjects\YOLOX-main\datasets\HIE\JPEGImages\{file_seq}'

# 获取文件夹名作为前缀
    folder_name = os.path.basename(folder_path)

# 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否是普通文件
        if os.path.isfile(file_path):
            # 新的文件名：将文件夹名添加到原文件名开头
            if not filename.startswith(folder_name + '_'):
                new_filename = folder_name + '_' + filename
                new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"已将文件重命名为: {new_filename}")
            else:
                print(f"文件 {filename} 已经包含文件夹名作为前缀，跳过重命名。")
