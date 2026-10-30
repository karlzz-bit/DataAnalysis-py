import os
import shutil


def move_cpp_files(source_dir, target_dir):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.cpp'):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)

                # 转移文件
                shutil.move(source_file, target_file)
                print(f'Moved: {source_file} to {target_file}')


# 指定源目录和目标目录
source_directory = r'C:\Users\Administrator\AppData\Roaming\Code\User\History'
target_directory = r'E:\zz'
move_cpp_files(source_directory, target_directory)
