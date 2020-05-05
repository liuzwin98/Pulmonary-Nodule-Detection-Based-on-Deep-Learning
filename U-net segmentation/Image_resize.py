import cv2  # 图像处理库 opencv
import os  # 处理操作系统 

# 修改之后的图片大小
size_ = (320, 320)
# directory为文件所在的目录， 
# 将会将修改后的图片保存在 directory下resized_image文件夹中

directory = r'D:/Image/'

image_to_save_directory = os.path.join(directory, 'resized_image/')
# 新建文件夹
if not os.path.isdir(image_to_save_directory):
    os.mkdir(image_to_save_directory)
file_names = os.listdir(directory)
# print(file_names)


for filename in file_names:
    file_path = os.path.join(directory, filename)
    # print(file_path)
    img = cv2.imread(file_path)
    im2 = cv2.resize(img, size_, interpolation=cv2.INTER_CUBIC)

    saved_path = os.path.join(image_to_save_directory, filename)
    cv2.imwrite(saved_path, im2)

print(' image  resized successfully')
