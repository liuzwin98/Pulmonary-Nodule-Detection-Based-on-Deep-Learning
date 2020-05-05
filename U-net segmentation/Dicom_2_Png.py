"""
读取dicom图像并将其转换为png图像

读取某文件夹内的所有dicom文件
:param src_dir: dicom文件夹路径
:return: dicom list
"""

import os
import SimpleITK
import pydicom
import numpy as np
import cv2
from tqdm import tqdm


def is_dicom_file(filename):
    # 判断某文件是否是dicom格式的文件

    file_stream = open(filename, 'rb')
    # seek() 方法用于移动文件读取指针到指定位置。
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False


def load_patient(src_dir):
    # 读取某文件夹内的所有dicom文件

    files = os.listdir(src_dir)
    slices = []
    for s in files:
        if is_dicom_file(src_dir + '/' + s):
            instance = pydicom.read_file(src_dir + '/' + s)
            slices.append(instance)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] \
                                 - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu_by_simpleitk(dicom_dir):
    # 读取某文件夹内的所有dicom文件,并提取像素值(-4000 ~ 4000)

    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(image)
    img_array[img_array == -2000] = 0
    return img_array


def normalize_hu(image):
    # 将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


if __name__ == '__main__':
    dicom_dir = './data/dicom_demo/'  # 读取dicom文件的元数据(dicom tags)
    slices = load_patient(dicom_dir)
    print('The number of dicom files : ', len(slices))

    image = get_pixels_hu_by_simpleitk(dicom_dir)  # 提取dicom文件中的像素值
    for i in tqdm(range(image.shape[0])):
        img_path = "./temp_dir/dcm_2_png/img_" + str(i).rjust(4, '0') + "_i.png"
        org_img = normalize_hu(image[i])  # 将像素值归一化到[0,1]区间
        cv2.imwrite(img_path, org_img * 255)  # 保存图像数组为灰度图(.png)
