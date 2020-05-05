# 准备U-net训练数据

from scipy import ndimage as ndi
import numpy
import cv2

MASK_MARGIN = 5


def make_mask(v_center, v_diam, width, height):
    mask = numpy.zeros([height, width])
    v_xmin = numpy.max([0, int(v_center[0] - v_diam) - MASK_MARGIN])
    v_xmax = numpy.min([width - 1, int(v_center[0] + v_diam) + MASK_MARGIN])
    v_ymin = numpy.max([0, int(v_center[1] - v_diam) - MASK_MARGIN])
    v_ymax = numpy.min([height - 1, int(v_center[1] + v_diam) + MASK_MARGIN])
    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = v_x
            p_y = v_y
            if numpy.linalg.norm(numpy.array([v_center[0], v_center[1]])\
                                 - numpy.array([p_x, p_y]))<= v_diam * 2:
                mask[p_y, p_x] = 1.0   # 设置节点区域的像素值为1
    return mask


if __name__ == '__main__':
    imagePath = './data/chaper3_img_01.png'
    # 读取dicom文件的元数据(dicom tags)
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    print('before resize: ', img.shape)
    img_X = ndi.interpolation.zoom(img, [320/512, 320/512], mode='nearest')  # 被缩放成了320
    print('after resize: ', img_X.shape)
    # cv2.imwrite('./temp_dir/chapter3_img_XX.png', img_X)
    img_Y = make_mask((217, 160), 3, 320, 320)   # 结节信息由标注文件给出
    img_Y[img_Y < 0.5] = 0
    img_Y[img_Y > 0.5] = 255
    nodule_mask = img_Y.astype('uint8')
    # cv2.imwrite('./temp_dir/chapter3_img_Y.png', img_Y)
