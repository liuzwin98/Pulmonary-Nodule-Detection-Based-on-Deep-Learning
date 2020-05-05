# -- coding: utf-8 --
# 训练图像分割网络(u-net)模型
import csv
import glob
import random
import cv2
import numpy
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps
from typing import List, Tuple
from tensorflow import Tensor
from keras.optimizers import SGD
from keras.layers import Input, Conv2D, MaxPooling2D,\
                         UpSampling2D, merge, BatchNormalization, SpatialDropout2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter    # 高斯卷积核

MEAN_FRAME_COUNT = 1
CHANNEL_COUNT = 1
SEGMENTER_IMG_SIZE = 320
MODEL_DIR = './model/again/'   # 模型保存路径
BATCH_SIZE = 8
ELASTIC_INDICES = None

TRAIN_LIST = ''
VAL_LIST = ''
TRAIN_TEMP_DIR = './temp_dir/chapter4/'


# 数据增强
# 随机缩放图像的函数，用于数据增广(augmentation)
def random_scale_img(img, xy_range, lock_xy=False):
    if random.random() > xy_range.chance:
        return img

    if not isinstance(img, list):
        img = [img]

    import cv2
    scale_x = random.uniform(xy_range.x_min, xy_range.x_max)
    scale_y = random.uniform(xy_range.y_min, xy_range.y_max)
    if lock_xy:
        scale_y = scale_x

    org_height, org_width = img[0].shape[:2]
    xy_range.last_x = scale_x
    xy_range.last_y = scale_y

    res = []
    for img_inst in img:
        scaled_width = int(org_width * scale_x)
        scaled_height = int(org_height * scale_y)
        scaled_img = cv2.resize(img_inst, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
        if scaled_width < org_width:
            extend_left = (org_width - scaled_width) / 2
            extend_right = org_width - extend_left - scaled_width
            scaled_img = cv2.copyMakeBorder(scaled_img, 0, 0, extend_left, extend_right, borderType=cv2.BORDER_CONSTANT)
            scaled_width = org_width

        if scaled_height < org_height:
            extend_top = (org_height - scaled_height) / 2
            extend_bottom = org_height - extend_top - scaled_height
            scaled_img = cv2.copyMakeBorder(scaled_img, extend_top, extend_bottom, 0, 0, borderType=cv2.BORDER_CONSTANT)
            scaled_height = org_height

        start_x = (scaled_width - org_width) / 2
        start_y = (scaled_height - org_height) / 2
        tmp = scaled_img[start_y: start_y + org_height, start_x: start_x + org_width]
        res.append(tmp)

    return res


class XYRange:
    def __init__(self, x_min, x_max, y_min, y_max, chance=1.0):
        self.chance = chance
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.last_x = 0
        self.last_y = 0

    def get_last_xy_txt(self):
        res = "x_" + str(int(self.last_x * 100)).replace("-", "m") + "-" + "y_" + str(int(self.last_y * 100)).replace(
            "-", "m")
        return res


# 随机变换图像的函数，用于数据增广(augmentation)
def random_translate_img(img, xy_range, padding="constant"):
    if random.random() > xy_range.chance:
        return img
    if not isinstance(img, list):
        img = [img]

    org_height, org_width = img[0].shape[:2]
    translate_x = random.randint(xy_range.x_min, xy_range.x_max)
    translate_y = random.randint(xy_range.y_min, xy_range.y_max)
    trans_matrix = numpy.float32([[1, 0, translate_x], [0, 1, translate_y]])

    border_const = cv2.BORDER_CONSTANT
    if padding == "reflect":
        border_const = cv2.BORDER_REFLECT

    res = []
    for img_inst in img:
        img_inst = cv2.warpAffine(img_inst, trans_matrix, (org_width, org_height), borderMode=border_const)
        res.append(img_inst)
    if len(res) == 1:
        res = res[0]
    xy_range.last_x = translate_x
    xy_range.last_y = translate_y
    return res


# 随机旋转图像的函数，用于数据增广(augmentation)
def random_rotate_img(img, chance, min_angle, max_angle):
    if random.random() > chance:
        return img
    if not isinstance(img, list):
        img = [img]

    angle = random.randint(min_angle, max_angle)
    center = (img[0].shape[0] / 2, img[0].shape[1] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    res = []
    for img_inst in img:
        img_inst = cv2.warpAffine(img_inst, rot_matrix, dsize=img_inst.shape[:2], borderMode=cv2.BORDER_CONSTANT)
        res.append(img_inst)
    if len(res) == 0:
        res = res[0]
    return res


# 反转图像的函数，用于数据增广(augmentation)
def random_flip_img(img, horizontal_chance=0.0, vertical_chance=0.0):
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val)  # 0 = X axis, 1 = Y axis,  -1 = both
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
    return res


# 图像弹性变换的函数，用于数据增广(augmentation)
def elastic_transform(image, alpha, sigma, random_state=None):
    global ELASTIC_INDICES
    shape = image.shape

    if ELASTIC_INDICES is None:
        if random_state is None:
            random_state = numpy.random.RandomState(1301)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
        ELASTIC_INDICES = numpy.reshape(y + dy, (-1, 1)), numpy.reshape(x + dx, (-1, 1))
    return map_coordinates(image, ELASTIC_INDICES, order=1).reshape(shape)


# 随机改变亮暗、对比度和颜色等
def random_distort(img):
    # 随机改变亮度
    def random_brightness(img, lower=0.5, upper=1.5):
        e = numpy.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    # 随机改变对比度
    def random_contrast(img, lower=0.5, upper=1.5):
        e = numpy.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    # 随机改变颜色
    def random_color(img, lower=0.5, upper=1.5):
        e = numpy.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    numpy.random.shuffle(ops)

    if not isinstance(img, list):
        img = Image.fromarray(img)
        img = ops[0](img)
        img = ops[1](img)
        img = ops[2](img)
        res = img
    else:
        res = []
        for img_item in img:
            img_dist = Image.fromarray(img_item)
            img_dist = ops[0](img_dist)
            img_dist = ops[1](img_dist)
            img_dist = ops[2](img_dist)
            img_dist = numpy.asarray(img_dist)
            res.append(img_dist)
    return res


# 测试前图像准备—图像数据类型转换—通道归一化（灰度图）
# 图像类型转换，通道归一化（灰度图）
def prepare_image_for_net(img):
    img = img.astype(numpy.float)
    img /= 255.
    if len(img.shape) == 3:
        img = img.reshape(img.shape[-3], img.shape[-2], img.shape[-1])
    else:
        img = img.reshape(1, img.shape[-2], img.shape[-1], 1)
    return img


# 训练集图片预处理函数
def get_train_holdout_files():
    train_path = TRAIN_LIST
    val_path = VAL_LIST
    train_res = []
    holdout_res = []
    with open(train_path, 'r') as f:
        reader = csv.reader(f)
        samples_train = list(reader)
        random.shuffle(samples_train)
    with open(val_path, 'r') as f:
        reader = csv.reader(f)
        samples_holdout = list(reader)
        random.shuffle(samples_holdout)
    for img_path in samples_train:
        if len(img_path) == 0:
            print('space line, skip')
            continue
        overlay_path = img_path[0].replace("_img.png", "_mask.png")
        train_res.append((img_path[0], overlay_path))

    for img_path in samples_holdout:
        if len(img_path) == 0:
            print('space line, skip')
            continue
        overlay_path = img_path[0].replace("_img.png", "_mask.png")
        holdout_res.append((img_path[0], overlay_path))

    print("Train count: ", len(train_res), ", holdout count: ", len(holdout_res))
    return train_res, holdout_res


# 定义神经网络并训练
# unet模型损失函数
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 100) / (K.sum(y_true_f) + K.sum(y_pred_f) + 100)


# unet模型损失函数
def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = numpy.sum(y_true_f * y_pred_f)
    return (2. * intersection + 100) / (numpy.sum(y_true_f) + numpy.sum(y_pred_f) + 100)


# unet模型损失函数
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# 将每个epoch结束后的验证结果保存为图片
class DumpPredictions(Callback):
    def __init__(self, dump_filelist: List[Tuple[str, str]], model_type):
        super(DumpPredictions, self).__init__()
        self.dump_filelist = dump_filelist
        self.batch_count = 0
        if not os.path.exists(TRAIN_TEMP_DIR):
            os.mkdir(TRAIN_TEMP_DIR)
        for file_path in glob.glob(TRAIN_TEMP_DIR + "*.*"):
            os.remove(file_path)
        self.model_type = model_type

    def on_epoch_end(self, epoch, logs=None):
        model = self.model  # type: Model
        generator = image_generator(self.dump_filelist, 1, train_set=False)
        for i in range(0, 10):
            x, y = next(generator)
            y_pred = model.predict(x, batch_size=1)

            x = x.swapaxes(0, 3)
            x = x[0]
            # print(x.shape, y.shape, y_pred.shape)
            x *= 255.
            x = x.reshape((x.shape[0], x.shape[0])).astype(numpy.uint8)
            y *= 255.
            y = y.reshape((y.shape[1], y.shape[2])).astype(numpy.uint8)
            y_pred *= 255.
            y_pred = y_pred.reshape((y_pred.shape[1], y_pred.shape[2])).astype(numpy.uint8)
            cv2.imwrite(TRAIN_TEMP_DIR + "img_{0:03d}_{1:02d}_i.png".format(epoch, i), x)
            cv2.imwrite(TRAIN_TEMP_DIR + "img_{0:03d}_{1:02d}_o.png".format(epoch, i), y)
            cv2.imwrite(TRAIN_TEMP_DIR + "img_{0:03d}_{1:02d}_p.png".format(epoch, i), y_pred)


# 训练集数据加载器
def image_generator(batch_files, batch_size, train_set):
    # train_set：是否采用数据增强
    global ELASTIC_INDICES
    while True:
        if train_set:
            random.shuffle(batch_files)

        img_list = []
        overlay_list = []
        ELASTIC_INDICES = None
        for batch_file_idx, batch_file in enumerate(batch_files):
            images = []
            img = cv2.imread(batch_file[0], cv2.IMREAD_GRAYSCALE)
            images.append(img)
            overlay = cv2.imread(batch_file[1], cv2.IMREAD_GRAYSCALE)

            if train_set:
                if random.randint(0, 100) > 50:
                    for img_index, img in enumerate(images):
                        images[img_index] = elastic_transform(img, 128, 15)
                    overlay = elastic_transform(overlay, 128, 15)

                if True:
                    augmented = images + [overlay]
                    augmented = random_rotate_img(augmented, 0.8, -20, 20)
                    augmented = random_flip_img(augmented, 0.5, 0.5)
                    augmented = random_distort(augmented)

                    augmented = random_translate_img(augmented, XYRange(-30, 30, -30, 30, 0.8))
                    images = augmented[:-1]
                    overlay = augmented[-1]

            for index, img in enumerate(images):
                img = prepare_image_for_net(img)
                images[index] = img

            overlay = prepare_image_for_net(overlay)
            images3d = numpy.vstack(images)
            images3d = images3d.swapaxes(0, 3)

            img_list.append(images3d)
            overlay_list.append(overlay)
            if len(img_list) >= batch_size:
                x = numpy.vstack(img_list)
                y = numpy.vstack(overlay_list)
                # if len(img_list) >= batch_size:
                yield x, y
                img_list = []
                overlay_list = []


# 实现unet的网络结构，并加载预训练好的权重
def get_unet(learn_rate=0.0001) -> Model:
    inputs = Input((SEGMENTER_IMG_SIZE, SEGMENTER_IMG_SIZE, CHANNEL_COUNT))
    filter_size = 32
    growth_step = 32
    x = BatchNormalization()(inputs)
    conv1 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(x)
    conv1 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    pool1 = BatchNormalization()(pool1)
    filter_size += growth_step  # 通道filter_size：64
    conv2 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    filter_size += growth_step  # 通道filter_size：128
    conv3 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    filter_size += growth_step  # 通道filter_size：256
    conv4 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    # 通道filter_size：256
    conv5 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(filter_size, (3, 3), activation='relu', padding='same', name="conv5b")(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name="pool5")(conv5)
    pool5 = BatchNormalization()(pool5)

    conv6 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv2D(filter_size, (3, 3), activation='relu', padding='same', name="conv6b")(conv6)

    up6 = merge.concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=3)
    up6 = BatchNormalization()(up6)

    filter_size -= growth_step  # 通道filter_size：128
    conv66 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(up6)
    conv66 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(conv66)

    up7 = merge.concatenate([UpSampling2D(size=(2, 2))(conv66), conv4], axis=3)
    up7 = BatchNormalization()(up7)

    filter_size -= growth_step  # 通道filter_size：64
    conv7 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(conv7)

    up8 = merge.concatenate([UpSampling2D(size=(2, 2))(conv7), conv3], axis=3)
    up8 = BatchNormalization()(up8)
    filter_size -= growth_step  # 通道filter_size：32
    conv8 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(conv8)

    up9 = merge.concatenate([UpSampling2D(size=(2, 2))(conv8), conv2], axis=3)
    up9 = BatchNormalization()(up9)
    conv9 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(filter_size, (3, 3), activation='relu', padding='same')(conv9)

    up10 = UpSampling2D(size=(2, 2))(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(up10)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=SGD(lr=learn_rate, momentum=0.9, nesterov=True),
                  loss=dice_coef_loss, metrics=[dice_coef])
    # Adam(lr=1e-5)
    # ETA：Estimated Time of Arrival。Loss: 系统的损失。acc：Accuracy正确率。
    model.summary()
    return model


# unet的训练过程
def train_model(model_type, continue_from=None):
    batch_size = BATCH_SIZE
    train_files, holdout_files = get_train_holdout_files()

    train_gen = image_generator(train_files, batch_size, True)
    holdout_gen = image_generator(holdout_files, batch_size, False)  # holdout验证方式

    if continue_from is None:
        model = get_unet(0.001)
    else:
        model = get_unet(0.0005)
        model.load_weights(continue_from)

    checkpoint1 = ModelCheckpoint(
        MODEL_DIR + model_type + "_{epoch:02d}-{val_loss:.2f}.hd5", monitor='val_loss',
        verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    dumper = DumpPredictions(holdout_files[::10], model_type)   # callback：将验证结果保存为图片
    tensorbd = TensorBoard(log_dir='./logs', write_images=True)    # 记录信息，准备Tensorboard可视
    hist = model.fit_generator(train_gen, steps_per_epoch=200, epochs=80, validation_data=holdout_gen,
                               verbose=1, callbacks=[checkpoint1, tensorbd], validation_steps=10)

    # test_loss, test_acc = model.evaluate(test_images, test_labels)    # 测试数据通过get_train_holdout_files()处理
    # print('Test accuracy:', test_acc)

    plt.figure(1)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("./temp_dir/chapter5/Unet_loss_curve.jpg")  # 保存训练过程的学习效果曲线

    plt.figure(2)
    plt.plot(hist.history['dice_coef'])
    plt.plot(hist.history['val_dice_coef'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    # 保存训练过程的学习效果曲线
    plt.savefig("./temp_dir/chapter5/Unet_acc_curve.jpg")


if __name__ == "__main__":
    TRAIN_LIST = './data/chapter4/train_img.txt'
    VAL_LIST = './data/chapter4/val_img.txt'
    # train_model(model_type='u-net', continue_from='./model/u-net_02--0.78.hd5')
    train_model(model_type='u-net', continue_from='./model/unet.hd5')
    # train_model(model_type='u-net', continue_from=None)  # 从头训练

