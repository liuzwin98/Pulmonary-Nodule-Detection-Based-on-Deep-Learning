import os
import settings as st
import numpy as np
from glob import glob
from scipy import ndimage as nd
# from scipy.misc import imsave

import scipy.stats
import pickle
save_data_path = "G:/DL Project/FN/data/"

slice_z = 28
slice_y = 42
slice_x = 42
PATIENT_NUM = 46  # 每个病例有1000个左右的可疑结节（class=0, 1）


def load_raw_data():
    import SimpleITK as sitk
    import csv

    # raw_path = st.data_path+"raw/"
    raw_path = st.data_path
    csv_path = "G:\\LUNA_Dataset\\candidates_V2.csv"

    def load_itk_image(filename):
        itkimage = sitk.ReadImage(filename)

        numpyImage = sitk.GetArrayFromImage(itkimage)
        numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
        numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
        return numpyImage, numpyOrigin, numpySpacing

    def normalizePlanes(npzarray):
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray > 1] = 1.
        npzarray[npzarray < 0] = 0.
        return npzarray

    def worldToVoxelCoord(worldCoord, origin, spacing):
        stretchedVoxelCoord = np.absolute(worldCoord - origin)
        voxelCoord = stretchedVoxelCoord / spacing
        voxelCoord = np.round(voxelCoord)
        return voxelCoord

    fn_list = {}
    fn_cnt = 0  # 病例数量

    candidates = []
    with open(csv_path, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            if line[0] == "seriesuid":
                continue
            if line[0] not in fn_list:
                fn_list[line[0]] = fn_cnt
                fn_cnt += 1
            candidates.append([fn_list[line[0]], int(line[4]), float(line[3]), float(line[2]), float(line[1])])
            if fn_cnt > PATIENT_NUM:  # 手动设置读取多少病例
                break
    candidates = np.array(candidates, dtype=np.float32)
    # print(len(candidates))
    # print(len(fn_list))
    # （28，42，42）要提取的表示cube大小
    patch_dat = np.zeros(shape=(len(candidates), slice_z, slice_y, slice_x), dtype=np.float32)
    patch_lbl = np.zeros(shape=(len(candidates), 8), dtype=np.float32)
    patch_cnt = 0

    for cnt, (fn, fn_cnt) in enumerate(fn_list.items()):
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(raw_path + "%s.mhd" % fn)
        numpyImage = numpyImage.astype('float32')
        whole_brain = normalizePlanes(numpyImage)

        cand = candidates[candidates[:, 0] == fn_cnt, 1:]
        voxel_coord = worldToVoxelCoord(cand[:, 1:], numpyOrigin, numpySpacing)

        for vox_cnt, ((lbl, oZ, oY, oX), (Z, Y, X)) in enumerate(zip(cand, voxel_coord)):
            X, Y, Z = int(X), int(Y), int(Z)
            dat = whole_brain[max(Z - slice_z // 2, 0):Z + slice_z // 2,
                  max(Y - slice_y // 2, 0):Y + slice_y // 2, max(X - slice_x // 2, 0):X + slice_x // 2]

            if np.any(dat.shape != (slice_z, slice_y, slice_x)):
                dZ, dY, dX = whole_brain.shape
                dat = np.pad(dat, ((max(slice_z // 2 - Z, 0), slice_z // 2 - min(dZ - Z, slice_z // 2)),
                                   (max(slice_y // 2 - Y, 0), slice_y // 2 - min(dY - Y, slice_y // 2)),
                                   (max(slice_x // 2 - X, 0), slice_x // 2 - min(dX - X, slice_x // 2))),
                             mode="constant", constant_values=0.)

            patch_dat[patch_cnt] = dat
            patch_lbl[patch_cnt] = [lbl, fn_cnt, Z, Y, X, oZ, oY, oX]
            patch_cnt += 1
        print(cnt)
    np.save(save_data_path + "patch_dat.npy", patch_dat)
    np.save(save_data_path + "patch_lbl.npy", patch_lbl)
    print('data loading done!')


def augment_data():
    if not os.path.exists(save_data_path + "patch_lbl.npy"):
        print("patch_lbl.npy not exist")
        print("Start loading raw data...")
        load_raw_data()

    patch_dat = np.load(save_data_path + "patch_dat.npy", mmap_mode="r")
    patch_lbl = np.load(save_data_path + "patch_lbl.npy", mmap_mode="r")

    augment_dat = np.zeros(shape=(np.count_nonzero(patch_lbl[:, 0]), 3, 3, 3, 4, slice_z-2, slice_y-2, slice_x-2), dtype=np.float32)
    augment_lbl = np.zeros(shape=(len(augment_dat), 3, 3, 3, 4, patch_lbl.shape[-1]), dtype=np.float32)

    for cnt, idx in enumerate(np.argwhere(patch_lbl[:, 0]).squeeze()):
        dat = patch_dat[idx]
        dat = np.stack(
            [dat, np.rot90(dat, 1, axes=(1, 2)), np.rot90(dat, 2, axes=(1, 2)), np.rot90(dat, 3, axes=(1, 2))])

        for z in range(3):
            for y in range(3):
                for x in range(3):
                    augment_dat[cnt, z, y, x] = dat[:, z:(z + slice_z-2), y:(y + slice_y-2), x:(x + slice_x-2)]
                    # print(cnt, z, y, x, 'done')
        augment_lbl[cnt] = patch_lbl[idx]

    np.save(save_data_path + "augment_dat.npy", augment_dat)
    np.save(save_data_path + "augment_lbl.npy", augment_lbl)
    print('data augmentation done!')


def split_fold():
    if not os.path.exists(save_data_path + "augment_lbl.npy"):
        augment_data()

    patch_lbl = np.load(save_data_path + "patch_lbl.npy", mmap_mode="r")
    augment_lbl = np.load(save_data_path + "augment_lbl.npy", mmap_mode="r")
    augment_lbl = augment_lbl.reshape(-1, augment_lbl.shape[-1])
    augment_lbl = np.concatenate((patch_lbl, augment_lbl))

    if not os.path.exists(save_data_path + "patch_dat_resize.npy"):
        patch_dat = np.load(save_data_path + "patch_dat.npy", mmap_mode="r")[:, 1:-1, 1:-1, 1:-1]

        btm_dat_r = patch_dat[:, 10:-10, 10:-10, 10:-10]
        mid_dat_r = patch_dat[:, 8:-8, 5:-5, 5:-5]

        dat_resize_r = np.zeros(shape=(len(patch_dat), 3, 6, 20, 20), dtype=np.float32)

        for cnt, (bd, md, td) in enumerate(zip(btm_dat_r, mid_dat_r, patch_dat)):
            dat_resize_r[cnt, 0] = bd
            dat_resize_r[cnt, 1] = nd.interpolation.zoom(md, zoom=(6 / 10, 20 / 30, 20 / 30), mode="nearest")
            dat_resize_r[cnt, 2] = nd.interpolation.zoom(td, zoom=(6 / (slice_z-2), 20 / (slice_y-2), 20 / (slice_x-2)), mode="nearest")

            if cnt % 100 == 0:
                print("\rReal %d/%d" % (cnt, len(dat_resize_r)), end="")

        dat_resize_r = np.transpose(dat_resize_r, (0, 1, 3, 4, 2))

        np.save(save_data_path + "patch_dat_resize.npy", dat_resize_r)
        print('\npatch_dat_resize done!')
    else:
        dat_resize_r = np.load(save_data_path + "patch_dat_resize.npy", mmap_mode="r")

    if not os.path.exists(save_data_path + "aug_dat_resize.npy"):
        augment_dat = np.load(save_data_path + "augment_dat.npy", mmap_mode="r")
        augment_dat = augment_dat.reshape(-1, slice_z-2, slice_y-2, slice_x-2)

        btm_dat_a = augment_dat[:, 10:-10, 10:-10, 10:-10]
        mid_dat_a = augment_dat[:, 8:-8, 5:-5, 5:-5]

        dat_resize_a = np.zeros(shape=(len(augment_dat), 3, 6, 20, 20), dtype=np.float32)

        for cnt, (bd, md, td) in enumerate(zip(btm_dat_a, mid_dat_a, augment_dat)):
            dat_resize_a[cnt, 0] = bd
            dat_resize_a[cnt, 1] = nd.interpolation.zoom(md, zoom=(6 / 10, 20 / 30, 20 / 30), mode="nearest")
            dat_resize_a[cnt, 2] = nd.interpolation.zoom(td, zoom=(6 / (slice_z-2), 20 / (slice_y-2), 20 / (slice_x-2)), mode="nearest")
            if cnt % 100 == 0:
                print("\rAug %d/%d" % (cnt, len(dat_resize_a)), end="")

        dat_resize_a = np.transpose(dat_resize_a, (0, 1, 3, 4, 2))
        dat_resize_a = np.concatenate((dat_resize_r, dat_resize_a))
        np.save(save_data_path + "aug_dat_resize.npy", dat_resize_a)
        print('\naugmented data resize done!')
    else:
        dat_resize_a = np.load(save_data_path + "aug_dat_resize.npy", mmap_mode="r")

    fold_idx = np.linspace(start=0, stop=PATIENT_NUM, num=6, endpoint=True, dtype=np.int64)  # stop参数需要根据所用病例数量设置

    for cur_fold_cnt in range(st.max_fold):
        leave_idx = np.arange(fold_idx[cur_fold_cnt], fold_idx[cur_fold_cnt + 1])
        train_idx = np.array([i for i in range(PATIENT_NUM) if i not in leave_idx])

        leave_idx = np.any(leave_idx[:, None] == patch_lbl[:, 1], axis=0)
        train_idx = np.any(train_idx[:, None] == augment_lbl[:, 1], axis=0)

        fold_dat = dat_resize_a[train_idx]
        fold_lbl = augment_lbl[train_idx]

        np.save(save_data_path + "trn_dat_%d.npy" % cur_fold_cnt, fold_dat)
        np.save(save_data_path + "trn_lbl_%d.npy" % cur_fold_cnt, fold_lbl)

        leave_dat = dat_resize_r[leave_idx]
        leave_lbl = patch_lbl[leave_idx]

        np.save(save_data_path + "tst_dat_%d.npy" % cur_fold_cnt, leave_dat)
        np.save(save_data_path + "tst_lbl_%d.npy" % cur_fold_cnt, leave_lbl)
    print('Train data, Test data done!')


def load_fold(fold_num=-1):
    if not os.path.exists(save_data_path + "tst_lbl_%d.npy" % st.fold_num):
        split_fold()
    if fold_num == -1:
        fold_num = st.fold_num
    trn_dat = np.load(save_data_path + "trn_dat_%d.npy" % fold_num, mmap_mode="r")
    trn_lbl = np.load(save_data_path + "trn_lbl_%d.npy" % fold_num, mmap_mode="r")
    tst_dat = np.load(save_data_path + "tst_dat_%d.npy" % fold_num, mmap_mode="r")
    tst_lbl = np.load(save_data_path + "tst_lbl_%d.npy" % fold_num, mmap_mode="r")

    return trn_dat, trn_lbl[..., 0, None], tst_dat, tst_lbl[..., 0, None]
