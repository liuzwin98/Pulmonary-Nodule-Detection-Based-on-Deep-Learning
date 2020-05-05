import settings as st
import utils
import numpy as np
import tensorflow as tf
import model
from time import time
import os
from glob import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

"""
    bdat: Big patch
    mdat: middle patch
    tdat: tiny patch
"""


def train_proposed():
    trn_dat, trn_lbl, tst_dat, tst_lbl = utils.load_fold()

    placeholders = {"bdat": tf.placeholder(shape=(None, 20, 20, 6), dtype=tf.float32, name="bdat_place"),
                    "mdat": tf.placeholder(shape=(None, 20, 20, 6), dtype=tf.float32, name="mdat_place"),
                    "tdat": tf.placeholder(shape=(None, 20, 20, 6), dtype=tf.float32, name="tdat_place"),
                    "lbl": tf.placeholder(shape=(None, 1), dtype=tf.int64),
                    "train": st.is_training}

    models = model.MGICNN(placeholders=placeholders)
    models.build_proposed()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    summary_writer = tf.summary.FileWriter(logdir=st.summ_path)  # 用于Tensorboard可视化
    saver = tf.train.Saver(max_to_keep=0)

    tst_ones_true = tst_lbl[..., 0] == 1
    tst_zeros_true = tst_lbl[..., 0] == 0

    global_time = time()
    local_time = time()
    trn_cnt = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('Start training...')
        for cur_epoch in range(st.epoch):
            rand_idx = np.random.permutation(len(trn_dat))
            for cur_step in range(0, len(trn_dat), st.batch_size):
                cur_idx = rand_idx[cur_step:cur_step + st.batch_size]

                feed_dict = {placeholders["bdat"]: trn_dat[cur_idx, 0],
                             placeholders["mdat"]: trn_dat[cur_idx, 1],
                             placeholders["tdat"]: trn_dat[cur_idx, 2],
                             placeholders["lbl"]: trn_lbl[cur_idx],
                             placeholders["train"]: True}

                sess.run(models.optim, feed_dict=feed_dict)
                if time() - local_time > 60:
                    feed_dict.update({placeholders["train"]: False})
                    loss, summ = sess.run([models.loss, models.summary_op], feed_dict=feed_dict)
                    summary_writer.add_summary(summ, global_step=trn_cnt)
                    print("\rEpoch %d, Step %d/%d, Loss %f" % (cur_epoch, cur_step, len(trn_dat), loss))
                    local_time = time()
                trn_cnt += 1
            if cur_epoch % 5 == 0 or cur_epoch == (st.epoch - 1):  # 每5轮及最后一轮验证、保存
                saver.save(sess=sess, save_path=st.summ_path + "%d_%d_%d_%d.ckpt" % (st.fold_num, st.multistream_mode, st.model_mode, cur_epoch))
                print("\nSaved in %s" % st.summ_path + "%d.ckpt" % cur_epoch)
                tst_pred = np.zeros(shape=len(tst_dat), dtype=np.uint8)
                for tst_step in range(0, len(tst_dat), st.batch_size):
                    test_feed_dict = {placeholders["bdat"]: tst_dat[tst_step:tst_step + st.batch_size, 0],
                                      placeholders["mdat"]: tst_dat[tst_step:tst_step + st.batch_size, 1],
                                      placeholders["tdat"]: tst_dat[tst_step:tst_step + st.batch_size, 2],
                                      placeholders["lbl"]: tst_lbl[tst_step:tst_step + st.batch_size],
                                      placeholders["train"]: False}
                    tst_pred[tst_step:tst_step + st.batch_size] = sess.run(models.prediction, feed_dict=test_feed_dict)
                tst_correct = np.equal(tst_lbl[..., 0], tst_pred)
                tst_wrong = np.not_equal(tst_lbl[..., 0], tst_pred)
                tst_TP = np.count_nonzero(np.logical_and(tst_correct, tst_ones_true))
                tst_TN = np.count_nonzero(np.logical_and(tst_correct, tst_zeros_true))
                tst_FP = np.count_nonzero(np.logical_and(tst_wrong, tst_ones_true))
                tst_FN = np.count_nonzero(np.logical_and(tst_wrong, tst_zeros_true))
                print("\n(In training) validation result: ", "TP/FP/TN/FN (%d/%d/%d/%d)" % (tst_TP, tst_FP, tst_TN, tst_FN))


def test_proposed():
    # tst_dat:测试数据、tst_lbl:标签，5折交叉验证
    _, _, tst_dat1, tst_lbl1 = utils.load_fold(fold_num=0)
    _, _, tst_dat2, tst_lbl2 = utils.load_fold(fold_num=1)
    _, _, tst_dat3, tst_lbl3 = utils.load_fold(fold_num=2)
    _, _, tst_dat4, tst_lbl4 = utils.load_fold(fold_num=3)
    _, _, tst_dat5, tst_lbl5 = utils.load_fold(fold_num=4)

    all_tst_pm = []
    all_tst_dat = [tst_dat1, tst_dat2, tst_dat3, tst_dat4, tst_dat5]
    all_tst_lbl = [tst_lbl1, tst_lbl2, tst_lbl3, tst_lbl4, tst_lbl5]
    print("data loading over!")
    placeholders = {"bdat": tf.compat.v1.placeholder(shape=(None, 20, 20, 6), dtype=tf.float32, name="bdat_place"),
                    "mdat": tf.compat.v1.placeholder(shape=(None, 20, 20, 6), dtype=tf.float32, name="mdat_place"),
                    "tdat": tf.compat.v1.placeholder(shape=(None, 20, 20, 6), dtype=tf.float32, name="tdat_place"),
                    "lbl": tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.int64),
                    "train": st.is_training}

    models = model.MGICNN(placeholders=placeholders)
    models.build_proposed()
    print("model loading over!")
    with tf.compat.v1.Session() as sess:
        print("In sess!")
        saver = tf.compat.v1.train.Saver()
        for cur_fold in range(st.max_fold):
            tst_dat = all_tst_dat[cur_fold]
            tst_lbl = all_tst_lbl[cur_fold]
            tst_cnt = 0
            tst_pm = np.zeros(shape=len(tst_dat), dtype=np.float32)
            sess.run(tf.compat.v1.global_variables_initializer())
            # saver.restore(sess=sess, save_path=os.path.join(st.tst_model_path, "%d_%d_%d_%d.ckpt" % (
            # st.fold_num, st.multistream_mode, st.model_mode, st.tst_epoch)))
            ckpt = tf.train.latest_checkpoint(st.tst_model_path)
            saver.restore(sess, ckpt)
            print("len of tst_data is:", len(tst_dat))
            for tst_step in range(0, len(tst_dat), st.batch_size):
                test_feed_dict = {placeholders["bdat"]: tst_dat[tst_step:tst_step + st.batch_size, 0],
                                  placeholders["mdat"]: tst_dat[tst_step:tst_step + st.batch_size, 1],
                                  placeholders["tdat"]: tst_dat[tst_step:tst_step + st.batch_size, 2],
                                  placeholders["lbl"]: tst_lbl[tst_step:tst_step + st.batch_size],
                                  placeholders["train"]: False}

                tst_pm[tst_step:tst_step + st.batch_size] = sess.run(models.pred_sig, feed_dict=test_feed_dict)[:, 0]

            all_tst_pm += [tst_pm]
            tst_cnt += len(tst_pm)
    all_tst_pm = np.concatenate(all_tst_pm, axis=0)  # 跨行拼接
    print("test result's shape:", all_tst_pm.shape)
    np.save(st.summ_path_root + "pred_result/%d_%d_pm.npy" % (st.model_mode, st.multistream_mode), all_tst_pm)
    print("result saving done!")


if __name__ == "__main__":
    # #### Uncomment if you want to use GPUtil and/or GPU flag #####
    # import GPUtil
    # if st.set_gpu==-1:
    #     devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
    # else:
    #     devices = "%d"%st.set_gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = devices
    # print("Using only device %s" % devices)
    # #########################################################

    if st.train:
        train_proposed()
    else:
        test_proposed()
