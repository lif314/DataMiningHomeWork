# coding: utf-8


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import pandas as pd
import numpy as np
# import math
import tensorflow as tf
import random
# tf.enable_eager_execution()
# import datetime

import util, model, metric, seq

# from imp import reload


# 指定训练数据集
drname = "jd2g-2"
# drname = "sp4g"


def load_train_test_dtst(drname):
    path = "../result/"
    f = np.load(path + drname + "-tr_f.npy")
    l = np.load(path + drname + "-tr_l.npy")
    m = np.load(path + drname + "-tr_m.npy")
    t = np.load(path + drname + "-tr_t.npy")

    ef = np.load(path + drname + "-te_f.npy")
    el = np.load(path + drname + "-te_l.npy")
    em = np.load(path + drname + "-te_m.npy")
    et = np.load(path + drname + "-te_t.npy")

    return f, l, m, t, ef, el, em, et


# reproduce npy

def re_npy(drname):
    tr_d, tr_l, tr_t, tr_m, te_d, te_l, te_t, te_m = util.seq_sli(drname)
    f, l, t, m = util.data_prepare(tr_d, tr_l, tr_t, tr_m, _)
    f = np.array(f)
    l = np.array(l)
    t = np.array(t)
    m = np.array(m)
    ef, el, et, em = util.data_prepare(te_d, te_l, te_t, te_m, _)
    ef = np.array(ef)
    el = np.array(el)
    et = np.array(et)
    em = np.array(em)

    return f, l, m, t, ef, el, em, et


yscaler = util.preprocess(drname)
f, l, m, t, ef, el, em, et = load_train_test_dtst(drname)

_ = util.mode_dict(drname)
m_hist = util.speed_hist(_)

lox_sel = 0
if lox_sel == 0:
    raw_model = seq.Seq()
else:
    raw_model = model.Seq()

# 优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
# 损失函数
loss_metric = tf.keras.metrics.Mean()

m_med_err = 999
m_mea_err = 999
m_nin_err = 999
max_acc = 0

shuffle_list = random.sample(range(f.shape[0]), f.shape[0])

# 训练轮数
epoches = 2000

for epoch in range(0, epoches):
    print("=====>第{}轮训练开始：".format(epoch))

    for j in shuffle_list:
        x_batch_train = f[j, :, :, :, :]
        y_batch_train = l[j, :, :]
        t_batch_time = t[j, :]
        m_batch_time = m[j, :]
        x_batch_train = tf.cast(x_batch_train, tf.float32)
        y_batch_train = tf.cast(y_batch_train, tf.float32)
        t_batch_time = tf.cast(t_batch_time, tf.float32)
        m_batch_time = tf.cast(m_batch_time, tf.float32)
        batch_input = []
        #         print (x_batch_train.shape, t_batch_time.shape)
        batch_input.append(x_batch_train)
        batch_input.append(t_batch_time)

        with tf.GradientTape() as tape:
            tape.watch(raw_model.trainable_variables)
            if lox_sel == 0:
                pred = raw_model(batch_input)
            else:
                pred, _ = raw_model(batch_input)
            pred = tf.reshape(pred, [util.BATCH_S * util.Max_S, 2])
            if lox_sel == 0:
                loss = metric.loss_function_const(y_batch_train, pred, t_batch_time, m_batch_time, yscaler, m_hist)
            else:
                loss = metric.mix_loss(y_batch_train, pred, _, t_batch_time, m_batch_time, yscaler, m_hist)

        grads = tape.gradient(loss, raw_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, raw_model.trainable_variables))
        loss_metric(loss)

        if j % 2000 == 0:
            print('step %s: loss = %s' % (j, loss))

    if epoch % 10 == 0 and epoch <= 400:

        pred, true = metric.predict_t(raw_model, ef, el, et, lox_sel)
        te_err = sorted(metric.f_error(pred, true, yscaler))
        print(epoch, np.median(te_err), np.mean(te_err), te_err[int(len(te_err) * 0.9)])

        if lox_sel != 0:
            p1, _1, acc1 = metric.mode_accu_pred(raw_model, ef, em, et)
            print(epoch, acc1)
        # save
        m_nin_err = util.save(drname, m_nin_err, te_err, raw_model, lox_sel)

    if epoch > 400:
        pred, true = metric.predict_t(raw_model, ef, el, et, lox_sel)
        te_err = sorted(metric.f_error(pred, true, yscaler))
        print(epoch, np.median(te_err), np.mean(te_err), te_err[int(len(te_err) * 0.9)])

        if lox_sel != 0:
            p1, _1, acc1 = metric.mode_accu_pred(raw_model, ef, em, et)
            print(epoch, acc1)
        m_nin_err = util.save(drname, m_nin_err, te_err, raw_model, lox_sel)

# load_trained_model
load_model = tf.keras.models.load_model("../md/0706-prnet+_jd2g-2",
                                        custom_objects={"Seq": model.Seq, "Global": model.Global,
                                                        "Attention": model.Attention, "Local": model.Local})

pred, true = metric.predict_t(load_model, ef, el, et, lox_sel)
te_err = sorted(metric.f_error(pred, true, yscaler))
print(np.median(te_err), np.mean(te_err), te_err[int(len(te_err) * 0.9)])
