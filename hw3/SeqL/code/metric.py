import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

import datetime
import numpy as np
import pandas as pd
import math
import util

def loss_function(y_true, y_pred):
    judge = tf.math.equal(y_true, -1.0)
    mask = tf.math.logical_not(tf.math.logical_and(judge[:,0], judge[:,1]))
    
    loss_ = tf.keras.losses.MSE(y_true, y_pred)
    loss_ = tf.cast(loss_, dtype = 'float32')
    count = tf.math.count_nonzero(mask) 
    count = tf.cast(count, dtype='float32')
    mask = tf.cast(mask, dtype='float32')

    loss_ *= mask

    return tf.reduce_sum(loss_)/(count+1)

Max_S = 10
BATCH_S = 4

def proba(v, mode, m_hist):
    prob = 0
    if mode >=0 :
        ref = m_hist[mode]
        if v <= 2: prob = ref[0]
        elif v<=6: prob = ref[1]
        elif v<=12: prob = ref[2]
        elif v<=18: prob = ref[3]
        elif v<=24: prob = ref[4]
        elif v<=30: prob = ref[5]
        else: prob = ref[6]
    return prob



def mode_const(y_pred, time_list, mode_list, yscaler, m_hist):
    sum_p = 0
    c = 0
    for i in range(1,y_pred.shape[0]):
        mode = int(mode_list[i].numpy())
        if mode >= 0:
            loc_former = yscaler.inverse_transform(y_pred[i-1,:].reshape(1,2))
            loc_later = yscaler.inverse_transform(y_pred[i,:].reshape(1,2))
            dis = util.distance(loc_former[0], loc_later[0])
            v = dis / time_list[i]
            p = - math.log(1 + proba(v, mode, m_hist))
            sum_p += p
            c += 1
    if c > 0: return sum_p / c
    else: return 0



def loss_function_const(y_true, y_pred, time_list, mode_list, yscaler, m_hist):
    judge = tf.math.equal(y_true, -1.0)
    mask = tf.math.logical_not(tf.math.logical_and(judge[:,0], judge[:,1]))
   
    loss_ = tf.keras.losses.MSE(y_true, y_pred)
    loss_ = tf.cast(loss_, dtype = 'float32')
    count = tf.math.count_nonzero(mask) 
    count = tf.cast(count, dtype='float32')
    mask = tf.cast(mask, dtype='float32')
    loss_ *= mask
    
    constraint = mode_const(y_pred.numpy(), time_list, mode_list, yscaler, m_hist)
    thres_alpha = 0.05
   
    return tf.reduce_sum(loss_)/(count + 1) + thres_alpha * constraint

mode_class_num=3


def mode_loss(y_true, y_pred, isSeq = False):
    if mode_class_num > 2:
        cce = tf.keras.losses.CategoricalCrossentropy()
    else:
        cce = tf.keras.losses.BinaryCrossentropy()
    y_m = tf.keras.utils.to_categorical(y_true, num_classes=mode_class_num)
    if isSeq: 
        y_pred = tf.reshape(y_pred,[BATCH_S * Max_S, mode_class_num])

    return cce(y_m, y_pred)


log_var_a = tf.Variable((0.), trainable=True, name = 'var_1')
# log_var_b = tf.Variable((0.),trainable=True, name = 'var_2')
log_var_c = tf.Variable((2.),trainable=True, name = 'var_3')



def mix_loss(loc_true, loc_pred, mode_pred, time_list, mode_list, yscaler, m_hist, w = False):
    m_ls = mode_loss(mode_list, mode_pred, True)
    p_ls = loss_function_const(loc_true, loc_pred, time_list, mode_list, yscaler, m_hist)
    if w:
        loss = tf.math.exp(-log_var_a) * p_ls + tf.math.exp(-log_var_c) * m_ls + log_var_a + log_var_c
    else:
#         loss = loss_function_const(loc_true, loc_pred, time_list, mode_list)
        loss = p_ls + 0.0005* m_ls
    return loss



def compute_error(tr_loc, Y, yscaler):
    err = []
    m = 0
    i = 0
    while m <tr_loc.shape[0]:
        n=0
        while n<tr_loc.shape[1]:
            if Y[m,n,0]!=-1.0 and Y[m,n,0]!=-1.0:         
                tr_loc_real = yscaler.inverse_transform(tr_loc[m,n,:].reshape(1,2))
                y_true_real = yscaler.inverse_transform(Y[m,n,:].reshape(1,2))
                err.append(distance(tr_loc_real.reshape(2), y_true_real.reshape(2)))
                i += 1
            n += 1
        m += 1
    return err

def predict(model, fea, lab, batch_size = BATCH_S):
    pred = []
    true = []
    for i in range(0, fea.shape[0]):
        x_batch = fea[i, :, :, :, :]
        y_batch = lab[i, :, :].reshape((batch_size * Max_S, 2))
        x_batch= tf.cast(x_batch, tf.float32)
#         t_batch = tf.cast(t_batch, tf.float32)
        
        pred_batch = model(x_batch)
        pred_batch = tf.reshape(pred_batch, [batch_size * Max_S, 2]).numpy()
        if i == 0:
            pred = pred_batch
            true = y_batch
        else:
            pred = np.concatenate((pred, pred_batch), axis = 0)
            true =  np.concatenate((true, y_batch), axis = 0)
    return pred, true

def predict_t(model, fea, lab, time, lox_sel, batch_size=BATCH_S):
    pred = []
    true = []
    for i in range(0, fea.shape[0]):
        x_batch = fea[i, :, :, :, :]
        y_batch = lab[i, :, :].reshape((batch_size * Max_S, 2))
        t_batch = time[i, :]
        x_batch= tf.cast(x_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)
        batch_input = []
        batch_input.append(x_batch)
        batch_input.append(t_batch)
        
#         pred_batch = model(x_batch, t_batch)
        if lox_sel == 0:
            pred_batch = model(batch_input)
        else:
            pred_batch, _ = model(batch_input)
        pred_batch = tf.reshape(pred_batch, [batch_size * Max_S, 2]).numpy()
        if i == 0:
            pred = pred_batch
            true = y_batch
        else:
            pred = np.concatenate((pred, pred_batch), axis = 0)
            true =  np.concatenate((true, y_batch), axis = 0)
    return pred, true

def predict_mtl(model, fea, lab, time, batch_size=BATCH_S):
    pred = []
    true = []
    for i in range(0, fea.shape[0]):
        x_batch = fea[i, :, :, :, :]
        y_batch = lab[i, :, :].reshape((batch_size * Max_S, 2))
        t_batch = time[i, :]
        x_batch= tf.cast(x_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)
       
        pred_batch, _ = model(x_batch, t_batch)
        pred_batch = tf.reshape(pred_batch, [batch_size * Max_S, 2]).numpy()
        if i == 0:
            pred = pred_batch
            true = y_batch
        else:
            pred = np.concatenate((pred, pred_batch), axis = 0)
            true =  np.concatenate((true, y_batch), axis = 0)
    return pred, true


def f_error(tr_loc, Y, yscaler):
    err = []
    m=0
                   
    while m <tr_loc.shape[0]:
        if Y[m,0]!=-1.0 and Y[m,1]!=-1.0:
            tr_loc_real = yscaler.inverse_transform(tr_loc[m,:].reshape(1,2))
            y_true_real = yscaler.inverse_transform(Y[m,:].reshape(1,2))
            err.append(util.distance(tr_loc_real.reshape(2), y_true_real.reshape(2)))
        m += 1
    
    print(m)
    return err


def mode_accu_pred(model, fea, lab, time, batch_size=BATCH_S):
    pred = []
    true = []
    for i in range(0, fea.shape[0]):
        x_batch = fea[i, :, :, :, :]
        y_batch = lab[i, :]
        t_batch = time[i, :]
        x_batch= tf.cast(x_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)
        batch_input = []
        batch_input.append(x_batch)
        batch_input.append(t_batch)
        _, pred_batch = model(batch_input)
        
        pred_batch = tf.reshape(pred_batch, [batch_size * Max_S, mode_class_num]).numpy()
        if i == 0:
            pred = pred_batch
            true = y_batch
        else:
            pred = np.concatenate((pred, pred_batch), axis = 0)
            true =  np.concatenate((true, y_batch), axis = 0)
    acc_count = 0
        
    for c, c1 in zip(pred, true):
        if c[0] >= c[1] and c1==0:
            acc_count += 1
        if c[0] < c[1] and c1==1:
            acc_count +=1   
    return pred, true, float(acc_count)/pred.shape[0]

