# coding: utf-8


import datetime
import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier

import math


def rad(d):
    return d * math.pi / 180.0

# 计算方位角
def azimuth(pt_a, pt_b):
    lon_a, lat_a = pt_a
    lon_b, lat_b = pt_b
    rlon_a, rlat_a = rad(lon_a), rad(lat_a)
    rlon_b, rlat_b = rad(lon_b), rad(lat_b)
    ec = rj + (rc - rj) * (90. - lat_a) / 90.
    ed = ec * math.cos(rlat_a)

    dx = (rlon_b - rlon_a) * ec
    dy = (rlat_b - rlat_a) * ed
    if dy == 0:
        angle = 90.
    else:
        angle = math.atan(abs(dx / dy)) * 180.0 / math.pi
    dlon = lon_b - lon_a
    dlat = lat_b - lat_a
    if dlon > 0 and dlat <= 0:
        angle = (90. - angle) + 90
    elif dlon <= 0 and dlat < 0:
        angle = angle + 180
    elif dlon < 0 and dlat >= 0:
        angle = (90. - angle) + 270
    return angle


def distance(true_pt, pred_pt):
    lat1 = float(true_pt[1])
    lng1 = float(true_pt[0])
    lat2 = float(pred_pt[1])
    lng2 = float(pred_pt[0])
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) +
                                math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * 6378.137
    s = round(s * 10000) / 10
    return s


def compute_time_interval(timestamp1, timestamp2):
    dateArray1 = datetime.datetime.utcfromtimestamp(timestamp1 / 1000)
    dateArray2 = datetime.datetime.utcfromtimestamp(timestamp2 / 1000)
    # otherStyleTime = dateArray1.strftime("%Y-%m-%d %H:%M:%S")
    return float((dateArray2 - dateArray1).seconds)


# # Monitor


def Monitor_Perpare(data, window, N, isXH=False):
    #      print (data, window, N, isXH)
    trajs = data.groupby(['TrajID'])
    seq = []
    dbm_col = []

    for i in range(N):
        if isXH:
            dbm_col += ['RSCP_%d' % (i + 1)]
        else:
            dbm_col += ['Dbm_%d' % (i + 1)]

    #     print ("ok")
    for n, traj in trajs:
        #          print (n)
        traj = traj[['mode'] + dbm_col]
        #         print (window)
        if traj.shape[0] <= window:
            seq.append(traj)
        else:
            start = 0
            while start + window < traj.shape[0]:
                seq.append(traj.iloc[start: start + window, :])
                start += window
            #                 print (start, traj.shape[0])
            seq.append(traj.iloc[start: traj.shape[0], :])
    #             print ("done")

    return seq


def stat_seq(seq, N):
    S = np.zeros((N))
    V = np.zeros((N))
    seq = seq.values

    for i in range(N):
        s = np.mean(seq[:, i + 1])
        _ = 0
        for rss in seq[:, i + 1]:
            _ += (rss - s) ** 2
        S[i] = s
        V[i] = _ / (seq.shape[0] - 1)

    return S, V


def vote(V, thre_b, thre_v, N):
    mode = [0, 0, 0]
    for i in range(N):
        if V[i] <= thre_b:
            mode[i] = 0
        elif V[i] > thre_b and V[i] <= thre_v:
            mode[i] = 1
        else:
            mode[i] = 2
    mode = np.array(mode)
    #     print (mode)
    unique, counts = np.unique(mode, return_counts=True)
    _ = dict(zip(unique, counts))
    flg = False
    result = None
    for key, value in _.items():
        if value >= 2:
            flg = True
            result = key
            break
    if not flg:
        result = 1
    return result


def Monitor(data, window=10, thre_b=3.0, thre_v=9.5, N=3, isXH=False):
    seqs = Monitor_Perpare(data, window, N, isXH)
    total = len(seqs)
    correct = 0
    pred_list = []
    mode_list = []
    for seq in seqs:
        #         print (len(seq))
        S, V = stat_seq(seq, N)
        ground_truth = int(seq.iloc[0, 0])
        pred = vote(V, thre_b, thre_v, N)
        pred_list.append(pred)
        mode_list.append(ground_truth)
        if pred == ground_truth: correct += 1

    return seqs, mode_list, pred_list, correct / total


# In[8]:


def evalute_monitor(path, isXH=False):
    data = pd.read_csv(path)
    data = data.fillna(-140)
    seqs, modes, preds, rate = Monitor(data, isXH=isXH)
    mode_w = [0, 0]
    mode_b = [0, 0]
    mode_c = [0, 0]

    for m, p in zip(modes, preds):
        if m == 0:
            mode_w[0] += 1
            if m == p:
                mode_w[1] += 1
        elif m == 1:
            mode_b[0] += 1
            if m == p:
                mode_b[1] += 1
        else:
            mode_c[0] += 1
            if m == p:
                mode_c[1] += 1

    if isXH:
        print(rate, mode_w[1] / mode_w[0], mode_b[1] / mode_b[0])
    else:
        if mode_c[0] == 0:
            print(rate, mode_w[1] / mode_w[0], mode_b[1] / mode_b[0])
        else:
            print(rate, mode_w[1] / mode_w[0], mode_b[1] / mode_b[0], mode_c[1] / mode_c[0])


# # MonoSense

def Mono_perpare(data, window=10, isXH=False):
    trajs = data.groupby(['TrajID'])
    seq = []
    if isXH:
        dbm_col = ['mode', 'RSCP_1', 'RNCID_1', 'CellID_1', 'MRTime']
    else:
        dbm_col = ['mode', 'Dbm_1', 'RNCID_1', 'CellID_1', 'MRTime']

    for n, traj in trajs:
        traj = traj[dbm_col]
        if traj.shape[0] <= window:
            seq.append(traj)
        else:
            start = 0
            while start + window < traj.shape[0]:
                seq.append(traj.iloc[start: start + window, :])
                start += window
            seq.append(traj.iloc[start: traj.shape[0], :])

    return seq


def stat_feauture(seq):
    S = seq[['RNCID_1', 'CellID_1']].drop_duplicates().shape[0]
    V = np.var(seq.values[:, 1])

    start = seq[['RNCID_1', 'CellID_1']].values[0, :]
    f = 0
    t = []
    seq = seq.values
    diff = 0
    for i in range(1, seq.shape[0]):
        diff += abs(seq[i, 1] - seq[i - 1, 1])
        if seq[i, 2] != start[0] or seq[i, 3] != start[1]:
            t.append(compute_time_interval(seq[f, 4], seq[i - 1, 4]))
            f = i
            start = [seq[i, 2], seq[i, 3]]
    if f == 0:
        t.append(compute_time_interval(seq[0, 4], seq[-1, 4]))
    T = np.mean(t)

    if seq.shape[0] > 1:
        D = diff / (seq.shape[0] - 1)
    else:
        D = 0
    alpha = -10.0
    return [S, V, T, D, math.log(S + 1e-4), math.log(V + 1e-4), math.log(T + 1e-4), math.log(D + 1e-4),
            alpha * S, alpha * V, alpha * T, alpha * D]


# In[10]:


def MonoSense(path):
    data = pd.read_csv(path)
    seqs = Mono_perpare(data)
    seq_feas = []
    labels = []
    for seq in seqs:
        seq_feas.append(stat_feauture(seq))
        labels.append(list(seq['mode'])[0])
    seq_feas = np.array(seq_feas)
    labels = np.array(labels)

    train_idx = random.sample(range(0, seq_feas.shape[0]), int(seq_feas.shape[0] * 0.8))
    all_idx = [i for i in range(0, seq_feas.shape[0])]
    test_idx = list(set(all_idx) - set(train_idx))
    train_f = seq_feas[train_idx, :]
    train_l = labels[train_idx]

    dt = DecisionTreeClassifier().fit(train_f, train_l)
    pred = dt.predict(seq_feas[test_idx])
    test_l = labels[test_idx]

    mode_w = [0, 0]
    mode_b = [0, 0]
    mode_c = [0, 0]

    correct = 0
    for m, p in zip(test_l, pred):
        if m == p:
            correct += 1
        if m == 0:
            mode_w[0] += 1
            if m == p:
                mode_w[1] += 1
        elif m == 1:
            mode_b[0] += 1
            if m == p:
                mode_b[1] += 1
        else:
            mode_c[0] += 1
            if m == p:
                mode_c[1] += 1

    print(correct / pred.shape[0], mode_w[1] / mode_w[0], mode_b[1] / mode_b[0], mode_c[1] / mode_c[0])


# # Evaluate

evalute_monitor("../data/jiading/2g/jiading-2g-1.csv")
MonoSense("../data/jiading/2g/jiading-2g-1.csv")
