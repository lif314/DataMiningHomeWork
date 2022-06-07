import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import datetime
import numpy as np
import pandas as pd

import math


def rad(d):
    return d * math.pi / 180.0


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


feature_col = ['RNCID_1', 'RNCID_2', 'RNCID_3', 'RNCID_4', 'RNCID_5', 'RNCID_6', 'RNCID_7',
               'CellID_1', 'CellID_2', 'CellID_3', 'CellID_4', 'CellID_5', 'CellID_6', 'CellID_7',
               'Dbm_1', 'Dbm_2', 'Dbm_3', 'Dbm_4', 'Dbm_5', 'Dbm_6', 'Dbm_7',
               'AsuLevel_1', 'AsuLevel_2', 'AsuLevel_3', 'AsuLevel_4', 'AsuLevel_5', 'AsuLevel_6', 'AsuLevel_7',
               'SignalLevel_1', 'SignalLevel_2', 'SignalLevel_3', 'SignalLevel_4', 'SignalLevel_5', 'SignalLevel_6',
               'SignalLevel_7',
               # 'Basic_psc_pci_1', 'Basic_psc_pci_2', 'Basic_psc_pci_3','Basic_psc_pci_4','Basic_psc_pci_5','Basic_psc_pci_6','Basic_psc_pci_7',
               # 'Arfcn_1','Arfcn_2','Arfcn_3','Arfcn_4','Arfcn_5','Arfcn_6','Arfcn_7',
               'Lon', 'Lon2', 'Lon3', 'Lon4', 'Lon5', 'Lon6', 'Lon7',
               'Lat', 'Lat2', 'Lat3', 'Lat4', 'Lat5', 'Lat6', 'Lat7',
               'Time_1', 'Time_2', 'Time_3', 'Time_4', 'Time_5', 'Time_6', 'Time_7']
feature_col4 = ['RNCID_1', 'RNCID_2', 'RNCID_3', 'RNCID_4', 'RNCID_5', 'RNCID_6', 'RNCID_7',
                'CellID_1', 'CellID_2', 'CellID_3', 'CellID_4', 'CellID_5', 'CellID_6', 'CellID_7',
                'Dbm_1', 'Dbm_2', 'Dbm_3', 'Dbm_4', 'Dbm_5', 'Dbm_6', 'Dbm_7',
                'AsuLevel_1', 'AsuLevel_2', 'AsuLevel_3', 'AsuLevel_4', 'AsuLevel_5', 'AsuLevel_6', 'AsuLevel_7',
                'SignalLevel_1', 'SignalLevel_2', 'SignalLevel_3', 'SignalLevel_4', 'SignalLevel_5', 'SignalLevel_6',
                'SignalLevel_7',
                'Basic_psc_pci_1', 'Basic_psc_pci_2', 'Basic_psc_pci_3', 'Basic_psc_pci_4', 'Basic_psc_pci_5',
                'Basic_psc_pci_6', 'Basic_psc_pci_7',
                # 'Arfcn_1','Arfcn_2','Arfcn_3','Arfcn_4','Arfcn_5','Arfcn_6','Arfcn_7',
                'Lon', 'Lon2', 'Lon3', 'Lon4', 'Lon5', 'Lon6', 'Lon7',
                'Lat', 'Lat2', 'Lat3', 'Lat4', 'Lat5', 'Lat6', 'Lat7',
                'Time_1', 'Time_2', 'Time_3', 'Time_4', 'Time_5', 'Time_6', 'Time_7']

data_dict = {"jd2g-1": "../data/jiading/2g/jiading-2g-1.csv",
             "jd2g-2": "../data/jiading/2g/jiading-2g-2.csv",
             "jd4g": "../data/jiading/4g/jiading-4g.csv",
             "sp2g": "../data/siping/2g/siping-2g.csv",
             "sp4g": "../data/siping/4g/siping_4g.csv"}


def preprocess(drname):
    path = data_dict[drname]
    data = pd.read_csv(path)
    #     for i in range(1,8):
    #         data['Time_%d' % i] = data['MRTime']
    #     if drname =="sp4":
    #         tot_mat = data[feature_col4].values
    #     else:
    #         tot_mat = data[feature_col].values
    #     # print (tot_mat.shape)
    #     xscaler = MinMaxScaler()
    #     xscaler.fit(tot_mat)
    yscaler = MinMaxScaler()
    yscaler.fit(data[['Longitude', 'Latitude']].values)

    return yscaler


Max_S = 10
BATCH_S = 4


def mode_dict(drname):
    path = data_dict[drname]
    data = pd.read_csv(path)
    trajs = data.groupby(["TrajID"])
    m_d = {}
    m_d[0] = []
    m_d[1] = []
    m_d[2] = []
    for trajid, traj in trajs:
        md = list(traj['mode'])
        loc = traj[['Longitude', 'Latitude']].values
        tl = list(traj['MRTime'])
        for j in range(1, traj.shape[0]):
            if j > 0:
                delta_t = compute_time_interval(tl[j - 1], tl[j])
                delta_s = distance(loc[j - 1, :], loc[j, :])
                m = int(md[j])
                if delta_t > 0:
                    m_d[m].append(delta_s / delta_t)

    return m_d


# _ = mode_dict(data)

def speed_hist(mode_dict):
    mode_hist_dict = {}
    for key, value in mode_dict.items():
        tmp = np.zeros((7))
        for t in value:
            if t <= 2:
                tmp[0] += 1
            elif t <= 6:
                tmp[1] += 1
            elif t <= 12:
                tmp[2] += 1
            elif t <= 18:
                tmp[3] += 1
            elif t <= 24:
                tmp[4] += 1
            elif t <= 30:
                tmp[5] += 1
            else:
                tmp[6] += 1
        prob = []
        if len(value) > 0:
            for t in tmp:
                prob.append(float(t) / float(len(value)))
        mode_hist_dict[key] = prob

    return mode_hist_dict


bs_c = 7
col = 8


def make_sequence1(data, xscaler, yscaler, max_subs=Max_S):
    seq_list = []
    time_list = []
    loc_list = []
    mode_list = []
    trajs = data.groupby(["TrajID"])
    max_seq = 0
    max_subseq = 0
    for trajid, traj in trajs:
        dt = xscaler.transform(traj[feature_col].values)

        lc = yscaler.transform(traj[['Longitude', 'Latitude']].values)
        tl = traj['MRTime']
        md = traj['mode']
        # traj = traj.sort_values(["MRTime"])
        _ = traj.shape[0] % max_subs
        c = int(traj.shape[0] / max_subs)

        if traj.shape[0] <= max_subs:
            seq = np.zeros((max_subs, bs_c * col))
            loc = np.ones((max_subs, 2)) * (-1)
            t_tl, m_l, ti = np.zeros((max_subs)), np.zeros((max_subs)), np.zeros((max_subs))
            seq[0: traj.shape[0], :] = dt[0: traj.shape[0], :]
            loc[0: traj.shape[0], :] = lc[0: traj.shape[0], :]
            t_tl[0: traj.shape[0]] = np.array(tl[0: traj.shape[0]])
            m_l[0: traj.shape[0]] = md[0: traj.shape[0]]
            for j in range(traj.shape[0]):
                if j == 0:
                    ti[j] = 0
                else:
                    ti[j] = compute_time_interval(t_tl[j - 1], t_tl[j])
            seq_list.append(seq)
            loc_list.append(loc)
            time_list.append(ti)
            mode_list.append(m_l)
        else:
            start = 0
            while start <= traj.shape[0] - max_subs - 1:
                seq = np.zeros((max_subs, bs_c * col))
                loc = np.ones((max_subs, 2)) * (-1)
                t_tl, m_l, ti = np.zeros((max_subs)), np.zeros((max_subs)), np.zeros((max_subs))
                #                 print (start, start+max_sub, seq[start: start + max_subs, :].shape, dt[start: start + max_subs, :].shape)
                seq = dt[start: start + max_subs, :]
                loc = lc[start: start + max_subs, :]
                t_tl = np.array(tl[start: start + max_subs])
                m_l = md[start: start + max_subs]
                #                 t_tl = np.array()
                for j in range(max_subs):
                    if j == 0:
                        ti[j] = 0
                    else:
                        ti[j] = compute_time_interval(t_tl[j - 1], t_tl[j])
                seq_list.append(seq)
                loc_list.append(loc)
                time_list.append(ti)
                mode_list.append(m_l)
                start += max_subs

            if start != traj.shape[0] - 1:
                seq = np.zeros((max_subs, bs_c * col))
                loc = np.ones((max_subs, 2)) * (-1)
                t_tl, m_l, ti = np.zeros((max_subs)), np.zeros((max_subs)), np.zeros((max_subs))
                seq[0: traj.shape[0] - start, :] = dt[start: traj.shape[0], :]
                loc[0: traj.shape[0] - start, :] = lc[start: traj.shape[0], :]
                t_tl[0: traj.shape[0] - start] = np.array(tl[start: traj.shape[0]])
                m_l[0: traj.shape[0] - start] = md[start: traj.shape[0]]
                for j in range(max_subs):
                    if j == 0:
                        ti[j] = 0
                    else:
                        ti[j] = compute_time_interval(t_tl[j - 1], t_tl[j])
                seq_list.append(seq)
                loc_list.append(loc)
                time_list.append(ti)
                mode_list.append(m_l)

    return seq_list, loc_list, time_list, mode_list


def seq_sli(drname):
    path = data_dict[drname]
    data = pd.read_csv(path)
    a, b, c, d = [], [], [], []
    ea, eb, ec, ed = [], [], [], []
    te = pd.DataFrame()
    for i in range(5):
        tmp = data.loc[range(i, len(data), 5), :]
        if i < 4:
            ta, tb, tc, td = make_sequence1(tmp)
        else:
            ta, tb, tc, td = make_sequence1(tmp)
        if i == 0:
            a, b, c, d = ta, tb, tc, td
        elif i < 4:
            a.extend(ta)
            b.extend(tb)
            c.extend(tc)
            d.extend(td)
        else:
            ea, eb, ec, ed = ta, tb, tc, td

    train_dataset = np.array(a)
    train_label = np.array(b)
    train_time = np.array(c)
    train_mode = np.array(d)

    test_dataset = np.array(ea)
    test_label = np.array(eb)
    test_time = np.array(ec)
    test_mode = np.array(ed)

    return train_dataset, train_label, train_time, train_mode, test_dataset, test_label, test_time, test_mode


def data_prepare(fea, lab, tl, md, batch_size=BATCH_S):
    f, l, t, d = [], [], [], []
    _ = fea.shape[0] % batch_size
    batch_cnt = batch_cnt = int(fea.shape[0] / batch_size)
    if _ != 0:
        batch_cnt += 1
    for i in range(0, batch_cnt):
        tmp_f = np.zeros((batch_size * Max_S, bs_c, col, 1))
        tmp_l = np.ones((batch_size * Max_S, 2)) * (-1)
        tmp_t = np.ones((batch_size * Max_S)) * (-1)
        tmp_m = np.ones((batch_size * Max_S)) * (-1)
        if i < batch_cnt - 1:
            for j in range(0, batch_size):
                f_t = fea[i * batch_size + j, :, :].reshape([Max_S, bs_c, col, 1])
                l_t = lab[i * batch_size + j, :]
                t_t = tl[i * batch_size + j]
                m_t = md[i * batch_size + j]
                for m in range(0, Max_S):
                    #                     print (j, m, j*Max_S+m)
                    tmp_f[j * Max_S + m, :, :, :] = f_t[m, :, :, :]
                    tmp_l[j * Max_S + m, :] = l_t[m, :]
                    tmp_t[j * Max_S + m] = t_t[m]
                    tmp_m[j * Max_S + m] = m_t[m]
        else:
            if _ == 0:
                for j in range(0, batch_size):
                    f_t = fea[i * batch_size + j, :, :].reshape([Max_S, bs_c, col, 1])
                    l_t = lab[i * batch_size + j, :]
                    t_t = tl[i * batch_size + j]
                    m_t = md[i * batch_size + j]
                    for m in range(0, Max_S):
                        tmp_f[j * Max_S + m, :, :, :] = f_t[m, :, :, :]
                        tmp_l[j * Max_S + m, :] = l_t[m, :]
                        tmp_t[j * Max_S + m] = t_t[m]
                        tmp_m[j * Max_S + m] = m_t[m]
            else:
                for j in range(0, _):
                    f_t = fea[i * batch_size + j, :, :].reshape([Max_S, bs_c, col, 1])
                    l_t = lab[i * batch_size + j, :]
                    t_t = tl[i * batch_size + j]
                    m_t = md[i * batch_size + j]
                    for m in range(0, Max_S):
                        tmp_f[j * Max_S + m, :, :, :] = f_t[m, :, :, :]
                        tmp_l[j * Max_S + m, :] = l_t[m, :]
                        tmp_t[j * Max_S + m] = t_t[m]
                        tmp_m[j * Max_S + m] = m_t[m]

                for j in range(_, batch_size):
                    f_t = fea[(i - 1) * batch_size + (_ - 1), :, :].reshape([Max_S, bs_c, col, 1])
                    l_t = lab[(i - 1) * batch_size + (_ - 1), :]
                    t_t = tl[(i - 1) * batch_size + (_ - 1)]
                    m_t = md[(i - 1) * batch_size + (_ - 1)]
                    for m in range(0, Max_S):
                        tmp_f[j * Max_S + m, :, :, :] = f_t[m, :, :, :]
                        tmp_l[j * Max_S + m, :] = l_t[m, :]
                        tmp_t[j * Max_S + m] = t_t[m]
                        tmp_m[j * Max_S + m] = m_t[m]

        f.append(tmp_f)
        l.append(tmp_l)
        t.append(tmp_t)
        d.append(tmp_m)

    return f, l, t, d


def save(drname, m_nin, te_r, model, lox_sel):
    if m_nin >= te_r[int(len(te_r) * 0.9)]:
        m_nin = te_r[int(len(te_r) * 0.9)]
        if lox_sel == 0:
            model.save("../md/xxxxx" + drname)  # xxxx self-defined
    return m_nin
