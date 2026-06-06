import os
from time import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer

from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)

def baro_old(
    data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, with_baro=False, **kwargs
):
    print(f"with_baro={with_baro}")
    
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        anomal_df = data.tail(len(data) - anomalies[0])


    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }
 
import time

def baro(
    data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, with_baro=False, seq_len=1, scalar_type="Robust", **kwargs
):
    seq_len = 12
    print(f"with_baro={with_baro}")
    
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )
    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect columns
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    train_start = time.time()  # timing training/fitting
    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        if seq_len > 1:
            a_seq = a[: (len(a) // seq_len) * seq_len].reshape(-1, seq_len)
            b_seq = b[: (len(b) // seq_len) * seq_len].reshape(-1, seq_len)
            a = a_seq.mean(axis=1)
            b = b_seq.mean(axis=1)

        # fit scaler on normal data
        if scalar_type == "Robust":
            scaler = RobustScaler().fit(a.reshape(-1,1))
        elif scalar_type == "Standard":
            scaler = StandardScaler().fit(a.reshape(-1,1))
        elif scalar_type == "Quantile":
            scaler = QuantileTransformer(output_distribution="normal").fit(a.reshape(-1,1))
        train_end = time.time()  # fit done

        # compute inference on anomalous data
        infer_start = time.time()
        if scalar_type in ["Robust","Standard","Quantile"]:
            zscores = scaler.transform(b.reshape(-1,1))[:,0]
        elif scalar_type == "MAD":
            median = np.median(a)
            mad = np.median(np.abs(a - median)) + 1e-8
            zscores = np.abs(b - median) / mad
        elif scalar_type == "IQR":
            q1, q3 = np.percentile(a, [25,75])
            iqr = q3 - q1 + 1e-8
            zscores = np.clip((b - q3)/iqr, 0, None)
        elif scalar_type == "EMA":
            span = kwargs.get("ema_span", 10)
            ema = pd.Series(a).ewm(span=span).mean().to_numpy()
            zscores = np.abs(b - ema[-len(b):]) / (np.std(a) + 1e-8)
        elif scalar_type == "ModifiedZ":
            median = np.median(a)
            mad = np.median(np.abs(a - median)) + 1e-8
            zscores = 0.6745 * (b - median) / mad
        elif scalar_type == "Rank":
            ranks_a = np.argsort(np.argsort(a)) / (len(a)-1)
            ranks_b = np.argsort(np.argsort(b)) / (len(b)-1)
            zscores = np.abs(ranks_b - ranks_a.mean())
        else:
            raise ValueError(f"Unknown scalar_type: {scalar_type}")
        infer_end = time.time()

        score = max(zscores)
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
        "train_time": train_end - train_start,
        "infer_time": infer_end - infer_start
    }
 

def mmnsigma(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    scaler_function = kwargs.get("scaler_function", StandardScaler) 

    metric = data["metric"]
    logs = data["logs"]
    logts = data["logts"]
    traces = data["traces"]
    traces_err = data["tracets_err"]
    traces_lat = data["tracets_lat"]
    cluster_info = data["cluster_info"]
    
    # ==== PREPARE DATA ====
    # the metric is currently sampled for 1 seconds, resample for 15s by just take 1 point every 15 points
    metric = metric.iloc[::15, :]

    # == metric ==
    normal_metric = metric[metric["time"] < inject_time]
    anomal_metric = metric[metric["time"] >= inject_time]
    normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
    anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
    intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]
    normal_metric = normal_metric[intersect]
    anomal_metric = anomal_metric[intersect]

    # == logts ==
    logts = drop_constant(logts)
    normal_logts = logts[logts["time"] < inject_time].drop(columns=["time"])
    anomal_logts = logts[logts["time"] >= inject_time].drop(columns=["time"])

    # == traces_err ==
    if dataset == "mm-tt" or dataset == "mm-ob":
        traces_err = traces_err.fillna(method='ffill')
        traces_err = traces_err.fillna(0)
        traces_err = drop_constant(traces_err)

        normal_traces_err = traces_err[traces_err["time"] < inject_time].drop(columns=["time"])
        anomal_traces_err = traces_err[traces_err["time"] >= inject_time].drop(columns=["time"])
    
     # == traces_lat ==
    if dataset == "mm-tt" or dataset == "mm-ob":
        traces_lat = traces_lat.fillna(method='ffill')
        traces_lat = traces_lat.fillna(0)
        traces_lat = drop_constant(traces_lat)
        normal_traces_lat = traces_lat[traces_lat["time"] < inject_time].drop(columns=["time"])
        anomal_traces_lat = traces_lat[traces_lat["time"] >= inject_time].drop(columns=["time"])
    
    # ==== PROCESS ====
    ranks = []
    
    # == metric ==
    for col in normal_metric.columns:
        if col == "time":
            continue
        a = normal_metric[col].to_numpy()
        b = anomal_metric[col].to_numpy()

        scaler = scaler_function().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    # == logs ==
    for col in normal_logts.columns:
        a = normal_logts[col].to_numpy()
        b = anomal_logts[col].to_numpy()

        scaler = scaler_function().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    # == traces_err ==
    if dataset == "mm-tt" or dataset == "mm-ob":
        for col in normal_traces_err.columns:
            a = normal_traces_err[col].to_numpy()[:-2]
            b = anomal_traces_err[col].to_numpy()
                
            scaler = scaler_function().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            score = max(zscores)
            ranks.append((col, score))
   
    # == traces_lat ==
    if dataset == "mm-tt" or dataset == "mm-ob":
        for col in normal_traces_lat.columns:
            a = normal_traces_lat[col].to_numpy()
            b = anomal_traces_lat[col].to_numpy()

            scaler = scaler_function().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            score = max(zscores)
            ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    if kwargs.get("verbose") is True:
        for r, score in ranks[:20]:
            print(f"{r}: {score:.2f}")

    ranks = [x[0] for x in ranks]

    return {
        "ranks": ranks,
    }
    

def mmbaro(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    return mmnsigma(
        data=data,
        inject_time=inject_time,
        dataset=dataset,
        sli=sli,
        scaler_function=RobustScaler, **kwargs
    )