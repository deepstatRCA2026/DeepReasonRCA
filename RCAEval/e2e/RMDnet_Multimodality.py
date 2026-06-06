import time
import numpy as np
import pandas as pd
import torch
from RCAEval.io.time_series import convert_mem_mb, drop_constant, drop_time, preprocess
from RCAEval.e2e.RMDnet import compute_pot_scores
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
import pandas as pd
import numpy as np

# multimodal
from RCAEval.e2e.multimodel.Eadro import MainModel as Eadro
from RCAEval.e2e.multimodel.SFlexRCAmulti import MainModel as SFlexRCAmulti
from RCAEval.e2e.multimodel.Art import ARTWrapper as Art_Model
from RCAEval.e2e.multimodel.Anofusion import AnoFusionWrapper as AnoFusion
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore")

from RCAEval.io.time_series import convert_mem_mb, drop_constant, drop_time, preprocess

from RCAEval.e2e.rcd import (
    _match_columns,
    add_fnode_and_concat,
    run_multi_phase,
)
torch.autograd.set_detect_anomaly(True)

#-------------------
# Multimodal preprocessing (FIXED)
#-------------------
def preprocess_multimodal(data, inject_time, dataset, kwargs):

    def clean(df):
        if df is None or len(df) == 0:
            return None
        df = df.ffill().fillna(0)
        df = drop_constant(df)
        return df.reset_index(drop=True)

    def scale(df):
        if df is None:
            return None
        scaler = StandardScaler()
        return pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns
        )

    def align(metric, logts, traces_err, traces_lat):
        """
        Align modalities safely across time dimension.
        Works even when traces are missing.
        """

        # collect only existing dfs
        dfs = []
        keys = []

        if metric is not None:
            dfs.append(metric)
            keys.append("metric")

        if logts is not None:
            dfs.append(logts)
            keys.append("logts")

        if traces_err is not None:
            dfs.append(traces_err)
            keys.append("traces_err")

        if traces_lat is not None:
            dfs.append(traces_lat)
            keys.append("traces_lat")

        # nothing to align
        if len(dfs) == 0:
            return metric, logts, traces_err, traces_lat

        # compute shared length
        min_len = min(len(df) for df in dfs)

        def trim(df):
            return df.iloc[:min_len].reset_index(drop=True)

        out = {}

        if metric is not None:
            out["metric"] = trim(metric)
        else:
            out["metric"] = None

        if logts is not None:
            out["logts"] = trim(logts)
        else:
            out["logts"] = None

        if traces_err is not None:
            out["traces_err"] = trim(traces_err)
        else:
            out["traces_err"] = None

        if traces_lat is not None:
            out["traces_lat"] = trim(traces_lat)
        else:
            out["traces_lat"] = None

        return out["metric"], out["logts"], out["traces_err"], out["traces_lat"]

    # ---------------- metric ----------------
    metric = data["metric"].iloc[::15, :]
    metric = preprocess(metric, dataset, kwargs.get("dk_select_useful", False))
    metric = clean(metric)

    # ---------------- log ----------------
    logts = clean(data["logts"])

    # ---------------- traces ----------------
    traces_err = clean(data.get("tracets_err", None))
    traces_lat = clean(data.get("tracets_lat", None))

    # ---------------- ALIGNMENT ----------------
    metric, logts, traces_err, traces_lat = align(
        metric, logts, traces_err, traces_lat
    )

    # ---------------- scaling ----------------
    metric = scale(metric)
    logts = scale(logts)
    traces_err = scale(traces_err)
    traces_lat = scale(traces_lat)

    # if traces_err only has 1 column, or traces_lat only has 1 column, consider it None
    if traces_err is not None and len(traces_err.columns) <= 2:
        traces_err = None
    if traces_lat is not None and len(traces_lat.columns) <= 2:
        traces_lat = None
    return {
        "metric": metric,
        "logts": logts,
        "tracets_err": traces_err,
        "tracets_lat": traces_lat
    }

def sanity_check(name, df):
    if df is None:
        print(f"[{name}] is None")
        return

    arr = df.values.astype(np.float32)

    print(f"\n[{name}]")
    print("shape:", arr.shape)
    print("mean:", np.mean(arr))
    print("std :", np.std(arr))
    print("min :", np.min(arr))
    print("max :", np.max(arr))
    print("nan :", np.isnan(arr).sum())
    print("inf :", np.isinf(arr).sum())

def build_service_map(columns):
    service_map = {}

    for col in columns:
        if col == "time":
            continue

        service = col.split("_")[0]

        if service not in service_map:
            service_map[service] = []
        service_map[service].append(col)

    return service_map

import numpy as np
import numpy as np

def to_service_tensor(df, service_map):
    """
    Converts dataframe → [T, N, F]

    T = time
    N = services
    F = feature dimension per service (varies but padded to max)
    """

    if df is None:
        return None, None

    df = df.copy()

    if "time" in df.columns:
        df = df.drop(columns=["time"])

    services = list(service_map.keys())
    N = len(services)

    # feature dimension per service
    feature_dims = {s: len(cols) for s, cols in service_map.items()}
    max_F = max(feature_dims.values()) if feature_dims else 1

    T = len(df)
    aligned = np.zeros((T, N, max_F), dtype=np.float32)

    for i, svc in enumerate(services):
        cols = service_map.get(svc, [])

        if len(cols) == 0:
            continue

        sub = df[cols].values.astype(np.float32)   # [T, F_svc]

        aligned[:, i, :sub.shape[1]] = sub

    return aligned, services


class TensorBuilder:
    def __init__(self, seq_len=12, stride=1):
        self.seq_len = seq_len
        self.stride = stride

    def build(self, data_dict):

        metric = data_dict["metric"]
        logts = data_dict["logts"]
        err = data_dict.get("tracets_err", None)
        lat = data_dict.get("tracets_lat", None)

        # -------------------------
        # 1. canonical service space
        # -------------------------
        service_map = build_service_map(metric.columns)
        services = list(service_map.keys())

        # -------------------------
        # 2. align ALL modalities to metric services
        # -------------------------
        metric_3d, _ = to_service_tensor(metric, service_map)

        log_3d = self.align_to_services(logts, services)
        err_3d = self.align_to_services(err, services) if err is not None else None
        lat_3d = self.align_to_services(lat, services) if lat is not None else None

        # check alignment sanity
        if log_3d is not None and log_3d.shape[1] != len(services):
            raise ValueError(f"Log alignment failed: expected {len(services)}")
        if err_3d is not None and err_3d.shape[1] != len(services):
            raise ValueError(f"Traces_err alignment failed: expected {len(services)}")
        if lat_3d is not None and lat_3d.shape[1] != len(services):
            raise ValueError(f"Traces_lat alignment failed: expected {len(services)}")

        # -------------------------
        # 3. window everything consistently
        # -------------------------
        metric_w = self.window(metric_3d)
        log_w = self.window(log_3d)
        err_w = self.window(err_3d) if err_3d is not None else None
        lat_w = self.window(lat_3d) if lat_3d is not None else None

        return {
            "metric": metric_w,
            "logts": log_w,
            "tracets_err": err_w,
            "tracets_lat": lat_w,
            "services": services
        },service_map

    # -------------------------
    # logs/traces alignment
    # -------------------------
    def align_to_services(self, df, services):
        """
        Converts any modality → [T, N, 1]
        by aggregating per service.
        """

        if df is None:
            return None

        df = df.copy()
        if "time" in df.columns:
            df = df.drop(columns=["time"])

        service_map = build_service_map(df.columns)

        T = len(df)
        N = len(services)

        out = np.zeros((T, N, 1), dtype=np.float32)

        for i, svc in enumerate(services):
            cols = service_map.get(svc, None)
            if cols:
                out[:, i, 0] = df[cols].mean(axis=1).values

        return out

    # -------------------------
    # windowing
    # -------------------------
    def window(self, x):
        """
        [T, N, F] → [B, seq_len, N, F]
        """

        if x is None:
            return None

        T, N, F = x.shape
        L = self.seq_len

        if T < L:
            pad = np.zeros((L - T, N, F), dtype=x.dtype)
            x = np.concatenate([pad, x], axis=0)
            T = L

        windows = []
        for t in range(0, T - L + 1, self.stride):
            windows.append(x[t:t + L])

        return np.stack(windows, axis=0)
    

#-------------------
# Model building, training, inference
#-------------------
def build_hub_plus_self_graph(num_nodes):
    A = np.eye(num_nodes, dtype=np.float32)

    # hub connections
    A[0, :] = 1
    A[:, 0] = 1

    return torch.tensor(A)

def build_model(model_class, model_config, args, graph=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = build_hub_plus_self_graph(model_config.num_services)

    if model_class == "Eadro":
        model = Eadro(
            model_config.log_len,
            model_config.raw_node,
            model_config.raw_edge,
            graph = graph
        )
    elif model_class == "SFlexRCAmulti":
        model = SFlexRCAmulti(
            model_config.log_len,
            model_config.raw_node,
            model_config.raw_edge
        )

    elif model_class == "Anofusion":
        model = AnoFusion(
            num_services=model_config.num_services,     # or raw_node count (IMPORTANT)
            window_size=12,#TODO: this should ideally be inferred from data, but we can keep it fixed for now
            metric_dim=model_config.raw_node,
            log_dim=model_config.log_len,
            trace_dim=model_config.raw_edge,
            graph=graph
        )

    elif model_class == "Art":
        model = Art_Model(
            graph,
            model_config.raw_node,
            model_config.log_len,
            model_config.raw_edge
        )
    else:
        raise ValueError(model_class)

    return model.to(device)

def sanity_check_before_training(metric, logts, traces, model_config):
    B, T, N, _ = metric.shape 
    print(f"Sanity check: metric shape={metric.shape}")
    if logts is not None:
        if logts.shape[0] != B or logts.shape[1] != T or logts.shape[2] != N:
            raise ValueError(f"Log shape mismatch: expected [B, T, N, L], got {logts.shape}")
        print(f"Sanity check: log shape={logts.shape}")
    if traces is not None:
        if traces.shape[0] != B or traces.shape[1] != T or traces.shape[2] != N or traces.shape[3] != N:
            raise ValueError(f"Traces shape mismatch: expected [B, T, N, N, D], got {traces.shape}")
        # check if N,N 
        if traces.shape[2] != traces.shape[3]:
            raise ValueError(f"Traces should have shape [B, T, N, N, D], got {traces.shape}")
        
        print(f"Sanity check: traces shape={traces.shape}")
        
def train_model(model, data_dict, model_config, device, seq_len=12, epochs=100):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = torch.nn.MSELoss()

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)

        if isinstance(x, pd.DataFrame):
            return torch.tensor(x.values, dtype=torch.float32, device=device)

        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32, device=device)

        return None  # allow missing traces

    metric = to_tensor(data_dict["metric"])
    logts = to_tensor(data_dict["logts"])
    traces_err = data_dict.get("tracets_err")
    traces_lat = data_dict.get("tracets_lat")

    if traces_err is not None or traces_lat is not None:
        # safely handle missing modalities
        parts = []
        if traces_err is not None:
            parts.append(traces_err)
        if traces_lat is not None:
            parts.append(traces_lat)

        # concatenate along feature dimension (last axis)
        traces = np.concatenate(parts, axis=-1)
        traces = to_tensor(traces)
        #change traces from B,T,N,D to B,T,N,N,D
        B, T, N, C = traces.shape
        traces = traces.unsqueeze(-2).expand(B, T, N, N, C)
        # sanity check
        # traces should now be [B, T, N, N, D] where D is combined trace feature dimension
        # so check if N == N 
        if traces.shape[2] != traces.shape[3]:
            raise ValueError(f"Traces should have shape [B, T, N, N, D], got {traces.shape}")

    else:
        traces = None

    sanity_check_before_training(data_dict["metric"], data_dict["logts"], traces, model_config)
    # target = reconstruction target (same fused space)
    def build_target(metric, logts, traces):
        """
        Builds unified target representation:
        [B, N, F]
        """

        # ---- metric ----
        metric = metric.mean(dim=1) if metric is not None else None   # [B, N, M]

        # ---- logts ----
        logts = logts.mean(dim=1) if logts is not None else None      # [B, N, L]

        # ---- traces ----
        if traces is None:
            traces_node = None
        else:
            if traces.dim() == 5:
                traces = traces.mean(dim=1)        # [B, N, N, D]
            elif traces.dim() == 4:
                pass
            else:
                raise ValueError(f"Unexpected traces shape: {traces.shape}")

            traces_node = traces.mean(dim=2)       # [B, N, D]

        # ---- safe concat ----
        parts = []
        if metric is not None:
            parts.append(metric)
        if logts is not None:
            parts.append(logts)
        if traces_node is not None:
            parts.append(traces_node)

        return torch.cat(parts, dim=-1)

    target = build_target(metric, logts, traces)
    patience = 10
    min_delta = 1e-4
    best_loss = float("inf")
    wait = 0
    for ep in range(epochs):

        optimizer.zero_grad()

        out = model({
            "metric": metric,
            "logts": logts,
            "traces": traces,
        })

        loss = mse(out, target)
        loss_value = loss.item()

        loss.backward()
        optimizer.step()

        print(f"[epoch {ep}] loss={loss_value:.6f}")

        # -------------------------
        # EARLY STOPPING (PATIENCE)
        # -------------------------
        if best_loss - loss_value > min_delta:
            best_loss = loss_value
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"[early stop] no improvement for {patience} epochs")
            break

def infer_model(model, data_dict, device):

    model.eval()
    traces = None
    if data_dict.get("tracets_err") is not None or data_dict.get("tracets_lat") is not None:
        parts = []
        if data_dict.get("tracets_err") is not None:
            parts.append(data_dict["tracets_err"])
        if data_dict.get("tracets_lat") is not None:
            parts.append(data_dict["tracets_lat"])

        traces = np.concatenate(parts, axis=-1)
        traces = torch.tensor(traces, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model({
            "metric": data_dict["metric"],
            "logts": data_dict["logts"],
            "traces": traces,
        })

    return out



#-------------------
# Scoring, ed scores, baro scores, fusion
#-------------------
def compute_scores(data_dict, recon_dict, CONFIG=None):

    scores = {}

    for k in ["metric", "logts", "tracets_err", "tracets_lat"]:

        if data_dict.get(k) is None or recon_dict.get(k) is None:
            continue

        real = data_dict[k].to_numpy()
        pred = recon_dict[k].cpu().numpy()

        residual = pred - real  # (T, D)

        num_vars = residual.shape[-1]
        pot_scores = np.zeros(num_vars)

        for i in range(num_vars):

            series = residual[:, i]

            pot_val, _, _ = compute_pot_scores(
                series,
                risk=getattr(CONFIG, "pot_risk", 1e-2),
                init_level=getattr(CONFIG, "pot_init_level", 0.98),
                num_candidates=getattr(CONFIG, "pot_num_candidates", 10),
                epsilon=getattr(CONFIG, "pot_epsilon", 1e-8),
            )

            pot_scores[i] = pot_val

        scores[k] = pot_scores

    return scores

def collapse_time(x):
    if x is None:
        return None

    if isinstance(x, np.ndarray):
        return x.mean(axis=1)

    return x.mean(dim=1) # [B, N, F] or [B, N, N, D]
    
def compute_ed_scores(data_dict, recon):

    device = recon.device
    B, N, F = recon.shape

    # ---------------- metric ----------------
    metric = collapse_time(data_dict.get("metric"))

    # ---------------- log ----------------
    logts = collapse_time(data_dict.get("logts"))

    # ---------------- edge ----------------
    edge = data_dict.get("edge")
    if edge is not None:
        edge = edge.mean(dim=1)      # [B,N,N,D]
        edge = edge.mean(dim=2)      # [B,N,D]

    # ---------------- determine expected dims ----------------
    metric_dim = 0 if metric is None else metric.shape[-1]
    log_dim    = 0 if logts is None else logts.shape[-1]
    edge_dim   = 0 if edge is None else edge.shape[-1]

    missing_dim = F - (metric_dim + log_dim + edge_dim)

    if missing_dim < 0:
        raise ValueError(
            f"Target larger than recon: "
            f"metric={metric_dim}, log={log_dim}, edge={edge_dim}, recon={F}"
        )

    # ---------------- fill missing modality ----------------
    if metric is None:
        metric = torch.zeros(B, N, missing_dim, device=device)
        missing_dim = 0

    if logts is None:
        logts = torch.zeros(B, N, missing_dim, device=device)
        missing_dim = 0

    if edge is None:
        edge = torch.zeros(B, N, missing_dim, device=device)
        missing_dim = 0

    # ---------------- build target ----------------
    def to_torch(x, device=None):#TODO: ideally this should be handled outside the model, but we can keep it here for flexibility
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        return x
    metric = to_torch(metric)
    logts  = to_torch(logts)
    edge   = to_torch(edge)

    target = torch.cat([metric, logts, edge], dim=-1)

    if target.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: recon={recon.shape}, target={target.shape}"
        )

    residual = (recon - target).abs()

    return residual.mean(dim=(0, 2)).detach().cpu().numpy()


def compute_baro_scores(
    full_data,
    node_names,
    inject_time=None,
    time_array=None,
    scalar_type="Robust",
    seq_len=12,
    **kwargs
):
    """
    BARO-style statistical deviation scoring.
    Works on raw dataframe (NOT tensors).
    """

    # -------------------------
    # split normal/anomaly
    # -------------------------
    if inject_time is not None and time_array is not None:
        normal_df = full_data[time_array < inject_time]
        anomal_df = full_data[time_array >= inject_time]
    else:
        split = len(full_data) // 2
        normal_df = full_data.iloc[:split]
        anomal_df = full_data.iloc[split:]

    baro_scores = []

    for col in node_names:

        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        if scalar_type == "Robust":
            scaler = RobustScaler().fit(a.reshape(-1, 1))
            z = scaler.transform(b.reshape(-1, 1))[:, 0]

        elif scalar_type == "Standard":
            scaler = StandardScaler().fit(a.reshape(-1, 1))
            z = scaler.transform(b.reshape(-1, 1))[:, 0]

        elif scalar_type == "Quantile":
            scaler = QuantileTransformer(output_distribution="normal")
            scaler.fit(a.reshape(-1, 1))
            z = scaler.transform(b.reshape(-1, 1))[:, 0]

        elif scalar_type == "MAD":
            med = np.median(a)
            mad = np.median(np.abs(a - med)) + 1e-8
            z = np.abs(b - med) / mad

        elif scalar_type == "IQR":
            q1, q3 = np.percentile(a, [25, 75])
            iqr = q3 - q1 + 1e-8
            z = np.clip((b - q3) / iqr, 0, None)

        elif scalar_type == "ModifiedZ":
            med = np.median(a)
            mad = np.median(np.abs(a - med)) + 1e-8
            z = 0.6745 * (b - med) / mad

        elif scalar_type == "Rank":
            ra = np.argsort(np.argsort(a)) / (len(a) - 1)
            rb = np.argsort(np.argsort(b)) / (len(b) - 1)
            z = np.abs(rb - ra.mean())

        else:
            raise ValueError(f"Unknown scalar_type: {scalar_type}")

        baro_scores.append(np.max(z))

    return np.array(baro_scores)



F_NODE = "F-node"

def _rcd_multimodal(data, inject_time, dataset=None, gamma=5, localized=True, bins=5, verbose=False, **kwargs):
    """RCD variant that accepts multimodal dict input (metric + logts).
    Ported from cfm/e2e/rcd.py multimodal branch."""
    metric = data["metric"]
    logts = data["logts"]

    # === metric ===
    metric = metric.iloc[::15, :]

    normal_metric = metric[metric["time"] < inject_time]
    anomal_metric = metric[metric["time"] >= inject_time]
    normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=False)
    anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=False)
    intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]
    normal_metric = normal_metric[intersect]
    anomal_metric = anomal_metric[intersect]

    normal_data = normal_metric
    anomal_data = anomal_metric

    # == logts ==
    logts = drop_constant(logts)
    normal_logts = logts[logts["time"] < inject_time].drop(columns=["time"])
    anomal_logts = logts[logts["time"] >= inject_time].drop(columns=["time"])

    normal_data = pd.concat([normal_data, normal_logts], axis=1)
    anomal_data = pd.concat([anomal_data, anomal_logts], axis=1)

    normal_data = normal_data.loc[:, ~normal_data.columns.duplicated()]
    normal_data = normal_data.fillna(0)

    anomal_data = anomal_data.loc[:, ~anomal_data.columns.duplicated()]
    anomal_data = anomal_data.fillna(0)

    normal_df = drop_constant(convert_mem_mb(drop_time(normal_data)))
    anomal_df = drop_constant(convert_mem_mb(drop_time(anomal_data)))

    normal_df, anomal_df = _match_columns(normal_df, anomal_df)

    df = add_fnode_and_concat(normal_df, anomal_df)
    normal_df = df[df[F_NODE] == "0"].drop(columns=[F_NODE])
    anomal_df = df[df[F_NODE] == "1"].drop(columns=[F_NODE])

    rc = run_multi_phase(normal_df, anomal_df, gamma, localized, bins, verbose)
    return {"ranks": rc}

def compute_torai_scores(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, normalize=True, addup=False, borda=False, service=None, fault_type=None, case=None, rank=None, enable_percentile=False, enable_weighted_rank=False, **kwargs):
    scaler_function = kwargs.get("scaler_function", StandardScaler)
    metric = data["metric"]
    logts = data["logts"]
    traces_err = data.get("tracets_err", pd.DataFrame())
    traces_lat = data.get("tracets_lat", pd.DataFrame())

    has_traces = traces_err is not None and len(traces_err) > 0

    # ==== PREPARE DATA ====
    # the metric is sampled every second, resample for 15s
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
    if has_traces:
        traces_err = traces_err.ffill()
        traces_err = traces_err.fillna(0)
        traces_err = drop_constant(traces_err)

        normal_traces_err = traces_err[traces_err["time"] < inject_time].drop(columns=["time"])
        anomal_traces_err = traces_err[traces_err["time"] >= inject_time].drop(columns=["time"])

    # == traces_lat ==
    if has_traces:
        traces_lat = traces_lat.ffill()
        traces_lat = traces_lat.fillna(0)
        traces_lat = drop_constant(traces_lat)
        normal_traces_lat = traces_lat[traces_lat["time"] < inject_time].drop(columns=["time"])
        anomal_traces_lat = traces_lat[traces_lat["time"] >= inject_time].drop(columns=["time"])

    # ==== PROCESS ====
    ranks = []
    metric_ranks = []
    log_ranks = []
    trace_err_ranks = []
    trace_lat_ranks = []

    # == metric ==
    for col in normal_metric.columns:
        a = normal_metric[col].to_numpy()
        b = anomal_metric[col].to_numpy()

        scaler = scaler_function().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        zscores = np.abs(zscores)
        if enable_percentile:
            score = np.percentile(zscores, 95)
        else:
            score = max(zscores)
        metric_ranks.append((col, score))
    metric_ranks = sorted(metric_ranks, key=lambda x: x[1], reverse=True)
    metric_ranks = [(x[0], x[1] / sum([x[1] for x in metric_ranks])) for x in metric_ranks]
    ranks = metric_ranks.copy()

    # == logs ==
    for col in normal_logts.columns:
        a = normal_logts[col].to_numpy()
        b = anomal_logts[col].to_numpy()

        if a.size == 0:
            continue

        scaler = scaler_function().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        zscores = np.abs(zscores)
        if enable_percentile:
            score = np.percentile(zscores, 95)
        else:
            score = max(zscores)
        log_ranks.append((col, score))
    log_ranks = sorted(log_ranks, key=lambda x: x[1], reverse=True)
    log_ranks = [(x[0], x[1] / sum([x[1] for x in log_ranks])) for x in log_ranks]
    ranks.extend(log_ranks)

    # == traces_err ==
    if has_traces:
        for col in normal_traces_err.columns:
            a = normal_traces_err[col].to_numpy()[:-2]
            b = anomal_traces_err[col].to_numpy()

            scaler = scaler_function().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            zscores = np.abs(zscores)
            if enable_percentile:
                score = np.percentile(zscores, 95)
            else:
                score = max(zscores)
            if scaler.mean_ == 0 and scaler.var_ == 0:
                score = score * 1e9
            trace_err_ranks.append((col, score))
        trace_err_ranks = sorted(trace_err_ranks, key=lambda x: x[1], reverse=True)
        trace_err_ranks = [(x[0], x[1] / sum([x[1] for x in trace_err_ranks])) for x in trace_err_ranks]
        ranks.extend(trace_err_ranks)

    # == traces_lat ==
    if has_traces:
        for col in normal_traces_lat.columns:
            a = normal_traces_lat[col].to_numpy()
            b = anomal_traces_lat[col].to_numpy()

            scaler = scaler_function().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            zscores = np.abs(zscores)
            if enable_percentile:
                score = np.percentile(zscores, 95)
            else:
                score = max(zscores)
            trace_lat_ranks.append((col, score))
        trace_lat_ranks = sorted(trace_lat_ranks, key=lambda x: x[1], reverse=True)
        trace_lat_ranks = [(x[0], x[1] / sum([x[1] for x in trace_lat_ranks])) for x in trace_lat_ranks]
        ranks.extend(trace_lat_ranks)

    def fine2coarse_addup(fine_ranks):
        _coarse_ranks = [(i.split("_")[0], s) for i, s in fine_ranks]
        score_dict = {}
        for i, s in _coarse_ranks:
            if i in score_dict:
                score_dict[i] += s
            else:
                score_dict[i] = s
        coarse_ranks = [(i, s) for i, s in score_dict.items()]
        coarse_ranks = sorted(coarse_ranks, key=lambda x: x[1], reverse=True)
        return coarse_ranks

    def fine2coarse_highest(fine_ranks):
        if not fine_ranks:
            return []
        fine_ranks = sorted(fine_ranks, key=lambda x: x[1], reverse=True)
        _coarse_ranks = [(i.split("_")[0], s) for i, s in fine_ranks]

        coarse_ranks = [_coarse_ranks[0]]
        for svc, score in _coarse_ranks[1:]:
            if svc not in [i for i, _ in coarse_ranks]:
                coarse_ranks.append((svc, score))

        coarse_ranks = [(i, s / sum([s for _, s in coarse_ranks])) for i, s in coarse_ranks]
        return coarse_ranks

    svc_metric_ranks = fine2coarse_addup(metric_ranks)
    svc_log_ranks = fine2coarse_highest(log_ranks)
    svc_trace_lat_ranks = fine2coarse_addup(trace_lat_ranks)
    svc_trace_err_ranks = fine2coarse_addup(trace_err_ranks)

    score_dict = {}
    for i, s in svc_metric_ranks + svc_log_ranks:
        if i in score_dict:
            score_dict[i] += s
        else:
            score_dict[i] = s
    svc_ranks = [(i, s) for i, s in score_dict.items()]
    svc_ranks = sorted(svc_ranks, key=lambda x: x[1], reverse=True)

    # change service "frontendservice" to "frontend"
    svc_trace_lat_ranks = [("frontend", s) if i == "frontendservice" else (i, s) for i, s in svc_trace_lat_ranks]
    svc_trace_err_ranks = [("frontend", s) if i == "frontendservice" else (i, s) for i, s in svc_trace_err_ranks]

    m = {
        "metric": pd.Series([i[1] for i in svc_metric_ranks], index=[i[0] for i in svc_metric_ranks]),
        "log": pd.Series([i[1] for i in svc_log_ranks], index=[i[0] for i in svc_log_ranks]),
        "trace_lat": pd.Series([i[1] for i in svc_trace_lat_ranks], index=[i[0] for i in svc_trace_lat_ranks]),
        "trace_err": pd.Series([i[1] for i in svc_trace_err_ranks], index=[i[0] for i in svc_trace_err_ranks]),
    }

    m = pd.DataFrame(m)
    m = m.fillna(0)

    service_list = m.index.to_list()

    # =========================
    # TORAI NUMERICAL SERVICE SCORE
    # =========================

    X = m.to_numpy(dtype=np.float32)  # [S, 4 modalities]

    # additive evidence fusion (TORAI core principle)
    if enable_weighted_rank:
        w = np.std(X, axis=0)

        if w.sum() < 1e-12:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()

        service_scores = X @ w
    else:
        service_scores = X.sum(axis=1)#TODO check if it is even being used 

    # normalize to stable scale
    if normalize:
        s = service_scores.sum()
        if s > 0:
            service_scores = service_scores / s


    X_train = m.to_numpy()
    bic_score_all = []

    for n_comp in range(1, X_train.shape[0] + 1):
        estimator = GaussianMixture(
            n_components=n_comp,
            covariance_type="full",
            max_iter=50, random_state=0
        )
        estimator.fit(X_train)
        bic_score = estimator.bic(X_train)
        bic_score_all.append(bic_score)
    idx_min = np.argmin(bic_score_all)
    n_comp_opt = idx_min + 1
    print("Optimal number of cluters: {}".format(n_comp_opt))

    estimator = GaussianMixture(
        n_components=n_comp_opt,
        covariance_type="full",
        max_iter=50, random_state=0
    )
    estimator.fit(X_train)
    y_pred = estimator.predict(X_train)
    y_train = np.mean(X_train, axis=1)

    cluster_rank = []
    for cluster_idx in list(set(y_pred)):
        service_indices_of_this_cluster = np.where(y_pred == cluster_idx)
        services_of_this_cluster = [service_list[i] for i in service_indices_of_this_cluster[0]]

        scores_of_them = y_train[service_indices_of_this_cluster]
        cluster_score = np.mean(scores_of_them)
        cluster_rank.append((cluster_idx, cluster_score))

    cluster_rank.sort(key=lambda x: x[1], reverse=True)

    service_ranks = []
    service_ranks_rcd = []
    for cluster_idx, score in cluster_rank:
        service_indices_of_this_cluster = np.where(y_pred == cluster_idx)
        services_of_this_cluster = [service_list[i] for i in service_indices_of_this_cluster[0]]
        scores_of_them = y_train[service_indices_of_this_cluster]

        if len(services_of_this_cluster) == 1:
            service_ranks.append(services_of_this_cluster[0])
            service_ranks_rcd.append(services_of_this_cluster[0])
            continue

        # sort by score within cluster
        aa = list(zip(services_of_this_cluster, scores_of_them))
        aa.sort(key=lambda x: x[1], reverse=True)
        for a, s in aa:
            service_ranks.append(a)

        # get metric subset for cluster
        tmp_metric = metric.loc[:, metric.columns.str.startswith(tuple(services_of_this_cluster))]
        tmp_metric["time"] = metric["time"]

        # get log subset for cluster
        tmp_logts = logts.loc[:, logts.columns.str.startswith(tuple(services_of_this_cluster))]
        tmp_logts["time"] = logts["time"]

        # get trace subsets for cluster
        tmp_traces_err = None
        tmp_traces_lat = None
        if has_traces:
            tmp_traces_err = traces_err.loc[:, traces_err.columns.str.startswith(tuple(services_of_this_cluster))]
            tmp_traces_err["time"] = traces_err["time"]

            tmp_traces_lat = traces_lat.loc[:, traces_lat.columns.str.startswith(tuple(services_of_this_cluster))]
            tmp_traces_lat["time"] = traces_lat["time"]

        tmp_ranks = _rcd_multimodal(
            data={
                "metric": tmp_metric,
                "logts": tmp_logts,
                "tracets_err": tmp_traces_err,
                "tracets_lat": tmp_traces_lat,
            },
            inject_time=inject_time,
            dataset=dataset,
        )
        tmp_ranks = [s.split("_")[0] for s in tmp_ranks["ranks"]]
        internal_service_ranks = []
        if tmp_ranks:
            internal_service_ranks = [tmp_ranks[0]]
            for s in tmp_ranks[1:]:
                if s not in internal_service_ranks:
                    internal_service_ranks.append(s)

        if len(internal_service_ranks) == len(services_of_this_cluster):
            service_ranks_rcd.extend(internal_service_ranks)
        else:
            for a, s in aa:
                service_ranks_rcd.append(a)

    #ranks = [svc for svc in service_ranks_rcd]

    return ranks
    
def fuse_scores(
    ed_scores,
    baro_scores,
    method="static",
    alpha=0.6
):
    """
    Fuse ED + BARO into final anomaly scores.
    """

    ed_scores = np.asarray(ed_scores)
    baro_scores = np.asarray(baro_scores)

    if method == "static":
        return alpha * ed_scores + (1 - alpha) * baro_scores

    elif method == "max":
        return np.maximum(ed_scores, baro_scores)

    elif method == "geometric":
        return np.sqrt(np.abs(ed_scores * baro_scores))

    elif method == "rank":
        ed_r = np.argsort(np.argsort(-ed_scores))
        baro_r = np.argsort(np.argsort(-baro_scores))

        agreement = np.abs(ed_r - baro_r)
        agreement = agreement / (agreement.max() + 1e-8)

        w = 1 - agreement
        return w * ed_scores + (1 - w) * baro_scores

    elif method == "attention":
        s = np.stack([ed_scores, baro_scores], axis=1)
        w = np.exp(s)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
        return w[:, 0] * ed_scores + w[:, 1] * baro_scores

    elif method == "rank_product":
        ed_r = np.argsort(np.argsort(-ed_scores)) + 1
        baro_r = np.argsort(np.argsort(-baro_scores)) + 1
        return 1.0 / (ed_r * baro_r)

    else:
        raise ValueError(f"Unknown fusion method: {method}")

def collapse_baro_to_services(baro_scores, service_map, baro_columns):
    baro_index = {c: i for i, c in enumerate(baro_columns)}

    service_scores = []

    for svc, cols in service_map.items():
        idx = [baro_index[c] for c in cols if c in baro_index]

        if len(idx) == 0:
            service_scores.append(0.0)
        else:
            service_scores.append(np.mean(baro_scores[idx]))

    return np.array(service_scores)

#=========================
# Main multimodal RCA function
#=========================
def RMDnet_Multimodality(
    data,
    inject_time=None,
    dataset=None,
    model_class=None,
    model_config=None,
    ensemble_method="static",
    with_baro_post=False,
    scalar_type=None,
    tensor_builder=None,
    **kwargs
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 1. preprocess
    # -------------------------
    data_dict_original = data.copy()  # avoid modifying original data
    data_dict = preprocess_multimodal(data, inject_time, dataset, kwargs)

    sanity_check("metric", data_dict["metric"])
    sanity_check("logts", data_dict["logts"])
    sanity_check("tracets_err", data_dict["tracets_err"])
    sanity_check("tracets_lat", data_dict["tracets_lat"])

    # -------------------------
    # 2. tensor construction
    # -------------------------
    tensor_builder = TensorBuilder()

    tensors, service_map = tensor_builder.build(data_dict)
    node_names = tensors["services"]

    # -------------------------
    # 3. model config
    # -------------------------
    model_config.log_len = tensors["logts"].shape[-1] if tensors["logts"] is not None else 0
    model_config.raw_node = tensors["metric"].shape[-1] if tensors["metric"] is not None else 0

    trace_dim = 0
    if data_dict.get("tracets_err") is not None:
        trace_dim += tensors["tracets_err"].shape[-1]
    if data_dict.get("tracets_lat") is not None:
        trace_dim += tensors["tracets_lat"].shape[-1]

    model_config.raw_edge = trace_dim
    model_config.num_services = len(node_names)
    model = build_model(model_class, model_config, kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    num_trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    model_size_mb = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 ** 2)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_start = time.time()
    train_model(model, tensors, model_config, device)
    train_time = time.time() - train_start

    infer_start = time.time()
    recon = infer_model(model, tensors, device)
    infer_time = time.time() - infer_start
    if torch.cuda.is_available():
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )
    else:
        peak_memory_mb = 0.0
    energy_joules = 0.0

    if torch.cuda.is_available():
        try:
            import pynvml

            pynvml.nvmlInit()

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            power_watts = (
                pynvml.nvmlDeviceGetPowerUsage(handle)
                / 1000.0
            )

            energy_joules = power_watts * (
                train_time + infer_time
            )

        except:
            energy_joules = 0.0
    # -------------------------
    # 4. ED scoring
    # -------------------------
    ed_scores = compute_ed_scores(tensors, recon)

    # -------------------------
    # 5. BARO
    # -------------------------
    if with_baro_post:
        #baro_scores = compute_baro_scores(
        #    full_data=data_dict["metric"],
        #    node_names=data_dict["metric"].columns.tolist(),
        #    inject_time=inject_time,
        #    time_array=None,
        #    scalar_type=scalar_type,
        #    seq_len=kwargs.get("seq_len", 12),
        #    **kwargs
        #)
#
        #baro_scores = collapse_baro_to_services(
        #    baro_scores,
        #    service_map,
        #    baro_columns=data_dict["metric"].columns.tolist()
        #)
        if model_class == "SFlexRCAmulti":
            torai_out = compute_torai_scores(
                data=data_dict_original,
                node_names=node_names,
                inject_time=inject_time,
                scaler_function=StandardScaler,
                enable_percentile=True,
                enable_weighted_rank=True,
                topk=5
            )
        else:
            torai_out = compute_torai_scores(
                data=data_dict_original,
                node_names=node_names,
                inject_time=inject_time,
                scaler_function=StandardScaler,
                topk=5
            )

        metric_to_service = {
            metric: svc
            for svc, metrics in service_map.items()
            for metric in metrics
        }

        service_scores = {svc: 0.0 for svc in service_map}

        for metric_name, score in torai_out:
            svc = metric_to_service.get(metric_name)
            if svc is not None:
                service_scores[svc] += score

        torai_scores = np.array(
            [service_scores[svc] for svc in service_map.keys()],
            dtype=np.float32
        )

        # optional normalization
        torai_scores /= (torai_scores.sum() + 1e-12)

        # handle both dict / array return safely
        #if isinstance(torai_out, dict):
        #    baro_scores = np.array([
        #        torai_out["scores"].get(n, 0.0)
        #        for n in node_names
        #    ])
        #else:
        #    baro_scores = np.asarray(torai_out)
    else:
        torai_scores = np.zeros_like(ed_scores)

    # -------------------------
    # 6. fusion
    # -------------------------
    hybrid_scores = fuse_scores(
        ed_scores,
        torai_scores,
        method=ensemble_method
    )

    # -------------------------
    # 7. ranking
    # -------------------------
    ranks = [node_names[i] for i in np.argsort(-hybrid_scores)]

    return {
        "scores": hybrid_scores.tolist(),
        "ranks": ranks,
        "ed_scores": ed_scores.tolist(),
        "baro_scores": torai_scores.tolist(),

        "train_time": train_time,
        "infer_time": infer_time,
        "total_time": train_time + infer_time,

        "num_params": num_params,
        "num_trainable_params": num_trainable_params,
        "model_size_mb": model_size_mb,
        "peak_memory_mb": peak_memory_mb,
        "energy_joules": energy_joules,
    }
