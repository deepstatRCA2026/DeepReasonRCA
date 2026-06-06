import argparse
import gc
import utils
import glob
import json
import os
import shutil
import warnings
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import Pool
from os.path import abspath, basename, dirname, exists, join
#from RCAEval.e2e.models.SFlexRCA import OrthTransform
# turn off all warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.classes.graph import Node

from RCAEval.io.time_series import drop_constant, drop_time, preprocess
from RCAEval.utility import (
    dump_json,
    is_py38,
    is_py312,
    load_json,
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
    download_re1_dataset,
    download_re2_dataset,
    download_re3_dataset, 
    download_multi_source_sample
)

# clear contents of output/results folder, bec it uses it for generating the results
if os.path.exists("output/results"):
    shutil.rmtree("output/results")

if is_py312():
    from RCAEval.e2e import (
        baro,
        causalrca,
        causalrca_anomaly,
        circa,
        cloudranger,
        cmlp_pagerank,
        dummy,
        e_diagnosis,
        easyrca,
        fci_pagerank,
        fci_randomwalk,
        ges_pagerank,
        granger_pagerank,
        granger_randomwalk,
        lingam_pagerank,
        lingam_randomwalk,
        micro_diag,
        microcause,
        microrank,
        mscred,
        nsigma,
        ntlr_pagerank,
        ntlr_randomwalk,
        pc_pagerank,
        pc_randomwalk,
        run,
        rcd,
        mmrcd,
        RMDnet,
        RMDnet_Multimodality,
        tracerca,
        torai
    )

elif is_py38():
    from RCAEval.e2e import dummy, e_diagnosis, ht, rcd, mmrcd
else:
    print("Please use Python 3.8 or 3.12")
    exit(1)

try:
    import torch
    ####os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    from RCAEval.e2e.causalrca import causalrca
except ImportError:
    pass


def parse_args():
    """
    options:
  -h, --help            Show this help message and exit
  --dataset DATASET     Choose a dataset. Valid options:
                        [re2-ob, re2-ss, re2-tt, etc.]
  --method METHOD       Choose a method (`causalrca`, `microcause`, `e_diagnosis`, `baro`, `rcd`, `circa`, etc.)
  """
    parser = argparse.ArgumentParser(description="RCAEval evaluation")
    parser.add_argument("--method", type=str, default="easyrca", help="Choose a method.")
    parser.add_argument("--model_class", type=str, default="TimeMixerpp", help="Choose a model class.")
    parser.add_argument("--dataset", type=str, default="re1-ob", help="Choose a dataset.", choices=[#sock-shop-1
        "online-boutique", "sock-shop-1", "sock-shop-2", "train-ticket",
        "re1-ob", "re1-ss", "re1-tt", "re2-ob", "re2-ss", "re2-tt", "re3-ob", "re3-ss", "re3-tt",
        "torai-ob", "torai-ss", "torai-tt"
    ])
    parser.add_argument("--length", type=int, default=100, help="Time series length (RQ4)")
    parser.add_argument("--tdelta", type=int, default=0, help="Specify $t_delta$ `to simulate delay in anomaly detection")
    parser.add_argument("--test", action="store_true", default=False, help="Perform smoke test on certain methods without fully run on all data")
    # include logs or not
    parser.add_argument("--with-logs", action="store_true", default=False, help="Include logs as input")
    parser.add_argument("--combine_baro_pre", action="store_true", default=False, help="Combine BARO with another method")
    parser.add_argument("--combine_baro_post", action="store_true", default=False, help="Combine BARO with another method")
    # "ensemble_method" -> only works if combine_baro_post is true (and uses rank as it is the best)
    parser.add_argument("--ensemble_method", type=str, default="rank", help="How to combine deep and stat methods, only works if combine_baro_post is true. Options: static (equal weight), dynamic (weight based on confidence), rank (choose the one with higher rank), sigmoid (weight based on sigmoid of rank)")
    parser.add_argument("--scaler_type", type=str, default="Robust", help="Scaler type to use")
    parser.add_argument("--num_files", type=int, default=10, help="Number of files to process, default is all")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--research_question", type=str, default="RQ_1", help="Specify the research question")
    parser.add_argument("--seed", type=int, default=22, help="Random seed")
    args = parser.parse_args()

    if args.method not in globals():
        raise ValueError(f"{args.method=} not defined. Please check imported methods.")

    return args

args = parse_args()
print(args.scaler_type)
# download dataset
if "online-boutique" in args.dataset or "re1-ob" in args.dataset:
    download_online_boutique_dataset()
elif "sock-shop-1" in args.dataset:
    download_sock_shop_1_dataset()
elif "sock-shop-2" in args.dataset or "re1-ss" in args.dataset:
    download_sock_shop_2_dataset()
elif "train-ticket" in args.dataset or "re1-tt" in args.dataset:
    download_train_ticket_dataset()
elif "re2" in args.dataset:
    download_re2_dataset()
elif "re3" in args.dataset:
    download_re3_dataset()
elif "torai" in args.dataset:
    pass  # torai data is expected to be local
else:
    raise Exception(f"{args.dataset} is not defined!")

DATASET_MAP = {
    "online-boutique": "data/online-boutique",
    "sock-shop-1": "data/sock-shop-1",
    "sock-shop-2": "data/sock-shop-2",
    "train-ticket": "data/train-ticket",
    "re1-ob": "data/online-boutique",
    "re1-ss": "data/sock-shop-2",
    "re1-tt": "data/train-ticket",
    "re2-ob": "data/RE2/RE2-OB",
    "re2-ss": "data/RE2/RE2-SS",
    "re2-tt": "data/RE2/RE2-TT",
    "re3-ob": "data/RE3/RE3-OB",
    "re3-ss": "data/RE3/RE3-SS",
    "re3-tt": "data/RE3/RE3-TT",
    "torai-ob": "data/torai-data/torai-OB",
    "torai-ss": "data/torai-data/torai-SS",
    "torai-tt": "data/torai-data/torai-TT",
}
dataset = DATASET_MAP[args.dataset]


# prepare input paths
data_paths = list(glob.glob(os.path.join(dataset, "**/data.csv"), recursive=True))
if not data_paths: 
    data_paths = list(glob.glob(os.path.join(dataset, "**/simple_metrics.csv"), recursive=True))
# new_data_paths = []
# for p in data_paths: 
#     if os.path.exists(p.replace("data.csv", "simple_data.csv")):
#         new_data_paths.append(p.replace("data.csv", "simple_data.csv"))
#     elif os.path.exists(p.replace("data.csv", "simple_metrics.csv")):
#         new_data_paths.append(p.replace("data.csv", "simple_metrics.csv"))
#     else:
#         new_data_paths.append(p)
# data_paths = new_data_paths
if args.test is True:
    data_paths = data_paths[:args.num_files]  # only test on num_files cases

# if not test make num_files = num of data paths
if args.test is False:
    args.num_files = len(data_paths)

# prepare output paths
from tempfile import TemporaryDirectory
# output_path = TemporaryDirectory().name
output_path = "output"
report_path = join(output_path, f"report.csv")
result_path = join(output_path, "results")
os.makedirs(result_path, exist_ok=True)

seed = args.seed
utils.set_seed(seed)

class Config:
    """Lightweight config container (like an empty struct)."""
    pass

def build_anomaly_config():
    config = Config()

    # ===== Basic Parameters =====
    config.win_size = 12                        # Window size
    config.DSR = 1                              # Downsampling rate
    config.cutfreq = 0                          # Cut frequency for FITS (0 = auto)

    if config.cutfreq == 0:
        config.cutfreq = int((config.win_size / config.DSR) / 2)

    assert (config.win_size / config.DSR) / 2 >= config.cutfreq, \
        'cutfreq should be smaller than half of the window size after downsampling'

    # ===== Sequence Parameters =====
    config.seq_len = config.win_size // config.DSR
    config.pred_len = 0                         # No prediction horizon for anomaly detection
    config.individual = False
    config.num_class = 1 #not used for anomaly detection
    config.task_name = "anomaly_detection"
    config.output_attention = True  # store_true equivalent (bool, not string)

    # ===== Embedding Parameters =====
    config.embed = "timeF"                      # Time features encoding: [timeF, fixed, learned]
    config.freq = "h"                           # Frequency (hourly)
    config.dropout = 0.1

    # ===== Model Architecture =====
    config.d_model = 128
    config.factor = 1
    config.n_heads = 8
    config.d_ff = 256
    config.activation = "gelu"
    config.e_layers = 2
    config.device = device
    #if itransformer for train_ticket, use smaller model
    #config.d_model = 64
    #config.factor = 1
    #config.n_heads = 4
    #config.d_ff = 256
    #config.activation = "gelu"
    #config.e_layers = 1
    #config.output_attention = True  # store_true equivalent (bool, not string)

    return config

def build_anomaly_config_smaller_network():
    config = Config()

    # ===== Basic Parameters =====
    config.win_size = 12                        # Window size
    config.DSR = 1                              # Downsampling rate
    config.cutfreq = 0                          # Cut frequency for FITS (0 = auto)

    if config.cutfreq == 0:
        config.cutfreq = int((config.win_size / config.DSR) / 2)

    assert (config.win_size / config.DSR) / 2 >= config.cutfreq, \
        'cutfreq should be smaller than half of the window size after downsampling'

    # ===== Sequence Parameters =====
    config.seq_len = config.win_size // config.DSR
    config.pred_len = 0                         # No prediction horizon for anomaly detection
    config.individual = False
    config.num_class = 1 #not used for anomaly detection
    config.task_name = "anomaly_detection"
    config.output_attention = True  # store_true equivalent (bool, not string)

    # ===== Embedding Parameters =====
    config.embed = "timeF"                      # Time features encoding: [timeF, fixed, learned]
    config.freq = "h"                           # Frequency (hourly)
    config.dropout = 0.1

    # ===== Model Architecture =====
    config.d_model = 64
    config.factor = 1
    config.n_heads = 8
    config.d_ff = 256
    config.activation = "gelu"
    config.e_layers = 1
    config.device = device
    #if itransformer for train_ticket, use smaller model
    #config.d_model = 64
    #config.factor = 1
    #config.n_heads = 4
    #config.d_ff = 256
    #config.activation = "gelu"
    #config.e_layers = 1
    #config.output_attention = True  # store_true equivalent (bool, not string)

    return config


def build_anomaly_config_iTrans_trainticket():
    config = Config()

    # ===== Basic Parameters =====
    config.win_size = 12                        # Window size
    config.DSR = 1                              # Downsampling rate
    config.cutfreq = 0                          # Cut frequency for FITS (0 = auto)

    if config.cutfreq == 0:
        config.cutfreq = int((config.win_size / config.DSR) / 2)

    assert (config.win_size / config.DSR) / 2 >= config.cutfreq, \
        'cutfreq should be smaller than half of the window size after downsampling'

    # ===== Sequence Parameters =====
    config.seq_len = config.win_size // config.DSR
    config.pred_len = 0                         # No prediction horizon for anomaly detection
    config.individual = False
    config.num_class = 1 #not used for anomaly detection
    config.task_name = "anomaly_detection"
    config.output_attention = True  # store_true equivalent (bool, not string)

    # ===== Embedding Parameters =====
    config.embed = "timeF"                      # Time features encoding: [timeF, fixed, learned]
    config.freq = "h"                           # Frequency (hourly)
    config.dropout = 0.1

    # ===== Model Architecture =====
    ##########config.d_model = 128
    ##########config.factor = 1
    ##########config.n_heads = 8
    ##########config.d_ff = 256
    ##########config.activation = "gelu"
    ##########config.e_layers = 2
    #if itransformer for train_ticket, use smaller model
    config.d_model = 64
    config.factor = 1
    config.n_heads = 4
    config.d_ff = 256
    config.activation = "gelu"
    config.e_layers = 1
    config.output_attention = True  # store_true equivalent (bool, not string)

    return config




def reciprocal_rank(ranks, answer):
    for idx, node in enumerate(ranks, start=1):
        if node == answer:
            return 1.0 / idx
    return 0.0


def hit_at_k(ranks, answer, k):
    return float(answer in ranks[:k])


def ndcg_at_k(ranks, answer, k):
    for idx, node in enumerate(ranks[:k], start=1):
        if node == answer:
            return 1.0 / np.log2(idx + 1)
    return 0.0


def rank_position(ranks, answer):
    for idx, node in enumerate(ranks, start=1):
        if node == answer:
            return idx
    return len(ranks) + 1


def summarize(values):
    values = np.asarray(values)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }

def process(data_path):
    run_args = argparse.Namespace()
    run_args.root_path = os.getcwd()
    run_args.data_path = data_path
    
    # convert length from minutes to seconds
    if args.length is None:
        args.length = 10
    data_length = args.length * 60 // 2

    data_dir = dirname(data_path)

    service, metric = basename(dirname(dirname(data_path))).split("_")
    case = basename(dirname(data_path))

    rp = join(result_path, f"{service}_{metric}_{case}.json")

    # == Load and Preprocess data ==
    data = pd.read_csv(data_path)
    
    # remove lat-50, only selecte lat-90 
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    
    if "mm-tt" in data_path or "torai-TT" in data_path:
        time_col = data["time"]
        data = data.loc[:, data.columns.str.startswith("ts-")]
        data["time"] = time_col
        
    # handle inf
    data = data.replace([np.inf, -np.inf], np.nan)

    # handle na
    data = data.fillna(method="ffill")
    data = data.fillna(0)

    with open(join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip()) + args.tdelta
    # for metrics, minutes -> seconds // 2
    normal_df = data[data["time"] < inject_time].tail(args.length * 60 // 2)
    anomal_df = data[data["time"] >= inject_time].head(args.length * 60 // 2)

    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    # num column, exclude time
    num_node = len(data.columns) - 1

    # rename latency
    data = data.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in data.columns
            if c.endswith("_latency-90")
        }
    )
    if args.with_logs   is True:
        print("$$$$ Including logs in the input...$$$ ")
        # add logs to data, under datapath/logts.csv
        log_path = join(data_dir, "logts.csv")
        if os.path.exists(log_path):
            log_data = pd.read_csv(log_path)
            new_data = {}
            new_data["metric"] = data
            new_data["logts"] = log_data
            data = new_data
    else:
        pass

    # == Get SLI ===
    sli = None
    if "my-sock-shop" in data_path or "fse-ss" in data_path:
        sli = "front-end_cpu"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service}_lat_90" in data:
            sli = f"{service}_lat_90"
    elif "train-ticket" in data_path or "fse-tt" in data_path or "RE2-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "online-boutique" in data_path or "fse-ob" in data_path or "RE2-OB" in data_path or "RE2-SS" in data_path:
        sli = "frontend_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
        elif "frontend_1" in data:
            sli = "frontend_1"
    elif "torai-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
    elif "torai-OB" in data_path or "torai-SS" in data_path:
        sli = "frontend_latency"
    else:
        raise ValueError("SLI not implemented")



     # == Multimodal data loading for torai ==
    if "torai" in args.dataset:
        logts = pd.read_csv(os.path.join(data_dir, "logts.csv"))

        traces_err = pd.DataFrame()
        traces_lat = pd.DataFrame()
        try:
            traces_err = pd.read_csv(os.path.join(data_dir, "tracets_err.csv"))
            traces_lat = pd.read_csv(os.path.join(data_dir, "tracets_lat.csv"))
        except FileNotFoundError:
            pass

        # window logts and trace ts to match metric window
        logts_length = args.length * 4 // 2 if args.length else 20
        a = logts[logts["time"] < inject_time].tail(logts_length)
        b = logts[logts["time"] >= inject_time].head(logts_length)
        logts = pd.concat([a, b], ignore_index=True)

        if traces_err.shape[0] > 0:
            a = traces_err[traces_err["time"] < inject_time].tail(logts_length)
            b = traces_err[traces_err["time"] >= inject_time].head(logts_length)
            traces_err = pd.concat([a, b], ignore_index=True)

        if traces_lat.shape[0] > 0:
            a = traces_lat[traces_lat["time"] < inject_time].tail(logts_length)
            b = traces_lat[traces_lat["time"] >= inject_time].head(logts_length)
            traces_lat = pd.concat([a, b], ignore_index=True)

        data = {
            "metric": data.copy(deep=True) if isinstance(data, pd.DataFrame) else data,
            "logts": logts,
            "tracets_err": traces_err,
            "tracets_lat": traces_lat,
        }

    # == PROCESS ==
    func = globals()[args.method].rcd if args.method == "rcd" else globals()[args.method]

    try:
        st = datetime.now()
        
        if args.method == "iTransformer" and args.dataset == "train-ticket":
            config = build_anomaly_config_iTrans_trainticket() #to use smaller model for train-ticket with iTransformer, as it is the only one that OOMs with the default config
        elif args.model_class == "GVAR":
            config = build_anomaly_config_smaller_network() #as GVAR is more memory intensive, we use a smaller network for all datasets to avoid OOM
        else:
            if args.model_class in ["iTransformer","TimeMixerpp","Dlinear","Fits","causalrca","Mamba_backbone","SFlexRCA","SFlexRCAmulti","Eadro","Art","Anofusion","CUTS_PLUS","GVAR"]:
                config = build_anomaly_config()

        #if args.method in ["SFlexRCA"]:
        #    OrthTransform(dataset_obj=None, save_path="output", time_lag=1, device=device)


        out = func(
            data,
            inject_time,
            dataset=args.dataset,
            anomalies=None,
            dk_select_useful=False,
            sli=sli,
            verbose=False,
            n_iter=num_node,
            with_baro_post=args.combine_baro_post,
            with_baro_pre=args.combine_baro_pre,
            model_class=args.model_class,
            model_config=config,
            ensemble_method=args.ensemble_method,
            args=run_args,
            scalar_type=args.scaler_type
        )

        train_time = out.get("train_time", 0.0)
        infer_time = out.get("infer_time", 0.0)

        num_params = out.get("num_params", 0)
        num_trainable_params = out.get("num_trainable_params", 0)
        model_size_mb = out.get("model_size_mb", 0.0)
        peak_memory_mb = out.get("peak_memory_mb", 0.0)
        energy_joules = out.get("energy_joules", 0.0)

        root_causes = out.get("ranks")
        # print("==============")
        # print(f"{data_path=}")
        # print(root_causes[:5])
        dump_json(filename=rp, data={0: root_causes})
        return {"train_time": train_time, "infer_time": infer_time, "num_params": num_params, "num_trainable_params": num_trainable_params, "model_size_mb": model_size_mb, "peak_memory_mb": peak_memory_mb, "energy_joules": energy_joules}
    except Exception as e:
        raise e
        print(f"{args.method=} failed on {data_path=}")
        print(e)
        rp = join(result_path, f"{service}_{metric}_{case}_failed.json")
        with open(rp, "w") as f:
            json.dump({"error": str(e)}, f)


start_time = datetime.now()

if args.test:
    print("=== Test mode ===")
    print(f"Number of cases: {len(data_paths)}")
    print("=================")
train_time_list = []
infer_time_list = []
num_params_list = []
num_trainable_params_list = []
model_size_mb_list = []
peak_memory_mb_list = []
energy_joules_list = []

for data_path in tqdm(sorted(data_paths)):
    #write the shape of data_path in a file named shape.txt under the same folder as data_path
    data = pd.read_csv(data_path)
    shape_path = "shape.csv"
    #name of file, shape 
    if not os.path.exists(shape_path):
        with open(shape_path, "w") as f:
            f.write("filename,shape\n")
            f.write(f"{data_path},{data.shape[0]},{data.shape[1]}\n")
    else:
        with open(shape_path, "a") as f:
            f.write(f"{data_path},{data.shape[0]},{data.shape[1]}\n")
    train_infer_times = process(data_path)
    train_time_list.append(train_infer_times["train_time"])
    infer_time_list.append(train_infer_times["infer_time"])
    num_params_list.append(train_infer_times["num_params"])
    num_trainable_params_list.append(train_infer_times["num_trainable_params"])
    model_size_mb_list.append(train_infer_times["model_size_mb"])
    peak_memory_mb_list.append(train_infer_times["peak_memory_mb"])
    energy_joules_list.append(train_infer_times["energy_joules"])
    # === SAFE MEMORY CLEANUP ===
    import gc
    import torch

    try:
        # clear Python garbage
        gc.collect()

        # clear PyTorch GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # delete local large variables explicitly (optional)
        del data_path  # optional, but good for large loops

    except Exception as e:
        print(f"Memory clear failed for {data_path}: {e}")
        
end_time = datetime.now()
time_taken = end_time - start_time
avg_speed = round(time_taken.total_seconds() / len(data_paths), 2)
avg_train_time = round(sum(train_time_list) / len(train_time_list), 2)
std_train_time = round(np.std(train_time_list), 2)
avg_infer_time = round(sum(infer_time_list) / len(infer_time_list), 2)
std_infer_time = round(np.std(infer_time_list), 2)
avg_num_params = round(sum(num_params_list) / len(num_params_list), 2)
std_num_params = round(np.std(num_params_list), 2)
avg_num_trainable_params = round(sum(num_trainable_params_list) / len(num_trainable_params_list), 2)
std_num_trainable_params = round(np.std(num_trainable_params_list) , 2)
avg_model_size_mb = round(sum(model_size_mb_list) / len(model_size_mb_list), 2)
std_model_size_mb = round(np.std(model_size_mb_list), 2)
avg_peak_memory_mb = round(sum(peak_memory_mb_list) / len(peak_memory_mb_list), 2)
std_peak_memory_mb = round(np.std(peak_memory_mb_list), 2)
avg_energy_joules = round(sum(energy_joules_list) / len(energy_joules_list), 2)
std_energy_joules = round(np.std(energy_joules_list), 2)
# ======== EVALUTION ===========
rps = glob.glob(join(result_path, "*.json"))
services = sorted(list(set([basename(x).split("_")[0] for x in rps])))
faults = sorted(list(set([basename(x).split("_")[1] for x in rps])))

eval_data = {
    "service-fault": [],
    "top_1_service": [],
    "top_3_service": [],
    "top_5_service": [],
    "avg@5_service": [],
    "top_1_metric": [],
    "top_3_metric": [],
    "top_5_metric": [],
    "avg@5_metric": [],
}

service_metrics = defaultdict(list)
metric_metrics = defaultdict(list)
# NEW
service_metrics_by_fault = defaultdict(lambda: defaultdict(list))
metric_metrics_by_fault = defaultdict(lambda: defaultdict(list))
s_evaluator_all = Evaluator()
f_evaluator_all = Evaluator()
s_evaluator_cpu = Evaluator()
f_evaluator_cpu = Evaluator()
s_evaluator_mem = Evaluator()
f_evaluator_mem = Evaluator()
s_evaluator_lat = Evaluator()
f_evaluator_lat = Evaluator()
s_evaluator_loss = Evaluator()
f_evaluator_loss = Evaluator()
s_evaluator_io = Evaluator()
f_evaluator_io = Evaluator()
s_evaluator_socket = Evaluator()
f_evaluator_socket = Evaluator()

for service in services:
    for fault in faults:
        s_evaluator = Evaluator()
        f_evaluator = Evaluator()

        for rp in rps:
            s, m = basename(rp).split("_")[:2]
            if s != service or m != fault:
                continue  # ignore

            data = load_json(rp)
            if "error" in data:
                continue  # ignore

            for i, ranks in data.items():
                s_ranks = [Node(x.split("_")[0].replace("-db", ""), "unknown") for x in ranks]
                # remove duplication
                old_s_ranks = s_ranks.copy()
                s_ranks = (
                    [old_s_ranks[0]]
                    + [
                        old_s_ranks[i]
                        for i in range(1, len(old_s_ranks))
                        if old_s_ranks[i] not in old_s_ranks[:i]
                    ]
                    if old_s_ranks
                    else []
                )

                # for loop instead of list comprehension to remove -db
                #f_ranks = []
                #for x in ranks:
                #    # case of _ in metric name
                #    parts = x.split("_")
                #    if len(parts) == 2:
                #        f_ranks.append(Node(parts[0], parts[1]))
                #    #case of no _ in metric name
                #    elif len(parts) == 1:
                #        f_ranks.append(Node(parts[0], "unknown"))
                #f_ranks = [Node(x.split("_")[0], x.split("_")[1]) for x in ranks]
                f_ranks = [
                    Node(x.split("_")[0], "_".join(x.split("_")[1:]) if len(x.split("_")) > 1 else "unknown")
                    for x in ranks
                ]
                s_evaluator.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                f_evaluator.add_case(ranks=f_ranks, answer=Node(service, fault))

                service_answer = Node(service, "unknown")
                metric_answer = Node(service, fault)

                # ==========================
                # SERVICE LEVEL METRICS
                # ==========================

                mrr = reciprocal_rank(s_ranks, service_answer)
                hit1 = hit_at_k(s_ranks, service_answer, 1)
                hit3 = hit_at_k(s_ranks, service_answer, 3)
                hit5 = hit_at_k(s_ranks, service_answer, 5)
                ndcg5 = ndcg_at_k(s_ranks, service_answer, 5)
                rank = rank_position(s_ranks, service_answer)

                service_metrics["MRR"].append(mrr)
                service_metrics["Hit@1"].append(hit1)
                service_metrics["Hit@3"].append(hit3)
                service_metrics["Hit@5"].append(hit5)
                service_metrics["NDCG@5"].append(ndcg5)
                service_metrics["Rank"].append(rank)

                # NEW
                service_metrics_by_fault[fault]["MRR"].append(mrr)
                service_metrics_by_fault[fault]["Hit@1"].append(hit1)
                service_metrics_by_fault[fault]["Hit@3"].append(hit3)
                service_metrics_by_fault[fault]["Hit@5"].append(hit5)
                service_metrics_by_fault[fault]["NDCG@5"].append(ndcg5)
                service_metrics_by_fault[fault]["Rank"].append(rank)

                # ==========================
                # METRIC LEVEL METRICS
                # ==========================

                mrr = reciprocal_rank(f_ranks, metric_answer)
                hit1 = hit_at_k(f_ranks, metric_answer, 1)
                hit3 = hit_at_k(f_ranks, metric_answer, 3)
                hit5 = hit_at_k(f_ranks, metric_answer, 5)
                ndcg5 = ndcg_at_k(f_ranks, metric_answer, 5)
                rank = rank_position(f_ranks, metric_answer)

                metric_metrics["MRR"].append(mrr)
                metric_metrics["Hit@1"].append(hit1)
                metric_metrics["Hit@3"].append(hit3)
                metric_metrics["Hit@5"].append(hit5)
                metric_metrics["NDCG@5"].append(ndcg5)
                metric_metrics["Rank"].append(rank)

                # NEW
                metric_metrics_by_fault[fault]["MRR"].append(mrr)
                metric_metrics_by_fault[fault]["Hit@1"].append(hit1)
                metric_metrics_by_fault[fault]["Hit@3"].append(hit3)
                metric_metrics_by_fault[fault]["Hit@5"].append(hit5)
                metric_metrics_by_fault[fault]["NDCG@5"].append(ndcg5)
                metric_metrics_by_fault[fault]["Rank"].append(rank)

                if fault == "cpu":
                    s_evaluator_cpu.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_cpu.add_case(ranks=f_ranks, answer=Node(service, fault))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, fault))

                elif fault == "mem":
                    s_evaluator_mem.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_mem.add_case(ranks=f_ranks, answer=Node(service, fault))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, fault))

                elif fault == "delay":
                    s_evaluator_lat.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_lat.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                elif fault == "loss":
                    s_evaluator_loss.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_loss.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                elif fault == "disk":
                    s_evaluator_io.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_io.add_case(ranks=f_ranks, answer=Node(service, "diskio"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "diskio"))
                elif fault == "socket":
                    s_evaluator_socket.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_socket.add_case(ranks=f_ranks, answer=Node(service, "socket"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "socket"))


        eval_data["service-fault"].append(f"{service}_{fault}")
        eval_data["top_1_service"].append(s_evaluator.accuracy(1))
        eval_data["top_3_service"].append(s_evaluator.accuracy(3))
        eval_data["top_5_service"].append(s_evaluator.accuracy(5))
        eval_data["avg@5_service"].append(s_evaluator.average(5))
        eval_data["top_1_metric"].append(f_evaluator.accuracy(1))
        eval_data["top_3_metric"].append(f_evaluator.accuracy(3))
        eval_data["top_5_metric"].append(f_evaluator.accuracy(5))
        eval_data["avg@5_metric"].append(f_evaluator.average(5))


print("--- Evaluation results ---")
result = {"cpu":"", "mem":"", "io":"", "socket":"", "delay":"", "loss":""}
for name, s_evaluator, f_evaluator in [
    ("cpu", s_evaluator_cpu, f_evaluator_cpu),
    ("mem", s_evaluator_mem, f_evaluator_mem),
    ("io", s_evaluator_io, f_evaluator_io),
    ("socket", s_evaluator_socket, f_evaluator_socket),
    ("delay", s_evaluator_lat, f_evaluator_lat),
    ("loss", s_evaluator_loss, f_evaluator_loss),
]:
    eval_data["service-fault"].append(f"overall_{name}")
    eval_data["top_1_service"].append(s_evaluator.accuracy(1))
    eval_data["top_3_service"].append(s_evaluator.accuracy(3))
    eval_data["top_5_service"].append(s_evaluator.accuracy(5))
    eval_data["avg@5_service"].append(s_evaluator.average(5))
    eval_data["top_1_metric"].append(f_evaluator.accuracy(1))
    eval_data["top_3_metric"].append(f_evaluator.accuracy(3))
    eval_data["top_5_metric"].append(f_evaluator.accuracy(5))
    eval_data["avg@5_metric"].append(f_evaluator.average(5))

    if name == "io":
        name = "disk"

    if s_evaluator.average(5) is not None:
        print( f"Avg@5-{name.upper()}:".ljust(12), round(s_evaluator.average(5), 2))
        # save all metrics to csv 
        # add dataset, method, length, speed to each row
        result[name] = f"{round(s_evaluator.average(5), 2)}"
         
result["method"] = args.method
result["dataset"] = args.dataset
result["length"] = args.length
result["tdelta"] = args.tdelta  
result["speed"] = avg_speed
result["with_baro_pre"] = args.combine_baro_pre
result["with_baro_post"] = args.combine_baro_post
result["seed"] = args.seed
result["model_class"] = args.model_class
result["scaler_type"] = args.scaler_type
result["train_time"] = avg_train_time
result["train_time_std"] = std_train_time
result["infer_time"] = avg_infer_time
result["infer_time_std"] = std_infer_time
result["num_params"] = avg_num_params
result["num_params_std"] = std_num_params
result["num_trainable_params"] = avg_num_trainable_params
result["num_trainable_params_std"] = std_num_trainable_params
result["model_size_mb"] = avg_model_size_mb
result["model_size_mb_std"] = std_model_size_mb
result["peak_memory_mb"] = avg_peak_memory_mb
result["peak_memory_mb_std"] = std_peak_memory_mb
result["energy_joules"] = avg_energy_joules
result["energy_joules_std"] = std_energy_joules
result["ensemble_method"] = args.ensemble_method
for metric_name, values in service_metrics.items():

    stats = summarize(values)

    result[f"service_{metric_name}_mean"] = stats["mean"]
    result[f"service_{metric_name}_std"] = stats["std"]
    result[f"service_{metric_name}_median"] = stats["median"]

for metric_name, values in metric_metrics.items():

    stats = summarize(values)

    result[f"metric_{metric_name}_mean"] = stats["mean"]
    result[f"metric_{metric_name}_std"] = stats["std"]
    result[f"metric_{metric_name}_median"] = stats["median"]
# ===============================
# Per-fault statistics
# ===============================

for fault_name, metrics in service_metrics_by_fault.items():
    for metric_name, values in metrics.items():

        stats = summarize(values)

        result[f"service_{fault_name}_{metric_name}_mean"] = stats["mean"]
        result[f"service_{fault_name}_{metric_name}_std"] = stats["std"]
        result[f"service_{fault_name}_{metric_name}_median"] = stats["median"]

for fault_name, metrics in metric_metrics_by_fault.items():
    for metric_name, values in metrics.items():

        stats = summarize(values)

        result[f"metric_{fault_name}_{metric_name}_mean"] = stats["mean"]
        result[f"metric_{fault_name}_{metric_name}_std"] = stats["std"]
        result[f"metric_{fault_name}_{metric_name}_median"] = stats["median"]
if args.research_question == "RQ_1":
    df = pd.DataFrame(result, index=[0])
    if os.path.exists(f"MASCOT/RQ_1_{args.dataset}.csv"):
        df_old = pd.read_csv(f"MASCOT/RQ_1_{args.dataset}.csv")
        df = pd.concat([df_old, df], ignore_index=True)
        df.to_csv(f"MASCOT/RQ_1_{args.dataset}.csv", index=False)
    else:
        df = pd.DataFrame(result, index=[0])    
        df.to_csv(f"MASCOT/RQ_1_{args.dataset}.csv", index=False)

elif args.research_question == "RQ_Ensemble":
    df = pd.DataFrame(result, index=[0])
    # append to report_RQ_Ensemble_<dataset>_<num_epochs>.txt.csv
    if os.path.exists(f"MASCOT/RQ_2_{args.dataset}.csv"):
        df_old = pd.read_csv(f"MASCOT/RQ_2_{args.dataset}.csv")
        df = pd.concat([df_old, df], ignore_index=True)
        df.to_csv(f"MASCOT/RQ_2_{args.dataset}.csv", index=False)
    else:
        df = pd.DataFrame(result, index=[0])    
        df.to_csv(f"MASCOT/RQ_2_{args.dataset}.csv", index=False)

elif args.research_question == "RQ_mitigation":
    df = pd.DataFrame(result, index=[0])
    # append to report_RQ_mitigation_<dataset>_<num_epochs>.txt.csv
    if os.path.exists(f"report_RQ_mitigation_{args.dataset}_mitigation.txt.csv"):
        df_old = pd.read_csv(f"report_RQ_mitigation_{args.dataset}_mitigation.txt.csv")
        df = pd.concat([df_old, df], ignore_index=True)
        df.to_csv(f"report_RQ_mitigation_{args.dataset}_mitigation.txt.csv", index=False)
    else:
        df = pd.DataFrame(result, index=[0])    
        df.to_csv(f"report_RQ_mitigation_{args.dataset}_mitigation.txt.csv", index=False)
print("---")
print("Avg speed:", avg_speed)

