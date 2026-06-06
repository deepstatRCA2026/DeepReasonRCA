import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# =========================
# LOAD DATA
# =========================
DATA_PATH = "MASCOT/RQ_1_re2-ob.csv"
df = pd.read_csv(DATA_PATH)

core_cols = ["metric_MRR_mean", "infer_time", "peak_memory_mb", "model_class"]
df = df.dropna(subset=core_cols)
df = df[
    (df["method"] == "RMDnet") |
    (df["method"] == "causalrca")
].copy()
#df.loc[df["method"] == "causalrca", "model_class"] = "causalrca"
# =========================
# AGGREGATION
# =========================
agg = df.groupby("model_class").agg(
    mrr_mean=("metric_MRR_mean", "mean"),
    mrr_std=("metric_MRR_mean", "std"),
    latency_mean=("infer_time", "mean"),
    latency_std=("infer_time", "std"),
    mem_mean=("peak_memory_mb", "mean"),
    mem_std=("peak_memory_mb", "std"),
).reset_index()

# sort by performance (stable ordering across both plots)
agg = agg.sort_values("mrr_mean", ascending=False)

methods = agg["model_class"].values
x = np.arange(len(methods))

your_method = "SFlexRCA_Robust2"


# =====================================================
# FIGURE 1: MRR vs Inference Time
# =====================================================
fig, ax1 = plt.subplots(figsize=(10, 5))

bars1 = ax1.bar(
    x - 0.2,
    agg["mrr_mean"],
    width=0.4,
    label="MRR",
    alpha=0.9,
    yerr=agg["mrr_std"],
    error_kw=dict(
        capsize=5,
        lw=1.5,
    )
)

ax1.grid(
    True,
    axis="y",
    linestyle="--",
    alpha=0.5
)
ax1.set_axisbelow(True)

ax1.set_ylabel("MRR")
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=45, ha="right")
ax1.set_title("Performance vs Latency Tradeoff")

ax2 = ax1.twinx()

bars2 = ax2.bar(
    x + 0.2,
    agg["latency_mean"],
    width=0.4,
    color="orange",
    alpha=0.6,
    label="Inference Time",
    yerr=agg["latency_std"],
    error_kw=dict(
        capsize=5,
        lw=1.5,
    )
)

ax2.set_ylabel("Inference Time (s)")
ax2.set_ylim(bottom=0)
# highlight
for i, m in enumerate(methods):
    if m == your_method:
        bars1[i].set_edgecolor("red")
        bars1[i].set_linewidth(2)
        bars2[i].set_edgecolor("red")
        bars2[i].set_linewidth(2)

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("Images/MASCOT/MRR_Latency_Tradeoff.pdf", dpi=300)
plt.show()


# =====================================================
# FIGURE 2: AC@5 (proxy) vs Peak Memory
# =====================================================

# compute AC@5 proxy (same idea you used earlier)
df["ac5_proxy"] = df[["cpu", "mem", "socket", "delay", "loss", "metric_MRR_mean"]].select_dtypes(include=[np.number]).mean(axis=1)

agg2 = df.groupby("model_class").agg(
    ac5_mean=("ac5_proxy", "mean"),
    ac5_std=("ac5_proxy", "std"),
    mem_mean=("peak_memory_mb", "mean"),
    mem_std=("peak_memory_mb", "std"),
).reset_index()

agg2 = agg2.set_index("model_class").loc[agg["model_class"]].reset_index()

fig, ax1 = plt.subplots(figsize=(10, 5))

bars1 = ax1.bar(
    x - 0.2,
    agg2["ac5_mean"],
    width=0.4,
    label="AC@5",
    alpha=0.9,
    yerr=agg2["ac5_std"],
    error_kw=dict(
        capsize=5,
        lw=1.5,
    )
)

ax1.grid(
    True,
    axis="y",
    linestyle="--",
    alpha=0.5
)
ax1.set_axisbelow(True)

ax1.set_ylabel("AC@5")
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=45, ha="right")
ax1.set_title("Effectiveness vs Memory Tradeoff")

ax2 = ax1.twinx()

bars2 = ax2.bar(
    x + 0.2,
    agg2["mem_mean"],
    width=0.4,
    color="green",
    alpha=0.6,
    label="Peak Memory (MB)",
    yerr=agg2["mem_std"],
    error_kw=dict(
        capsize=5,
        lw=1.5,
    )
)

ax2.set_ylabel("Peak Memory (MB)")

# highlight
for i, m in enumerate(methods):
    if m == your_method:
        bars1[i].set_edgecolor("red")
        bars1[i].set_linewidth(2)
        bars2[i].set_edgecolor("red")
        bars2[i].set_linewidth(2)

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("Images/MASCOT/AC5_Memory_Tradeoff.pdf", dpi=300)
plt.show()


#export the figs to csvs
agg.to_csv(
    "Images/MASCOT/Table_Efficiency_1.csv",
    float_format="%.3f"
)

agg2.to_csv(
    "Images/MASCOT/Table_Efficiency_2.csv",
    float_format="%.3f"
)