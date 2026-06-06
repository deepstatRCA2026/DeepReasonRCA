import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})


CSV_PATH = "MASCOT/RQ_2.csv"

fault_metrics = [
    "cpu",
    "mem",
    "socket",
    "delay",
    "loss",
    "disk"
]

stat_order = [
    "None",

    "Quantile",
    "MAD",
    "IQR",
    "ModifiedZ",
    "Standard",
    "Robust",
]

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(CSV_PATH)

# =====================================================
# DEFINE SCALING METHOD
# =====================================================

df["stat_method"] = np.where(
    df["with_baro_post"] == False,
    "None",
    df["scaler_type"]
)

df = df[df["stat_method"].isin(stat_order)].copy()

df["stat_method"] = pd.Categorical(
    df["stat_method"],
    categories=stat_order,
    ordered=True
)

# =====================================================
# METRICS
# =====================================================

df["AC@5"] = df[fault_metrics].mean(axis=1)

df["MRR"] = df["service_MRR_mean"]

# =====================================================
# AGGREGATE ACROSS ALL DATASETS / SEEDS
# =====================================================

grouped = (
    df.groupby("stat_method", observed=True)
      .agg(
          ac5_mean=("AC@5", "mean"),
          ac5_std=("AC@5", "std"),
          mrr_mean=("MRR", "mean"),
          mrr_std=("MRR", "std")
      )
      .reindex(stat_order)
)

mean_table = grouped[
    ["ac5_mean", "mrr_mean"]
].copy()

mean_table.columns = [
    "AC@5",
    "MRR"
]

std_table = grouped[
    ["ac5_std", "mrr_std"]
].copy()

std_table.columns = [
    "AC@5",
    "MRR"
]

# =====================================================
# HEATMAP
# =====================================================

def plot_heatmap(
    mean_table,
    std_table,
    title,
    save_path=None
):

    plt.figure(figsize=(5.5, 4.5))

    sns.heatmap(
        mean_table,
        cmap="YlGnBu",
        linewidths=0.5,
        annot=False,
        cbar_kws={"label": "Score"}
    )

    cmap = plt.get_cmap("YlGnBu")

    vmin = mean_table.min().min()
    vmax = mean_table.max().max()

    for i in range(mean_table.shape[0]):
        for j in range(mean_table.shape[1]):

            mean_val = mean_table.iloc[i, j]
            std_val = std_table.iloc[i, j]

            if pd.notna(mean_val):

                norm_val = (
                    (mean_val - vmin)
                    / (vmax - vmin + 1e-12)
                )

                r, g, b, _ = cmap(norm_val)

                luminance = (
                    0.299 * r +
                    0.587 * g +
                    0.114 * b
                )

                text_color = (
                    "black"
                    if luminance > 0.5
                    else "white"
                )

                # Mean
                plt.text(
                    j + 0.5,
                    i + 0.42,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color=text_color
                )

                # Std
                if pd.notna(std_val):
                    plt.text(
                        j + 0.5,
                        i + 0.78,
                        f"±{std_val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color
                    )

    plt.title(title, fontsize=14)

    plt.xlabel("")
    plt.ylabel("Scaling Method", fontsize=12)

    plt.xticks(
        fontsize=12,
        rotation=0
    )

    plt.yticks(
        fontsize=11,
        rotation=0
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

# =====================================================
# PLOT
# =====================================================

plot_heatmap(
    mean_table,
    std_table,
    title="",#Impact of Deviation-Based Scoring Module on RCA Performance
    save_path="Images/MASCOT/Heatmap_Scaling_Ablation.pdf"
)

# =====================================================
# EXPORT TABLES
# =====================================================

mean_table.to_csv(
    "Images/MASCOT/scaling_ablation_mean.csv",
    float_format="%.4f"
)

std_table.to_csv(
    "Images/MASCOT/scaling_ablation_std.csv",
    float_format="%.4f"
)