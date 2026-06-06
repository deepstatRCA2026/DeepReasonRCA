import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle
# =====================================================
# Configuration
# =====================================================

csvs = [
    "MASCOT/RQ_1_re1-ob.csv",
    "MASCOT/RQ_1_re1-ss.csv",
    "MASCOT/RQ_1_re2-ob.csv",
    "MASCOT/RQ_1_re2-ss.csv",
    "MASCOT/RQ_1_train-ticket.csv",
]

method_categories = {
    "Analytical / Scoring Baselines": {
        "e_diagnosis",
        "nsigma",
        "baro"
    },
    "DL-Based": {
        "Fits",
        "Dlinear",
        "iTransformer"
    },
    "Proposed": {
        "DeepReasonRCA_Robust",
        "DeepReasonRCA_Standard"
    },
}

methods = [
    "e_diagnosis",
    "nsigma",
    "baro",
    "Fits",
    "Dlinear",
    "iTransformer",
    "DeepReasonRCA",
    "DeepReasonRCA_Robust",
    "DeepReasonRCA_Standard",
]

fault_metrics = ["cpu", "mem", "socket", "delay", "loss", "disk"]

# =====================================================
# Load datasets
# =====================================================

all_data = []

for csv in tqdm(csvs):

    df = pd.read_csv(csv)
    df.loc[df["model_class"] == "SFlexRCA", "model_class"] = "DeepReasonRCA"
    # -------------------------------------------------
    # Normalize method names
    # -------------------------------------------------

    mask = (
        (df["method"] == "RMDnet")
        & (df["model_class"] == "DeepReasonRCA")
    )

    df.loc[mask, "method"] = (
        df.loc[mask, "model_class"]
        + "_"
        + df.loc[mask, "scaler_type"]
    )

    df.loc[df["method"] == "RMDnet", "method"] = \
        df.loc[df["method"] == "RMDnet", "model_class"]

    df = df[df["method"].isin(methods)].copy()

    df["method"] = pd.Categorical(
        df["method"],
        categories=methods,
        ordered=True
    )

    # =================================================
    # Task Effectiveness
    #
    # One value per seed
    # =================================================

    df["task_mean"] = df[fault_metrics].mean(axis=1)

    # =================================================
    # Robustness
    #
    # One value per seed
    # =================================================

    robustness_cols = [
        "service_cpu_MRR_mean",
        "service_mem_MRR_mean",
        "service_delay_MRR_mean",
        "service_disk_MRR_mean",
        "service_loss_MRR_mean",
    ]

    df["robustness_mean"] = df[robustness_cols].mean(axis=1)

    all_data.append(df)

# =====================================================
# Merge all datasets
# =====================================================

full_df = pd.concat(all_data, ignore_index=True)

# =====================================================
# Aggregate across seeds
# =====================================================

def build_mean_std_tables(metric_col):

    grouped = (
        full_df
        .groupby(
            ["method", "dataset"],
            observed=True
        )[metric_col]
        .agg(
            mean="mean",
            std=lambda x: x.std(ddof=1)
        )
        .reset_index()
    )

    mean_table = grouped.pivot(
        index="method",
        columns="dataset",
        values="mean"
    )

    std_table = grouped.pivot(
        index="method",
        columns="dataset",
        values="std"
    )

    return mean_table, std_table


# =====================================================
# Heatmap plotting
# =====================================================

def plot_heatmap(
    mean_table,
    std_table,
    title,
    cbar_label,
    save_name=None
):
    ax = plt.gca()

    plt.figure(figsize=(14, 8))
    mask = mean_table.isna()

    ax = sns.heatmap(
        mean_table,
        mask=mask,
        annot=False,
        cmap="YlGnBu",
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={
            "label": cbar_label,
            "shrink": 0.9
        }
    )
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color("black")
    cbar = ax.collections[0].colorbar
    cbar.set_label(
        cbar_label,
        fontsize=18,
        fontweight="bold"
    )
    cbar.ax.tick_params(
        labelsize=16
    )
    # ----------------------------------------
    # Grey hatched cells for missing values
    # ----------------------------------------

    for i in range(mean_table.shape[0]):
        for j in range(mean_table.shape[1]):

            if pd.isna(mean_table.iloc[i, j]):

                #rect = Rectangle(
                #    (j, i),
                #    1,
                #    1,
                #    facecolor="#F2F2F2",
                #    edgecolor="grey",
                #    hatch="///",
                #    linewidth=0.5,
                #    zorder=5
                #)

                rect = Rectangle(
                (j, i), 1, 1,
                fill=True,
                facecolor="#F2F2F2",     # light grey base
                hatch="///",             # diagonal pattern
                edgecolor="lightgray",
                linewidth=0.0
                 )
                
                ax.add_patch(rect)
    for y in [3, 6]:
        ax.plot(
            [-1.4, mean_table.shape[1]],
            [y, y],
            color="black",
            linewidth=3,
            clip_on=False,
            zorder=10
        )
    ax.text(
        -3.5,
        1.5,
        "Analytical \n Scoring Baselines",
        rotation=90,
        va="center",
        ha="center",
        fontsize=16,
        fontweight="bold"
    )

    ax.text(
        -3.5,
        4.5,
        "DL-Based",
        rotation=90,
        va="center",
        ha="center",
        fontsize=16,
        fontweight="bold"
    )

    ax.text(
        -3.5,
        7.0,
        "Proposed",
        rotation=90,
        va="center",
        ha="center",
        fontsize=16,
        fontweight="bold"
    )

    cmap = plt.get_cmap("YlGnBu")

    vmin = mean_table.min().min()
    vmax = mean_table.max().max()

    for i in range(mean_table.shape[0]):
        for j in range(mean_table.shape[1]):

            mean_val = mean_table.iloc[i, j]
            std_val = std_table.iloc[i, j]

            if pd.notna(mean_val):

                if pd.isna(std_val):
                    text = f"{mean_val:.3f}"
                else:
                    text = f"{mean_val:.3f}\n±{std_val:.3f}"

                norm_val = (mean_val - vmin) / (vmax - vmin + 1e-12)

                r, g, b, _ = cmap(norm_val)

                # perceived luminance
                luminance = 0.299 * r + 0.587 * g + 0.114 * b

                text_color = "black" if luminance > 0.5 else "white"

                # Mean (large)
                plt.text(
                    j + 0.5,
                    i + 0.42,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=18,
                    color=text_color,
                    fontweight="bold"
                )

                # Std (smaller)
                if pd.notna(std_val):
                    plt.text(
                        j + 0.5,
                        i + 0.8,
                        f"±{std_val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=15,
                        color=text_color
                    )
    # font size of xlabel and ylabel = 15
    plt.xlabel("", fontsize=18, fontweight="bold")    
    plt.ylabel("", fontsize=15)
    plt.title(title, fontsize=18, fontweight="bold")

    #fontsize of x-axis labels, y-axis labels = 15
    plt.yticks(
        rotation=0,
        fontsize=18
    )
    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=18
    )
    plt.subplots_adjust(left=0.18)
    plt.tight_layout()

    if save_name:
        plt.savefig(
            save_name,
            dpi=300,
            bbox_inches="tight"
        )

    #plt.show()


# =====================================================
# Task Effectiveness
# =====================================================

task_mean_table, task_std_table = build_mean_std_tables(
    "task_mean"
)

plot_heatmap(
    task_mean_table,
    task_std_table,
    title="AC@5 (Mean ± Std)",
    cbar_label="AC@5",
    save_name="Images/MASCOT/Heatmap_Task_Effectiveness_(SingleModality).pdf"
)

# =====================================================
# Robustness
# =====================================================

rob_mean_table, rob_std_table = build_mean_std_tables(
    "robustness_mean"
)

plot_heatmap(
    rob_mean_table,
    rob_std_table,
    title="MRR (Mean ± Std)",
    cbar_label="MRR",
    save_name="Images/MASCOT/Heatmap_Robustness_(SingleModality).pdf"
)

# =====================================================
# Export tables
# =====================================================

task_mean_table.to_csv(
    "Images/MASCOT/Table_Task_Mean.csv",
    float_format="%.3f"
)

task_std_table.to_csv(
    "Images/MASCOT/Table_Task_Std.csv",
    float_format="%.3f"
)

rob_mean_table.to_csv(
    "Images/MASCOT/Table_Robustness_Mean.csv",
    float_format="%.3f"
)

rob_std_table.to_csv(
    "Images/MASCOT/Table_Robustness_Std.csv",
    float_format="%.3f"
)

