import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================================================
# Configuration
# =====================================================

csvs = [
    "MASCOT/RQ_1_torai-ob.csv",
    "MASCOT/RQ_1_torai-ss.csv"
]

method_groups = {
    "Analytical / Scoring Baselines": ["torai"],
    "DL-Based": ["Art", "Anofusion", "Eadro"],
    "Proposed": ["DeepReasonRCA"],
}

methods = [
    "torai",
    "Art",
    "Anofusion",
    "Eadro",
    "DeepReasonRCA",
]

fault_metrics = ["cpu", "mem", "socket", "delay", "loss", "disk"]
# =====================================================
# Style (publication-grade)
# =====================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})


# =====================================================
# Load datasets
# =====================================================

all_data = []

for csv in tqdm(csvs):

    df = pd.read_csv(csv)
    df.loc[df["model_class"] == "SFlexRCAmulti", "model_class"] = "DeepReasonRCA"
    # -------------------------------------------------
    # Normalize method names
    # -------------------------------------------------


    df.loc[df["method"] == "RMDnet_Multimodality", "method"] = \
        df.loc[df["method"] == "RMDnet_Multimodality", "model_class"]

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

    fig, ax = plt.subplots(figsize=(10, 4.8))

    sns.heatmap(
        mean_table,
        ax=ax,
        annot=False,
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={
            "label": cbar_label,
            "shrink": 0.9,
            "aspect": 25
        }
    )
    # =================================================
    # Border
    # =================================================
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color("black")

    # =================================================
    # Colorbar styling
    # =================================================
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label, fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=14)

    cmap = plt.get_cmap("YlGnBu")

    vmin = mean_table.min().min()
    vmax = mean_table.max().max()
    
    # =================================================
    # Category separators
    # =================================================

    sep1 = len(method_groups["Analytical / Scoring Baselines"])
    sep2 = sep1 + len(method_groups["DL-Based"])

    for y in [sep1, sep2]:
        ax.plot(
            [-0.5, mean_table.shape[1]],
            [y, y],
            color="black",
            linewidth=2.5,
            clip_on=False
        )

    # =================================================
    # Category labels
    # =================================================

    ax.text(
        -1.8,
        sep1 / 2,
        "Analytical\nScoring",
        rotation=90,
        va="center",
        ha="center",
        fontsize=14,
        fontweight="bold"
    )

    ax.text(
        -1.8,
        sep1 + len(method_groups["DL-Based"]) / 2,
        "DL-Based",
        rotation=90,
        va="center",
        ha="center",
        fontsize=14,
        fontweight="bold"
    )

    ax.text(
        -1.8,
        sep2 + len(method_groups["Proposed"]) / 2,
        "Proposed",
        rotation=90,
        va="center",
        ha="center",
        fontsize=14,
        fontweight="bold"
    )

    # =================================================
    # Cell annotations
    # =================================================

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
                    ax.text(
                        j + 0.5,
                        i + 0.8,
                        f"±{std_val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=13,
                        color=text_color
                    )
    # font size of xlabel and ylabel = 15
    ax.set_xlabel("", fontsize=15, fontweight="bold")
    ax.set_ylabel("", fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    if save_name:
        plt.savefig(
            save_name,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


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
    save_name="Images/MASCOT/Heatmap_Task_Effectiveness_(MultiModality).pdf"
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
    save_name="Images/MASCOT/Heatmap_Robustness_(MultiModality).pdf"
)

# =====================================================
# Export tables
# =====================================================

task_mean_table.to_csv(
    "Images/MASCOT/Table_Task_Mean (multi).csv",
    float_format="%.3f"
)

task_std_table.to_csv(
    "Images/MASCOT/Table_Task_Std (multi).csv",
    float_format="%.3f"
)

rob_mean_table.to_csv(
    "Images/MASCOT/Table_Robustness_Mean (multi).csv",
    float_format="%.3f"
)

rob_std_table.to_csv(
    "Images/MASCOT/Table_Robustness_Std (multi).csv",
    float_format="%.3f"
)

