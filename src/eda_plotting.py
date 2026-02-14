import matplotlib.pyplot as plt
import seaborn as sns
import math

#######################################
# Correlation heatmap plotting function
def correlation_heatmap(data, columns, title):
    """Plots a correlation heatmap for the specified columns in the DataFrame."""
    f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data[columns].corr(method="pearson"), 
                linewidth=0.5, center=0, annot=True, square=True, vmin=-1, vmax=1,
                cmap = sns.diverging_palette(230, 20, as_cmap=True), annot_kws={"size": 8}, ax=ax)
    plt.title(title)
    plt.show()
#######################################


#######################################
#Swarm plot with percentages
def draw_swarm_with_percentages(ax, df, feature, target, labe, colr):
    # Compute percentages
    counts = df[feature].value_counts(normalize=True) * 100
    sns.swarmplot(
        x=df[feature],
        y=df[target],
        hue=df[feature],
        palette=colr,
        legend=False,
        ax=ax
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labe)
    ax.set_ylabel("Win percentage")
    ax.set_xlabel("")

    # Add percentage text
    ymax = df[target].max()
    for i, label in enumerate(labe):
        pct = counts.get(i, 0)
        ax.text(
            i,
            ymax + 3.5,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )
    ax.set_title(feature, fontweight="bold", fontsize=11, pad=12)

def swarm_grid(df, features, target, labels_dict, colors_dict, max_cols=3, figsize=(14, 4)):
    n = len(features)
    rows = math.ceil(n / max_cols)
    fig, axes = plt.subplots(rows, max_cols, figsize=(figsize[0], figsize[1] * rows))
    axes = axes.flatten()  # flatten for easy indexing

    for i, feat in enumerate(features):
        label = labels_dict[feat]
        color = colors_dict[feat]
        draw_swarm_with_percentages(axes[i], df, feat, target, label, color)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    plt.tight_layout()
    plt.show()
#######################################


#######################################
# Distribution plots
def draw_hist(ax, df, feature, title = None, color = "blue", binsize = 25, x_lim = (0, 100)):
    sns.histplot(
        df[feature],
        bins=binsize,
        kde=True,
        color=color,
        ax=ax
    )
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_title(title or feature, fontweight="bold", fontsize=11, pad=12)

def distribution_grid(df, features, colors_dict, bin_dict, limits_dict, max_cols=3, figsize=(14, 4)):
    n = len(features)
    rows = math.ceil(n / max_cols)
    fig, axes = plt.subplots(rows, max_cols, figsize=(figsize[0], figsize[1] * rows))
    axes = axes.flatten()  # flatten for easy indexing

    for i, feat in enumerate(features):
        draw_hist(axes[i], df, feat, f"Feature: {feat}", color=colors_dict[feat], binsize=bin_dict[feat], x_lim=limits_dict[feat])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    plt.tight_layout()
    plt.show()
#######################################


#######################################
# Boxplots
def draw_boxplot(ax, df, feature, color, x_lim=(0, 100)):
    sns.boxplot(
        x=df[feature],
        color=color,
        ax=ax
    )
    ax.set_xlabel(feature)
    ax.set_ylabel("Win Percentage")
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_title(f"Boxplot of {feature}", fontweight="bold", fontsize=11, pad=12)

def boxplot_grid(df, features, colors_dict, limits_dict, max_cols=3, figsize=(14, 4)):
    n = len(features)
    rows = math.ceil(n / max_cols)
    fig, axes = plt.subplots(rows, max_cols, figsize=(figsize[0], figsize[1] * rows))
    axes = axes.flatten()  # flatten for easy indexing

    for i, feat in enumerate(features):
        draw_boxplot(axes[i], df, feat, colors_dict[feat], x_lim=limits_dict[feat])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    plt.tight_layout()
    plt.show()
#######################################


#######################################
# Regression plots
def draw_regplot(ax, df, feature, target, color, x_lim=(0, 100), y_lim=(0, 100)):
    sns.regplot(
        x=df[feature],
        y=df[target],
        color=color,
        scatter_kws={"alpha": 0.6, "s": 25},
        line_kws={"linewidth": 2},
        ax=ax
    )
    ax.set_xlabel(feature)
    ax.set_ylabel("Win Percentage")
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_title(f"Regression of {target} on {feature}", fontweight="bold", fontsize=11, pad=12)


def regplot_grid(df, features, target, colors_dict, x_limits_dict, y_limits_dict, max_cols=3, figsize=(14, 4)):
    n = len(features)
    rows = math.ceil(n / max_cols)
    fig, axes = plt.subplots(rows, max_cols, figsize=(figsize[0], figsize[1] * rows))
    axes = axes.flatten()  # flatten for easy indexing

    for i, feat in enumerate(features):
        draw_regplot(axes[i], df, feat, target, colors_dict[feat], x_lim=x_limits_dict[feat], y_lim=y_limits_dict[target])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    plt.tight_layout()
    plt.show()
#######################################
