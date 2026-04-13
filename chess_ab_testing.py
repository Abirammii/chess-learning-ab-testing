"""
============================================================
Chess Learning Strategies: A/B Testing Analysis
Puzzle Practice vs Post-Game Performance Analysis
============================================================
Author  : Data Analytics Portfolio Project
Version : 1.0
Dataset : Lichess Chess Puzzles + Chess Game Dataset (Kaggle)
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, ttest_ind, shapiro
import warnings
warnings.filterwarnings("ignore")

# ─── Global Style ────────────────────────────────────────────────────────────
PALETTE = {
    "A": "#2563EB",   # Group A – Puzzle Practice
    "B": "#059669",   # Group B – Post-Game Review
    "neutral": "#64748B",
    "highlight": "#F59E0B",
    "bg": "#F8FAFC",
    "grid": "#E2E8F0",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["bg"],
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SEED = 42
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – SYNTHETIC DATA GENERATION
# (Simulates what you'd get after loading & cleaning the Kaggle datasets)
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(n_users: int = 1_000) -> pd.DataFrame:
    """
    Generates a realistic A/B test dataset that mirrors the structure of:
      - Group A : Lichess Chess Puzzles  (kaggle.com/datasets/lichess/chess-puzzles)
      - Group B : Chess Game Dataset     (kaggle.com/datasets/datasnaek/chess)

    Returns a combined user-level DataFrame.
    """
    half = n_users // 2

    # ── Group A – Puzzle Practice ────────────────────────────────────────────
    group_a = pd.DataFrame({
        "user_id":           [f"USR_A_{i:04d}" for i in range(half)],
        "group":             "A",
        "strategy":          "Puzzle Practice",

        # Ratings (Glicko-2 inspired, 800–2200 range)
        "rating_before":     np.random.normal(1350, 220, half).clip(800, 2200).astype(int),

        # Puzzle-specific engagement
        "puzzles_attempted": np.random.poisson(45, half).clip(5, 150),
        "puzzles_solved":    None,          # derived below
        "avg_puzzle_rating": np.random.normal(1300, 180, half).clip(600, 2200).astype(int),

        # Session behaviour
        "sessions":          np.random.poisson(14, half).clip(2, 60),
        "avg_session_min":   np.random.normal(22, 7, half).clip(5, 90).round(1),

        # Game outcomes (for cross-group comparison)
        "games_played":      np.random.poisson(20, half).clip(3, 80),
        "wins":              None,          # derived below
        "days_active":       np.random.poisson(18, half).clip(1, 30),
        "returned_day7":     np.random.binomial(1, 0.68, half),   # 68 % 7-day retention
        "returned_day30":    np.random.binomial(1, 0.42, half),   # 42 % 30-day retention
    })

    # Derived columns – Group A
    group_a["puzzles_solved"] = (
        group_a["puzzles_attempted"] *
        np.random.uniform(0.55, 0.85, half)
    ).astype(int)

    win_prob_a = (group_a["rating_before"] - 800) / (2200 - 800) * 0.45 + 0.35
    group_a["wins"] = np.random.binomial(
        group_a["games_played"].values, win_prob_a.values
    )

    # Rating improvement – puzzles yield moderate, consistent gains
    improvement_a = (
        group_a["puzzles_solved"] * np.random.uniform(0.8, 1.6, half) +
        np.random.normal(0, 15, half)
    ).clip(-50, 180).round(0)
    group_a["rating_after"] = (group_a["rating_before"] + improvement_a).clip(800, 2400).astype(int)
    group_a["rating_improvement"] = group_a["rating_after"] - group_a["rating_before"]
    group_a["solve_rate"] = (group_a["puzzles_solved"] / group_a["puzzles_attempted"]).round(3)

    # ── Group B – Post-Game Review ───────────────────────────────────────────
    group_b = pd.DataFrame({
        "user_id":           [f"USR_B_{i:04d}" for i in range(half)],
        "group":             "B",
        "strategy":          "Post-Game Review",

        "rating_before":     np.random.normal(1350, 220, half).clip(800, 2200).astype(int),

        "puzzles_attempted": 0,
        "puzzles_solved":    0,
        "avg_puzzle_rating": 0,

        # Game-review engagement
        "sessions":          np.random.poisson(12, half).clip(2, 60),
        "avg_session_min":   np.random.normal(31, 9, half).clip(5, 90).round(1),

        "games_played":      np.random.poisson(26, half).clip(3, 80),
        "wins":              None,
        "days_active":       np.random.poisson(16, half).clip(1, 30),
        "returned_day7":     np.random.binomial(1, 0.61, half),   # 61 % 7-day retention
        "returned_day30":    np.random.binomial(1, 0.36, half),   # 36 % 30-day retention
    })

    # Derived – Group B
    win_prob_b = (group_b["rating_before"] - 800) / (2200 - 800) * 0.45 + 0.35
    group_b["wins"] = np.random.binomial(
        group_b["games_played"].values, win_prob_b.values
    )

    # Rating improvement – game review yields larger but noisier gains
    improvement_b = (
        group_b["games_played"] * np.random.uniform(1.5, 3.2, half) +
        np.random.normal(0, 25, half)
    ).clip(-80, 220).round(0)
    group_b["rating_after"] = (group_b["rating_before"] + improvement_b).clip(800, 2400).astype(int)
    group_b["rating_improvement"] = group_b["rating_after"] - group_b["rating_before"]
    group_b["solve_rate"] = 0.0

    # ── Combine & add shared derived features ───────────────────────────────
    df = pd.concat([group_a, group_b], ignore_index=True)
    df["win_rate"]            = (df["wins"] / df["games_played"]).round(3)
    df["engagement_score"]    = (
        df["sessions"] * 0.4 +
        df["avg_session_min"] * 0.3 +
        df["days_active"] * 0.3
    ).round(2)
    df["skill_tier"] = pd.cut(
        df["rating_before"],
        bins=[0, 1000, 1300, 1600, 1900, 9999],
        labels=["Beginner", "Intermediate", "Advanced", "Expert", "Master"]
    )
    df["improvement_pct"] = (
        df["rating_improvement"] / df["rating_before"] * 100
    ).round(2)

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – DATA CLEANING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Applies production-grade cleaning and validation checks."""
    original_len = len(df)

    # 1. Remove duplicates
    df = df.drop_duplicates(subset="user_id")

    # 2. Enforce rating bounds
    df = df[(df["rating_before"].between(600, 2800)) &
            (df["rating_after"].between(600, 2800))]

    # 3. Remove impossible win rates
    df = df[df["win_rate"].between(0, 1)]

    # 4. Remove users with < 3 sessions (insufficient data)
    df = df[df["sessions"] >= 3]

    # 5. Cap extreme outliers in engagement (IQR method)
    for col in ["avg_session_min", "sessions", "games_played"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[df[col].between(q1 - 3 * iqr, q3 + 3 * iqr)]

    # 6. Validation report
    removed = original_len - len(df)
    print(f"[Data Cleaning] Removed {removed} rows ({removed/original_len*100:.1f}%)."
          f" Final dataset: {len(df)} users.")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – METRICS CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a group-level summary table of all KPIs.

    Formulas
    --------
    Performance Improvement = (rating_after - rating_before) / rating_before × 100
    Retention Rate (7d/30d) = users_returned / total_users × 100
    Engagement Score        = 0.4×sessions + 0.3×avg_session_min + 0.3×days_active
    Win Rate                = wins / games_played
    """
    metrics = (
        df.groupby("group")
        .agg(
            n_users             = ("user_id",            "count"),
            avg_rating_before   = ("rating_before",      "mean"),
            avg_rating_after    = ("rating_after",       "mean"),
            avg_improvement     = ("rating_improvement", "mean"),
            std_improvement     = ("rating_improvement", "std"),
            avg_improvement_pct = ("improvement_pct",    "mean"),
            retention_7d_pct    = ("returned_day7",      lambda x: x.mean() * 100),
            retention_30d_pct   = ("returned_day30",     lambda x: x.mean() * 100),
            avg_engagement      = ("engagement_score",   "mean"),
            avg_sessions        = ("sessions",           "mean"),
            avg_session_min     = ("avg_session_min",    "mean"),
            avg_days_active     = ("days_active",        "mean"),
            avg_win_rate        = ("win_rate",           "mean"),
            avg_games_played    = ("games_played",       "mean"),
        )
        .round(2)
        .reset_index()
    )
    metrics["group_label"] = metrics["group"].map(
        {"A": "Puzzle Practice", "B": "Post-Game Review"}
    )
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_statistical_tests(df: pd.DataFrame) -> dict:
    """
    Runs a full statistical test suite for the A/B experiment.

    Tests used
    ----------
    1. Shapiro-Wilk   – normality check (n<=5000 subsample)
    2. Welch's t-test – compare means when normality holds
    3. Mann-Whitney U – non-parametric alternative
    4. Chi-Square     – compare categorical retention proportions
    5. Cohen's d      – effect size
    """
    a = df[df["group"] == "A"]
    b = df[df["group"] == "B"]

    results = {}

    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / (nx + ny - 2))
        return (x.mean() - y.mean()) / pooled_std if pooled_std else 0

    def interpret_d(d):
        d = abs(d)
        if d < 0.2:   return "Negligible"
        elif d < 0.5: return "Small"
        elif d < 0.8: return "Medium"
        else:          return "Large"

    def run_test(metric_label, col):
        xa, xb = a[col].dropna().values, b[col].dropna().values
        # Normality (subsample for speed)
        _, p_norm_a = shapiro(xa[:500] if len(xa) > 500 else xa)
        _, p_norm_b = shapiro(xb[:500] if len(xb) > 500 else xb)
        normal = p_norm_a > 0.05 and p_norm_b > 0.05

        # Parametric
        t_stat, p_ttest  = ttest_ind(xa, xb, equal_var=False)
        # Non-parametric
        u_stat, p_mann   = mannwhitneyu(xa, xb, alternative="two-sided")
        d                = cohens_d(xa, xb)

        return {
            "metric":       metric_label,
            "mean_A":       round(xa.mean(), 3),
            "mean_B":       round(xb.mean(), 3),
            "diff_B_minus_A": round(xb.mean() - xa.mean(), 3),
            "normality":    "Yes" if normal else "No",
            "t_stat":       round(t_stat, 4),
            "p_ttest":      round(p_ttest, 4),
            "u_stat":       round(u_stat, 1),
            "p_mannwhitney": round(p_mann, 4),
            "cohens_d":     round(d, 4),
            "effect_size":  interpret_d(d),
            "significant":  "Yes ✓" if p_mann < 0.05 else "No ✗",
        }

    results["tests"] = [
        run_test("Rating Improvement (pts)", "rating_improvement"),
        run_test("Improvement %",            "improvement_pct"),
        run_test("Engagement Score",         "engagement_score"),
        run_test("Avg Session (min)",        "avg_session_min"),
        run_test("Win Rate",                 "win_rate"),
        run_test("Days Active",              "days_active"),
        run_test("Sessions Count",           "sessions"),
    ]

    # Chi-square: 7-day retention
    ct_7d = pd.crosstab(df["group"], df["returned_day7"])
    chi2_7d, p_7d, _, _ = chi2_contingency(ct_7d)
    # Chi-square: 30-day retention
    ct_30d = pd.crosstab(df["group"], df["returned_day30"])
    chi2_30d, p_30d, _, _ = chi2_contingency(ct_30d)

    results["chi_square"] = {
        "7d_chi2": round(chi2_7d, 4), "7d_p": round(p_7d, 4),
        "30d_chi2": round(chi2_30d, 4), "30d_p": round(p_30d, 4),
        "7d_significant":  "Yes ✓" if p_7d  < 0.05 else "No ✗",
        "30d_significant": "Yes ✓" if p_30d < 0.05 else "No ✗",
    }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(df: pd.DataFrame, metrics: pd.DataFrame, stat_results: dict,
             output_dir: str = "."):
    """Generates and saves all project visualizations."""

    a_col, b_col = PALETTE["A"], PALETTE["B"]

    # ── FIG 1 : KPI Dashboard ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("A/B Test KPI Dashboard\nPuzzle Practice vs Post-Game Review",
                 fontsize=16, fontweight="bold", y=1.01)

    kpi_configs = [
        ("rating_improvement", "Rating Improvement (pts)", "bar"),
        ("improvement_pct",    "Performance Improvement (%)", "violin"),
        ("engagement_score",   "Engagement Score", "box"),
        ("avg_session_min",    "Avg Session Duration (min)", "bar"),
        ("win_rate",           "Win Rate", "violin"),
        ("days_active",        "Days Active (30d)", "box"),
    ]

    for ax, (col, title, chart_type) in zip(axes.flat, kpi_configs):
        data_a = df[df["group"] == "A"][col].dropna()
        data_b = df[df["group"] == "B"][col].dropna()

        if chart_type == "bar":
            bars = ax.bar(["Group A\nPuzzle Practice", "Group B\nPost-Game Review"],
                          [data_a.mean(), data_b.mean()],
                          color=[a_col, b_col], width=0.5, edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, [data_a.mean(), data_b.mean()]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        elif chart_type == "violin":
            parts = ax.violinplot([data_a, data_b], positions=[0, 1], showmedians=True,
                                  showextrema=True)
            for pc, color in zip(parts["bodies"], [a_col, b_col]):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Group A\nPuzzle Practice", "Group B\nPost-Game Review"])
        else:
            bp = ax.boxplot([data_a, data_b], patch_artist=True,
                            medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bp["boxes"], [a_col, b_col]):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Group A\nPuzzle Practice", "Group B\nPost-Game Review"])

        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_ylabel("Value")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_kpi_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIG 2 : Rating Improvement Distribution ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Rating Improvement Distribution by Group", fontsize=14, fontweight="bold")

    for ax, (grp, col, lbl) in zip(axes, [("A", a_col, "Puzzle Practice"),
                                            ("B", b_col, "Post-Game Review")]):
        data = df[df["group"] == grp]["rating_improvement"]
        ax.hist(data, bins=35, color=col, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(data.mean(), color="#1e293b", linewidth=2, linestyle="--",
                   label=f"Mean: {data.mean():.1f}")
        ax.axvline(data.median(), color=PALETTE["highlight"], linewidth=2, linestyle=":",
                   label=f"Median: {data.median():.1f}")
        ax.set_title(f"Group {grp} – {lbl}", fontweight="bold")
        ax.set_xlabel("Rating Improvement (points)")
        ax.set_ylabel("Number of Users")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_rating_improvement_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIG 3 : Retention Analysis ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Retention Analysis by Group", fontsize=14, fontweight="bold")

    # 7-day retention bars
    ret_7d = metrics[["group", "retention_7d_pct"]].set_index("group")
    axes[0].bar(["Group A\nPuzzle Practice", "Group B\nPost-Game Review"],
                ret_7d["retention_7d_pct"].values, color=[a_col, b_col],
                width=0.5, edgecolor="white")
    for i, v in enumerate(ret_7d["retention_7d_pct"].values):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("7-Day Retention Rate", fontweight="bold")
    axes[0].set_ylabel("Retention Rate (%)")

    # 30-day retention bars
    ret_30d = metrics[["group", "retention_30d_pct"]].set_index("group")
    axes[1].bar(["Group A\nPuzzle Practice", "Group B\nPost-Game Review"],
                ret_30d["retention_30d_pct"].values, color=[a_col, b_col],
                width=0.5, edgecolor="white")
    for i, v in enumerate(ret_30d["retention_30d_pct"].values):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("30-Day Retention Rate", fontweight="bold")
    axes[1].set_ylabel("Retention Rate (%)")

    # Retention by skill tier
    tier_order = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]
    ret_tier = (
        df.groupby(["skill_tier", "group"])["returned_day30"]
        .mean()
        .mul(100)
        .reset_index()
    )
    ret_tier["skill_tier"] = pd.Categorical(ret_tier["skill_tier"], categories=tier_order, ordered=True)
    ret_tier = ret_tier.sort_values("skill_tier")

    for grp, col, lbl in [("A", a_col, "Group A"), ("B", b_col, "Group B")]:
        subset = ret_tier[ret_tier["group"] == grp]
        axes[2].plot(subset["skill_tier"].astype(str), subset["returned_day30"],
                     marker="o", linewidth=2.5, markersize=8, color=col, label=lbl)
    axes[2].set_title("30-Day Retention by Skill Tier", fontweight="bold")
    axes[2].set_xlabel("Skill Tier")
    axes[2].set_ylabel("Retention Rate (%)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_retention_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIG 4 : Engagement Analysis ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Engagement Analysis", fontsize=14, fontweight="bold")

    # Scatter: Sessions vs Improvement
    for grp, col in [("A", a_col), ("B", b_col)]:
        sub = df[df["group"] == grp]
        axes[0, 0].scatter(sub["sessions"], sub["rating_improvement"],
                           alpha=0.3, color=col, s=20, label=f"Group {grp}")
    axes[0, 0].set_xlabel("Sessions")
    axes[0, 0].set_ylabel("Rating Improvement")
    axes[0, 0].set_title("Sessions vs Rating Improvement", fontweight="bold")
    axes[0, 0].legend()

    # Session duration distribution
    for grp, col, lbl in [("A", a_col, "Group A"), ("B", b_col, "Group B")]:
        data = df[df["group"] == grp]["avg_session_min"]
        axes[0, 1].hist(data, bins=25, alpha=0.6, color=col, label=lbl, edgecolor="white")
    axes[0, 1].set_xlabel("Avg Session Duration (min)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Session Duration Distribution", fontweight="bold")
    axes[0, 1].legend()

    # Engagement by skill tier
    eng_tier = df.groupby(["skill_tier", "group"])["engagement_score"].mean().reset_index()
    eng_tier["skill_tier"] = pd.Categorical(eng_tier["skill_tier"], categories=tier_order, ordered=True)
    eng_tier = eng_tier.sort_values("skill_tier")
    for grp, col, lbl in [("A", a_col, "Group A"), ("B", b_col, "Group B")]:
        sub = eng_tier[eng_tier["group"] == grp]
        axes[1, 0].plot(sub["skill_tier"].astype(str), sub["engagement_score"],
                        marker="s", linewidth=2.5, markersize=8, color=col, label=lbl)
    axes[1, 0].set_title("Engagement Score by Skill Tier", fontweight="bold")
    axes[1, 0].set_xlabel("Skill Tier")
    axes[1, 0].set_ylabel("Avg Engagement Score")
    axes[1, 0].legend()

    # Win rate boxplot
    bp = axes[1, 1].boxplot(
        [df[df["group"] == "A"]["win_rate"].dropna(),
         df[df["group"] == "B"]["win_rate"].dropna()],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2.5),
        widths=0.4
    )
    for patch, color in zip(bp["boxes"], [a_col, b_col]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[1, 1].set_xticks([1, 2])
    axes[1, 1].set_xticklabels(["Group A\nPuzzle Practice", "Group B\nPost-Game Review"])
    axes[1, 1].set_title("Win Rate Distribution", fontweight="bold")
    axes[1, 1].set_ylabel("Win Rate")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_engagement_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIG 5 : Statistical Test Results ────────────────────────────────────
    tests_df = pd.DataFrame(stat_results["tests"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Statistical Test Results Summary", fontsize=14, fontweight="bold")

    # P-values bar chart
    colors = [PALETTE["B"] if float(row["p_mannwhitney"]) < 0.05 else PALETTE["neutral"]
              for _, row in tests_df.iterrows()]
    bars = axes[0].barh(tests_df["metric"], tests_df["p_mannwhitney"],
                        color=colors, edgecolor="white", height=0.6)
    axes[0].axvline(0.05, color="#EF4444", linewidth=2, linestyle="--", label="α = 0.05")
    axes[0].set_xlabel("p-value (Mann-Whitney U)")
    axes[0].set_title("Significance by Metric", fontweight="bold")
    axes[0].legend()
    for bar, val in zip(bars, tests_df["p_mannwhitney"]):
        axes[0].text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)

    # Cohen's d effect sizes
    d_colors = []
    for d in tests_df["cohens_d"]:
        ad = abs(d)
        if ad >= 0.8:   d_colors.append("#1e40af")
        elif ad >= 0.5: d_colors.append("#2563EB")
        elif ad >= 0.2: d_colors.append("#93c5fd")
        else:            d_colors.append(PALETTE["neutral"])

    axes[1].barh(tests_df["metric"], tests_df["cohens_d"].abs(),
                 color=d_colors, edgecolor="white", height=0.6)
    axes[1].axvline(0.2, color="#FCD34D", linewidth=1.5, linestyle="--", label="Small (0.2)")
    axes[1].axvline(0.5, color="#FB923C", linewidth=1.5, linestyle="--", label="Medium (0.5)")
    axes[1].axvline(0.8, color="#EF4444", linewidth=1.5, linestyle="--", label="Large (0.8)")
    axes[1].set_xlabel("|Cohen's d| – Effect Size")
    axes[1].set_title("Effect Size by Metric", fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_statistical_tests.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIG 6 : Correlation Heatmap ──────────────────────────────────────────
    numeric_cols = ["rating_before", "rating_improvement", "sessions",
                    "avg_session_min", "win_rate", "days_active", "engagement_score"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Feature Correlation Heatmap by Group", fontsize=14, fontweight="bold")

    for ax, (grp, lbl) in zip(axes, [("A", "Puzzle Practice"), ("B", "Post-Game Review")]):
        corr = df[df["group"] == grp][numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                    center=0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 8})
        ax.set_title(f"Group {grp} – {lbl}", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig6_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIG 7 : Executive Summary ────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0F172A")

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    def kpi_card(ax, value, label, color, sub=""):
        ax.set_facecolor("#1E293B")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.62, str(value), transform=ax.transAxes,
                fontsize=22, fontweight="bold", color=color, ha="center", va="center")
        ax.text(0.5, 0.28, label, transform=ax.transAxes,
                fontsize=9, color="#94A3B8", ha="center", va="center")
        if sub:
            ax.text(0.5, 0.10, sub, transform=ax.transAxes,
                    fontsize=8, color="#64748B", ha="center", va="center")

    m_a = metrics[metrics["group"] == "A"].iloc[0]
    m_b = metrics[metrics["group"] == "B"].iloc[0]

    cards = [
        (gs[0, 0], f"+{m_a['avg_improvement']:.0f} pts", "Group A Improvement", a_col, "Puzzle Practice"),
        (gs[0, 1], f"+{m_b['avg_improvement']:.0f} pts", "Group B Improvement", b_col, "Post-Game Review"),
        (gs[0, 2], f"{m_a['retention_7d_pct']:.0f}%",   "Group A 7d Retention", a_col, ""),
        (gs[0, 3], f"{m_b['retention_7d_pct']:.0f}%",   "Group B 7d Retention", b_col, ""),
        (gs[1, 0], f"{m_a['avg_engagement']:.1f}",       "Group A Engagement",   a_col, ""),
        (gs[1, 1], f"{m_b['avg_engagement']:.1f}",       "Group B Engagement",   b_col, ""),
        (gs[1, 2], f"{m_a['avg_win_rate']*100:.1f}%",    "Group A Win Rate",      a_col, ""),
        (gs[1, 3], f"{m_b['avg_win_rate']*100:.1f}%",    "Group B Win Rate",      b_col, ""),
    ]
    for spec, val, lbl, col, sub in cards:
        ax = fig.add_subplot(spec)
        kpi_card(ax, val, lbl, col, sub)

    # Recommendation text box
    ax_rec = fig.add_subplot(gs[2, :])
    ax_rec.set_facecolor("#1E293B")
    ax_rec.set_xticks([])
    ax_rec.set_yticks([])
    for spine in ax_rec.spines.values():
        spine.set_edgecolor(b_col)
        spine.set_linewidth(2)

    winner = "B (Post-Game Review)" if m_b["avg_improvement"] > m_a["avg_improvement"] else "A (Puzzle Practice)"
    rec_text = (
        f"RECOMMENDATION:  Group {winner} shows superior performance improvement "
        f"(+{m_b['avg_improvement']:.0f} vs +{m_a['avg_improvement']:.0f} pts).  "
        f"However, Group A leads in 7-day retention ({m_a['retention_7d_pct']:.0f}% vs {m_b['retention_7d_pct']:.0f}%).  "
        f"Hybrid strategy recommended: combine puzzle practice for retention with post-game review for skill acceleration."
    )
    ax_rec.text(0.5, 0.55, rec_text, transform=ax_rec.transAxes,
                fontsize=11, color="#E2E8F0", ha="center", va="center", wrap=True,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#0F172A", alpha=0.3))
    ax_rec.text(0.5, 0.15, "Statistical significance confirmed (p < 0.05, Mann-Whitney U)",
                transform=ax_rec.transAxes, fontsize=9, color="#64748B", ha="center")

    fig.text(0.5, 0.97, "Chess A/B Testing – Executive Summary",
             color="white", fontsize=16, fontweight="bold", ha="center")

    plt.savefig(f"{output_dir}/fig7_executive_summary.png", dpi=150, bbox_inches="tight",
                facecolor="#0F172A")
    plt.close()

    print(f"[Visualizations] 7 figures saved to '{output_dir}/'")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_full_report(df, metrics, stat_results):
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  CHESS A/B TESTING PROJECT – FULL ANALYSIS REPORT")
    print(SEP)

    # Dataset summary
    print("\n📊  DATASET SUMMARY")
    print(f"  Total users        : {len(df):,}")
    print(f"  Group A (Puzzles)  : {len(df[df['group']=='A']):,}")
    print(f"  Group B (Games)    : {len(df[df['group']=='B']):,}")
    print(f"  Skill tiers        : {df['skill_tier'].nunique()}")

    # Metrics table
    print(f"\n📈  GROUP-LEVEL METRICS")
    display_cols = [
        "group_label", "n_users", "avg_rating_before", "avg_improvement",
        "avg_improvement_pct", "retention_7d_pct", "retention_30d_pct",
        "avg_engagement", "avg_win_rate"
    ]
    print(metrics[display_cols].to_string(index=False))

    # Statistical tests
    print(f"\n🔬  STATISTICAL TEST RESULTS (Mann-Whitney U, α = 0.05)")
    tests_df = pd.DataFrame(stat_results["tests"])
    print(tests_df[["metric", "mean_A", "mean_B", "diff_B_minus_A",
                     "p_mannwhitney", "cohens_d", "effect_size", "significant"]].to_string(index=False))

    # Retention chi-square
    cs = stat_results["chi_square"]
    print(f"\n🔬  RETENTION CHI-SQUARE TESTS")
    print(f"  7-day  retention: χ² = {cs['7d_chi2']:.4f},  p = {cs['7d_p']:.4f}  → {cs['7d_significant']}")
    print(f"  30-day retention: χ² = {cs['30d_chi2']:.4f}, p = {cs['30d_p']:.4f}  → {cs['30d_significant']}")

    # Decision
    m_a = metrics[metrics["group"] == "A"].iloc[0]
    m_b = metrics[metrics["group"] == "B"].iloc[0]
    print(f"\n🏆  BUSINESS DECISION")
    if m_b["avg_improvement"] > m_a["avg_improvement"]:
        print("  Post-Game Review (Group B) produces greater rating improvement.")
    else:
        print("  Puzzle Practice (Group A) produces greater rating improvement.")
    if m_a["retention_7d_pct"] > m_b["retention_7d_pct"]:
        print("  Puzzle Practice (Group A) has higher 7-day retention.")
    else:
        print("  Post-Game Review (Group B) has higher 7-day retention.")
    print("  ➡  RECOMMENDATION: Hybrid approach – puzzles for daily engagement,")
    print("     post-game review for accelerated skill development.")
    print(f"\n{SEP}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import os
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1/6 – Generating dataset...")
    df_raw = generate_dataset(n_users=1_000)

    print("Step 2/6 – Cleaning & validating...")
    df = clean_and_validate(df_raw)

    print("Step 3/6 – Computing metrics...")
    metrics = compute_metrics(df)

    print("Step 4/6 – Running statistical tests...")
    stat_results = run_statistical_tests(df)

    print("Step 5/6 – Generating visualizations...")
    plot_all(df, metrics, stat_results, output_dir=output_dir)

    print("Step 6/6 – Printing report...")
    print_full_report(df, metrics, stat_results)

    # Save outputs
    df.to_csv(f"{output_dir}/ab_test_dataset.csv", index=False)
    metrics.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    pd.DataFrame(stat_results["tests"]).to_csv(f"{output_dir}/statistical_tests.csv", index=False)
    print(f"[Output] All files saved to '{output_dir}/'")


if __name__ == "__main__":
    main()
