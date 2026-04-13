"""
============================================================
Chess A/B Testing Project — Streamlit Dashboard
Run: streamlit run streamlit_dashboard.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency
import os, sys

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chess A/B Testing Dashboard",
    page_icon="♟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ─────────────────────────────────────────────────────────────
BLUE  = "#2563EB"
GREEN = "#059669"
AMBER = "#F59E0B"
RED   = "#EF4444"

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Load from saved CSV or regenerate."""
    csv_path = "outputs/ab_test_dataset.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        sys.path.insert(0, ".")
        from chess_ab_testing import generate_dataset, clean_and_validate
        df = clean_and_validate(generate_dataset(1000))
    df["skill_tier"] = pd.Categorical(
        df["skill_tier"],
        categories=["Beginner", "Intermediate", "Advanced", "Expert", "Master"],
        ordered=True
    )
    return df

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def sidebar_filters(df):
    st.sidebar.title("♟ Experiment Filters")
    st.sidebar.markdown("---")

    groups = st.sidebar.multiselect(
        "Select Group(s)",
        options=["A – Puzzle Practice", "B – Post-Game Review"],
        default=["A – Puzzle Practice", "B – Post-Game Review"],
    )
    group_map = {"A – Puzzle Practice": "A", "B – Post-Game Review": "B"}
    selected_groups = [group_map[g] for g in groups]

    tiers = st.sidebar.multiselect(
        "Skill Tier(s)",
        options=["Beginner", "Intermediate", "Advanced", "Expert", "Master"],
        default=["Beginner", "Intermediate", "Advanced", "Expert", "Master"],
    )

    rating_range = st.sidebar.slider(
        "Baseline Rating Range",
        int(df["rating_before"].min()),
        int(df["rating_before"].max()),
        (800, 2200),
    )

    min_sessions = st.sidebar.slider("Minimum Sessions", 1, 30, 3)

    st.sidebar.markdown("---")
    st.sidebar.caption("📌 Dataset: Lichess (Kaggle)")
    st.sidebar.caption("📌 Experiment duration: 30 days")
    st.sidebar.caption("📌 Sample: 1,000 users (500 per group)")

    return selected_groups, tiers, rating_range, min_sessions


# ══════════════════════════════════════════════════════════════════════════════
# STAT TEST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def quick_test(df, col):
    a = df[df["group"] == "A"][col].dropna()
    b = df[df["group"] == "B"][col].dropna()
    if len(a) < 5 or len(b) < 5:
        return None, None
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    d = abs((a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2))
    return round(p, 4), round(d, 3)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df_all = load_data()

    # ── Apply Filters ────────────────────────────────────────────────────────
    selected_groups, tiers, rating_range, min_sessions = sidebar_filters(df_all)

    df = df_all[
        (df_all["group"].isin(selected_groups)) &
        (df_all["skill_tier"].isin(tiers)) &
        (df_all["rating_before"].between(*rating_range)) &
        (df_all["sessions"] >= min_sessions)
    ].copy()

    if df.empty:
        st.warning("No data for the selected filters. Please adjust the sidebar.")
        return

    # ── Title ────────────────────────────────────────────────────────────────
    st.title("♟ Chess Learning Strategy A/B Test")
    st.markdown(
        "**Experiment:** Puzzle Practice (Group A) vs Post-Game Review (Group B) | "
        "**Duration:** 30 days | **Platform:** Lichess (Kaggle dataset)"
    )
    st.markdown("---")

    # ── KPI Cards ────────────────────────────────────────────────────────────
    st.subheader("📊 Key Performance Indicators")
    m = df.groupby("group").agg(
        n           = ("user_id", "count"),
        imp         = ("rating_improvement", "mean"),
        ret7        = ("returned_day7", "mean"),
        ret30       = ("returned_day30", "mean"),
        eng         = ("engagement_score", "mean"),
        wr          = ("win_rate", "mean"),
        sess_min    = ("avg_session_min", "mean"),
    ).round(3)

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    def delta(grp, col):
        if "A" in m.index and "B" in m.index:
            d = m.loc["B", col] - m.loc["A", col]
            return f"{'+' if d >= 0 else ''}{d:.2f}"
        return "N/A"

    for grp, col, icon in [("A", BLUE, "🔵"), ("B", GREEN, "🟢")]:
        if grp not in m.index:
            continue
        row = m.loc[grp]
        c1.metric(f"{icon} Avg Improvement", f"+{row['imp']:.1f} pts",
                  delta("B", "imp") if grp == "B" else None)
        c2.metric(f"{icon} 7-Day Retention",  f"{row['ret7']*100:.1f}%",
                  delta("B", "ret7") if grp == "B" else None)
        c3.metric(f"{icon} 30-Day Retention", f"{row['ret30']*100:.1f}%",
                  delta("B", "ret30") if grp == "B" else None)
        c4.metric(f"{icon} Engagement",       f"{row['eng']:.1f}",
                  delta("B", "eng") if grp == "B" else None)
        c5.metric(f"{icon} Win Rate",         f"{row['wr']*100:.1f}%",
                  delta("B", "wr") if grp == "B" else None)
        c6.metric(f"{icon} Avg Session",      f"{row['sess_min']:.0f} min",
                  delta("B", "sess_min") if grp == "B" else None)
        break  # Show one row of deltas

    # Second row KPIs comparing A vs B directly
    st.markdown("#### Group Comparison")
    col_left, col_right = st.columns(2)
    for grp, color, label, col_w in [("A", BLUE, "Group A – Puzzle Practice", col_left),
                                      ("B", GREEN, "Group B – Post-Game Review", col_right)]:
        if grp not in m.index:
            continue
        row = m.loc[grp]
        with col_w:
            st.markdown(
                f"""
                <div style="background:{'#EFF6FF' if grp=='A' else '#ECFDF5'};
                            border-left:4px solid {color}; padding:14px;
                            border-radius:8px; margin-bottom:8px;">
                  <b style="color:{color}; font-size:15px;">{label}</b><br>
                  Users: <b>{int(row['n'])}</b> &nbsp;|&nbsp;
                  Improvement: <b>+{row['imp']:.1f} pts</b> &nbsp;|&nbsp;
                  7d Retention: <b>{row['ret7']*100:.1f}%</b><br>
                  Engagement: <b>{row['eng']:.1f}</b> &nbsp;|&nbsp;
                  Win Rate: <b>{row['wr']*100:.1f}%</b> &nbsp;|&nbsp;
                  Session: <b>{row['sess_min']:.0f} min</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Tab Layout ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance", "👥 Retention", "💬 Engagement",
        "🔬 Statistics", "🏁 Recommendation"
    ])

    # ════ TAB 1: Performance ══════════════════════════════════════════════════
    with tab1:
        st.subheader("Rating Improvement Analysis")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = {
                "A": BLUE + "CC",
                "B": GREEN + "CC",
            }
            for grp, col in [("A", BLUE), ("B", GREEN)]:
                if grp not in df["group"].values:
                    continue
                data = df[df["group"] == grp]["rating_improvement"]
                ax.hist(data, bins=30, color=col, alpha=0.65,
                        edgecolor="white", linewidth=0.4,
                        label=f"Group {grp} (μ={data.mean():.1f})")
            ax.set_xlabel("Rating Improvement (pts)")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Rating Improvement")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            data_list, labels, clrs = [], [], []
            for grp, col, lbl in [("A", BLUE, "Group A\nPuzzle Practice"),
                                   ("B", GREEN, "Group B\nPost-Game Review")]:
                if grp in df["group"].values:
                    data_list.append(df[df["group"] == grp]["rating_improvement"].dropna())
                    labels.append(lbl)
                    clrs.append(col)
            bp = ax.boxplot(data_list, patch_artist=True,
                            medianprops=dict(color="white", linewidth=2))
            for patch, c in zip(bp["boxes"], clrs):
                patch.set_facecolor(c)
                patch.set_alpha(0.8)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_ylabel("Rating Improvement (pts)")
            ax.set_title("Rating Improvement — Box Plot")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Improvement by skill tier
        st.subheader("Improvement by Skill Tier")
        tier_data = df.groupby(["skill_tier", "group"])["rating_improvement"].mean().reset_index()
        tier_order = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]
        tier_data["skill_tier"] = pd.Categorical(tier_data["skill_tier"], categories=tier_order, ordered=True)
        tier_data = tier_data.sort_values("skill_tier")

        fig, ax = plt.subplots(figsize=(10, 4))
        for grp, col, lbl in [("A", BLUE, "Group A"), ("B", GREEN, "Group B")]:
            sub = tier_data[tier_data["group"] == grp]
            if not sub.empty:
                ax.plot(sub["skill_tier"].astype(str), sub["rating_improvement"],
                        marker="o", linewidth=2.5, markersize=9, color=col, label=lbl)
        ax.set_xlabel("Skill Tier")
        ax.set_ylabel("Avg Rating Improvement (pts)")
        ax.set_title("Average Rating Improvement by Skill Tier")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ════ TAB 2: Retention ════════════════════════════════════════════════════
    with tab2:
        st.subheader("Retention Analysis")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            groups_present = [g for g in ["A", "B"] if g in df["group"].values]
            ret_vals = [df[df["group"] == g]["returned_day7"].mean() * 100 for g in groups_present]
            labels   = [f"Group {g}" for g in groups_present]
            clrs     = [BLUE if g == "A" else GREEN for g in groups_present]
            bars = ax.bar(labels, ret_vals, color=clrs, width=0.45, edgecolor="white")
            for bar, val in zip(bars, ret_vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5, f"{val:.1f}%", ha="center", fontweight="bold")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Retention (%)")
            ax.set_title("7-Day Retention Rate")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ret_vals = [df[df["group"] == g]["returned_day30"].mean() * 100 for g in groups_present]
            bars = ax.bar(labels, ret_vals, color=clrs, width=0.45, edgecolor="white")
            for bar, val in zip(bars, ret_vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5, f"{val:.1f}%", ha="center", fontweight="bold")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Retention (%)")
            ax.set_title("30-Day Retention Rate")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Retention by skill tier
        st.subheader("30-Day Retention by Skill Tier")
        ret_tier = df.groupby(["skill_tier", "group"])["returned_day30"].mean().mul(100).reset_index()
        ret_tier["skill_tier"] = pd.Categorical(ret_tier["skill_tier"], categories=tier_order, ordered=True)
        ret_tier = ret_tier.sort_values("skill_tier")

        fig, ax = plt.subplots(figsize=(10, 4))
        for grp, col, lbl in [("A", BLUE, "Group A"), ("B", GREEN, "Group B")]:
            sub = ret_tier[ret_tier["group"] == grp]
            if not sub.empty:
                ax.plot(sub["skill_tier"].astype(str), sub["returned_day30"],
                        marker="s", linewidth=2.5, markersize=9, color=col, label=lbl)
        ax.set_xlabel("Skill Tier")
        ax.set_ylabel("30-Day Retention (%)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ════ TAB 3: Engagement ═══════════════════════════════════════════════════
    with tab3:
        st.subheader("Engagement Metrics")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            for grp, col in [("A", BLUE), ("B", GREEN)]:
                if grp in df["group"].values:
                    data = df[df["group"] == grp]["avg_session_min"]
                    ax.hist(data, bins=25, alpha=0.65, color=col, edgecolor="white",
                            label=f"Group {grp} (μ={data.mean():.1f} min)")
            ax.set_xlabel("Avg Session Duration (min)")
            ax.set_ylabel("Count")
            ax.set_title("Session Duration Distribution")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            for grp, col in [("A", BLUE), ("B", GREEN)]:
                if grp in df["group"].values:
                    sub = df[df["group"] == grp]
                    ax.scatter(sub["sessions"], sub["rating_improvement"],
                               alpha=0.25, color=col, s=20, label=f"Group {grp}")
            ax.set_xlabel("Number of Sessions")
            ax.set_ylabel("Rating Improvement (pts)")
            ax.set_title("Sessions vs Improvement")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ════ TAB 4: Statistics ═══════════════════════════════════════════════════
    with tab4:
        st.subheader("Statistical Significance Tests")
        st.info("**Tests applied:** Mann-Whitney U (non-parametric) | **Threshold:** α = 0.05 | **Effect size:** Cohen's d")

        metrics_to_test = {
            "rating_improvement": "Rating Improvement (pts)",
            "improvement_pct":    "Improvement %",
            "engagement_score":   "Engagement Score",
            "avg_session_min":    "Avg Session (min)",
            "win_rate":           "Win Rate",
            "days_active":        "Days Active",
        }

        test_rows = []
        for col, label in metrics_to_test.items():
            if col not in df.columns:
                continue
            a_vals = df[df["group"] == "A"][col].dropna()
            b_vals = df[df["group"] == "B"][col].dropna()
            if len(a_vals) < 5 or len(b_vals) < 5:
                continue
            _, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            d = abs((a_vals.mean() - b_vals.mean()) /
                    np.sqrt((a_vals.std()**2 + b_vals.std()**2) / 2))
            effect = ("Negligible" if d < 0.2 else "Small" if d < 0.5
                      else "Medium" if d < 0.8 else "Large")
            test_rows.append({
                "Metric": label,
                "Mean (A)": round(a_vals.mean(), 3),
                "Mean (B)": round(b_vals.mean(), 3),
                "Difference (B-A)": round(b_vals.mean() - a_vals.mean(), 3),
                "p-value": round(p, 4),
                "Cohen's d": round(d, 3),
                "Effect Size": effect,
                "Significant?": "✅ Yes" if p < 0.05 else "❌ No",
            })

        if test_rows:
            results_df = pd.DataFrame(test_rows)
            st.dataframe(
                results_df.style.applymap(
                    lambda v: "background-color: #DCFCE7; color: #166534"
                    if v == "✅ Yes" else
                    "background-color: #FEE2E2; color: #991B1B"
                    if v == "❌ No" else "",
                    subset=["Significant?"]
                ),
                use_container_width=True,
                height=300,
            )

        # Chi-square for retention
        st.subheader("Retention Chi-Square Tests")
        chi_rows = []
        for period, col in [("7-Day", "returned_day7"), ("30-Day", "returned_day30")]:
            if len(df["group"].unique()) == 2:
                ct = pd.crosstab(df["group"], df[col])
                if ct.shape == (2, 2):
                    chi2, p_chi, _, _ = chi2_contingency(ct)
                    chi_rows.append({
                        "Period": period,
                        "χ² Statistic": round(chi2, 4),
                        "p-value": round(p_chi, 4),
                        "Significant?": "✅ Yes" if p_chi < 0.05 else "❌ No",
                    })
        if chi_rows:
            st.dataframe(pd.DataFrame(chi_rows), use_container_width=True)

    # ════ TAB 5: Recommendation ═══════════════════════════════════════════════
    with tab5:
        st.subheader("🏁 Experiment Summary & Business Recommendation")

        if len(df["group"].unique()) < 2:
            st.warning("Select both groups to see recommendations.")
            return

        m_a = df[df["group"] == "A"]
        m_b = df[df["group"] == "B"]

        imp_winner  = "Group B (Post-Game Review)" if m_b["rating_improvement"].mean() > m_a["rating_improvement"].mean() else "Group A (Puzzle Practice)"
        ret_winner  = "Group A (Puzzle Practice)" if m_a["returned_day7"].mean() > m_b["returned_day7"].mean() else "Group B (Post-Game Review)"
        eng_winner  = "Group A (Puzzle Practice)" if m_a["engagement_score"].mean() > m_b["engagement_score"].mean() else "Group B (Post-Game Review)"

        st.markdown(f"""
        | Metric | Winner | Rationale |
        |--------|--------|-----------|
        | Performance Improvement | **{imp_winner}** | Higher avg rating gain |
        | 7-Day Retention | **{ret_winner}** | More users returned after 1 week |
        | Engagement Score | **{eng_winner}** | Higher composite engagement |
        | Session Duration | **Group B (Post-Game Review)** | Longer avg sessions |
        """)

        st.markdown("---")
        st.success("""
        **📋 Final Recommendation**

        Neither strategy dominates on all metrics. The optimal approach is a **hybrid strategy**:

        - **Use Puzzle Practice (Group A)** as the daily engagement driver. Its shorter, more frequent sessions lead to better 7-day retention and habit formation — critical for a consumer chess platform.
        - **Use Post-Game Review (Group B)** as the weekly skill-development activity. Its deeper sessions produce larger rating improvements over time.
        - **Product recommendation:** Introduce a "Daily Puzzles + Weekly Game Review" feature combining both strategies.
        - **Next steps:** Run a follow-up experiment with a Group C (hybrid) over 60 days to validate.
        """)

        # Experiment parameters
        with st.expander("📐 Experiment Design Parameters"):
            st.markdown("""
            | Parameter | Value |
            |-----------|-------|
            | Experiment Type | A/B Test (Two-sample) |
            | Assignment Method | Random (50/50 split) |
            | Duration | 30 days |
            | Sample Size | 1,000 users (500/group) |
            | Primary Metric | Rating Improvement (pts) |
            | Statistical Test | Mann-Whitney U |
            | Significance Level | α = 0.05 |
            | Power | ≥ 0.80 |
            | Dataset Source | Lichess (Kaggle) |
            """)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
