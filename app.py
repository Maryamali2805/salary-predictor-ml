"""
app.py  —  Data Science Salary Predictor
=========================================
A personal tool I built to help job seekers — especially fellow students
finishing an MSc Statistics degree — understand their market value before
entering the data science job market.

The app trains a Random Forest and an XGBoost model on synthetic data that
mirrors the Kaggle "Data Science Job Salaries" dataset, compares them, and
deploys the better one for interactive salary prediction.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_utils import (
    build_input_row,
    EXPERIENCE_LEVELS, EXPERIENCE_LABEL, EXPERIENCE_ORDER,
    EMPLOYMENT_TYPES, EMPLOYMENT_LABEL,
    JOB_CATEGORIES,
    REGIONS,
    COMPANY_SIZES, COMPANY_SIZE_LABEL,
    REMOTE_VALUES, REMOTE_LABEL,
    FEATURE_DISPLAY_NAMES,
)
from model_utils import train_and_evaluate, predict_with_confidence

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DS Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light custom CSS ───────────────────────────────────────────────────────────
# Kept minimal so it doesn't clash with Streamlit's own theming.
st.markdown("""
<style>
/* Larger, bolder hero salary number */
.salary-hero {
    font-size: 3.2rem;
    font-weight: 800;
    color: #1d6fa4;
    line-height: 1.1;
}
.salary-range {
    font-size: 1.2rem;
    color: #555;
    margin-top: 0.25rem;
}
/* Subtle card styling for metric blocks */
.metric-block {
    background: #f7f9fc;
    border-left: 4px solid #1d6fa4;
    border-radius: 4px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
}
/* Personal note banner */
.note-banner {
    background: #eef6ff;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    color: #1a3a52;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load / train models (cached so training runs only once per session) ────────
@st.cache_resource(show_spinner="Training models on the dataset… (first run only)")
def load_models():
    """
    st.cache_resource keeps the models in memory across reruns.
    We only retrain when the server restarts, not on every widget interaction.
    """
    return train_and_evaluate(force_retrain=False)


results = load_models()
rf_model       = results["rf_model"]
xgb_model      = results["xgb_model"]
best_name      = results["best_name"]
best_model     = results["best_model"]
metrics_df     = results["metrics_df"]
df_raw         = results["df_raw"]
rf_importance  = results["rf_importance"]
xgb_importance = results["xgb_importance"]
y_test         = results["y_test"]
rf_preds       = results["rf_test_preds"]
xgb_preds      = results["xgb_test_preds"]


# ── Helper: format salary ──────────────────────────────────────────────────────
def fmt(n: float) -> str:
    """Format a number as '$XXX,XXX'."""
    return f"${n:,.0f}"


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — user input form
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Your Profile")
    st.markdown(
        "Fill in your details below and switch to the **My Estimate** tab "
        "to see your predicted salary."
    )
    st.markdown("---")

    exp_choice = st.selectbox(
        "Experience level",
        options=EXPERIENCE_LEVELS,
        format_func=lambda x: EXPERIENCE_LABEL[x],
        index=0,   # default = Entry-Level (relevant for MSc graduates)
        help="EN = Entry (0–2 yrs), MI = Mid (2–5), SE = Senior (5–10), EX = Executive",
    )

    job_choice = st.selectbox(
        "Job category",
        options=JOB_CATEGORIES,
        index=0,
        help="Choose the role that best matches the position you're targeting.",
    )

    region_choice = st.selectbox(
        "Company location",
        options=REGIONS,
        index=0,
    )

    size_choice = st.selectbox(
        "Company size",
        options=COMPANY_SIZES,
        format_func=lambda x: COMPANY_SIZE_LABEL[x],
        index=1,   # default = Medium
    )

    remote_choice = st.selectbox(
        "Remote work",
        options=REMOTE_VALUES,
        format_func=lambda x: REMOTE_LABEL[x],
        index=2,   # default = Fully Remote (common for entry-level DS)
    )

    emp_choice = st.selectbox(
        "Employment type",
        options=EMPLOYMENT_TYPES,
        format_func=lambda x: EMPLOYMENT_LABEL[x],
        index=0,   # default = Full-Time
    )

    st.markdown("---")
    st.caption(
        "💡 Model: **" + best_name + "** (best on held-out test RMSE)\n\n"
        "Confidence intervals estimated from Random Forest tree variance."
    )


# ── Build the user's feature row once (shared across tabs) ────────────────────
user_row = build_input_row(
    experience_level=exp_choice,
    job_category=job_choice,
    company_region=region_choice,
    company_size=size_choice,
    remote_ratio=remote_choice,
    employment_type=emp_choice,
)

prediction = predict_with_confidence(rf_model, xgb_model, best_name, user_row)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("# 💼 Data Science Salary Predictor")
st.markdown("""
<div class="note-banner">
  <strong>Why I built this:</strong> As an MSc Statistics student about to enter the data
  science job market, I wanted a concrete, data-driven answer to "what salary should I
  expect?" — and I wanted to understand <em>why</em> certain profiles earn more. This tool
  uses a Random Forest and XGBoost model trained on data mirroring the Kaggle Data Science
  Jobs dataset to answer exactly that.
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_estimate, tab_models, tab_importance, tab_explore = st.tabs([
    "🎯 My Estimate",
    "🤖 Model Comparison",
    "📊 Feature Importance",
    "🔍 Explore the Data",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Salary Estimate
# ══════════════════════════════════════════════════════════════════════════════

with tab_estimate:
    col_pred, col_context = st.columns([1, 1], gap="large")

    with col_pred:
        st.markdown("### Your Estimated Salary")

        point = prediction["point_estimate"]
        lower = prediction["lower"]
        upper = prediction["upper"]

        # Large salary number
        st.markdown(
            f'<div class="salary-hero">{fmt(point)}<span style="font-size:1.2rem;color:#888;"> / year</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="salary-range">80% confidence range: <strong>{fmt(lower)}</strong> – <strong>{fmt(upper)}</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart — shows where the prediction sits between min and max
        salary_min = int(df_raw["salary_usd"].quantile(0.05))
        salary_max = int(df_raw["salary_usd"].quantile(0.95))

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=point,
            number={"prefix": "$", "valueformat": ",.0f"},
            delta={"reference": int(df_raw["salary_usd"].median()), "valueformat": ",.0f"},
            title={"text": f"vs. dataset median ({fmt(int(df_raw['salary_usd'].median()))})"},
            gauge={
                "axis": {"range": [salary_min, salary_max], "tickformat": "$,.0f"},
                "bar":  {"color": "#1d6fa4"},
                "steps": [
                    {"range": [salary_min, df_raw["salary_usd"].quantile(0.25)], "color": "#d4e6f5"},
                    {"range": [df_raw["salary_usd"].quantile(0.25), df_raw["salary_usd"].quantile(0.75)], "color": "#a8d0ec"},
                    {"range": [df_raw["salary_usd"].quantile(0.75), salary_max], "color": "#5ba8d4"},
                ],
                "threshold": {
                    "line": {"color": "orange", "width": 3},
                    "thickness": 0.75,
                    "value": int(df_raw["salary_usd"].median()),
                },
            },
        ))
        gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

        # Percentile message
        pct = int((df_raw["salary_usd"] < point).mean() * 100)
        if pct >= 75:
            emoji, msg = "🔥", f"Top {100 - pct}% of all profiles in the dataset"
        elif pct >= 50:
            emoji, msg = "📈", f"Above the median — top {100 - pct}% of profiles"
        elif pct >= 25:
            emoji, msg = "📊", f"Below the median — {pct}th percentile"
        else:
            emoji, msg = "🌱", f"Entry-stage range — {pct}th percentile (great starting point!)"
        st.info(f"{emoji} {msg}")

    with col_context:
        st.markdown("### Model Breakdown")

        # Side-by-side RF vs XGBoost estimates
        model_cols = st.columns(2)
        with model_cols[0]:
            st.metric(
                "Random Forest",
                fmt(prediction["rf_estimate"]),
                delta=fmt(prediction["rf_estimate"] - int(df_raw["salary_usd"].median()))
                      + " vs median",
                delta_color="normal",
            )
        with model_cols[1]:
            st.metric(
                "XGBoost",
                fmt(prediction["xgb_estimate"]),
                delta=fmt(prediction["xgb_estimate"] - int(df_raw["salary_usd"].median()))
                      + " vs median",
                delta_color="normal",
            )

        st.markdown(f"**Best model used:** {best_name}")
        st.markdown("<br>", unsafe_allow_html=True)

        # Profile summary table
        st.markdown("### Your Profile Summary")
        profile_data = {
            "Setting": [
                "Experience", "Job Category", "Company Region",
                "Company Size", "Remote Work", "Employment Type",
            ],
            "Value": [
                EXPERIENCE_LABEL[exp_choice],
                job_choice,
                region_choice,
                COMPANY_SIZE_LABEL[size_choice],
                REMOTE_LABEL[remote_choice],
                EMPLOYMENT_LABEL[emp_choice],
            ],
        }
        st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)

        # Confidence interval bar
        st.markdown("### 80% Confidence Interval")
        ci_fig = go.Figure()
        ci_fig.add_trace(go.Scatter(
            x=[lower, upper],
            y=["Salary Range", "Salary Range"],
            mode="lines",
            line=dict(color="#1d6fa4", width=6),
            name="80% CI",
        ))
        ci_fig.add_trace(go.Scatter(
            x=[point],
            y=["Salary Range"],
            mode="markers",
            marker=dict(color="#e05c1a", size=14, symbol="diamond"),
            name="Point Estimate",
        ))
        ci_fig.update_layout(
            height=130,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(tickformat="$,.0f"),
            showlegend=True,
            legend=dict(orientation="h", y=-0.6),
        )
        st.plotly_chart(ci_fig, use_container_width=True)
        st.caption(
            "The 80% confidence range is derived from the spread of "
            "individual Random Forest tree predictions for your profile. "
            "Narrower range = higher confidence."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab_models:
    st.markdown("### Random Forest vs XGBoost — Test Set Performance")
    st.markdown(
        "Both models were trained on 80% of the dataset and evaluated on the "
        "remaining 20%. Salaries are log-transformed during training (then "
        "exponentiated back) to handle the right-skewed distribution."
    )

    # Metrics table
    display_metrics = metrics_df.copy()
    for col in ["RMSE ($)", "MAE ($)"]:
        display_metrics[col] = display_metrics[col].apply(lambda x: f"${x:,}")

    # Highlight the winning model row
    def highlight_best(row):
        color = "#d6eaf8" if row["Model"] == best_name else ""
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        display_metrics.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(f"✅ **Winner: {best_name}** (lower RMSE on test set)")

    st.markdown("---")

    # Predicted vs Actual scatter for both models
    col_rf, col_xgb = st.columns(2)

    def pred_vs_actual_chart(y_true, y_pred, title, color):
        """Scatter plot of predicted vs actual salaries with a 45° ideal line."""
        fig = go.Figure()
        # Perfect prediction reference line
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            name="Perfect prediction",
        ))
        # Actual scatter
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred,
            mode="markers",
            marker=dict(color=color, opacity=0.4, size=5),
            name="Test samples",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Actual Salary ($)",
            yaxis_title="Predicted Salary ($)",
            height=380,
            xaxis=dict(tickformat="$,.0f"),
            yaxis=dict(tickformat="$,.0f"),
            margin=dict(t=40, b=40, l=60, r=20),
        )
        return fig

    with col_rf:
        st.plotly_chart(
            pred_vs_actual_chart(y_test.values, rf_preds, "Random Forest — Predicted vs Actual", "#1d6fa4"),
            use_container_width=True,
        )

    with col_xgb:
        st.plotly_chart(
            pred_vs_actual_chart(y_test.values, xgb_preds, "XGBoost — Predicted vs Actual", "#e07b39"),
            use_container_width=True,
        )

    # Residual distributions
    st.markdown("### Residual Distributions (Prediction Error)")
    st.caption(
        "A tight, zero-centred residual distribution means the model doesn't "
        "systematically over- or under-predict."
    )

    rf_residuals  = rf_preds  - y_test.values
    xgb_residuals = xgb_preds - y_test.values

    resid_fig = go.Figure()
    resid_fig.add_trace(go.Histogram(
        x=rf_residuals,
        name="Random Forest",
        opacity=0.65,
        marker_color="#1d6fa4",
        nbinsx=50,
    ))
    resid_fig.add_trace(go.Histogram(
        x=xgb_residuals,
        name="XGBoost",
        opacity=0.65,
        marker_color="#e07b39",
        nbinsx=50,
    ))
    resid_fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Zero error")
    resid_fig.update_layout(
        barmode="overlay",
        xaxis_title="Residual ($)",
        yaxis_title="Count",
        height=340,
        xaxis=dict(tickformat="$,.0f"),
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(resid_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

with tab_importance:
    st.markdown("### What Drives Data Science Salaries?")
    st.markdown(
        "Feature importances measure how much each input variable reduces "
        "prediction error across all splits in the ensemble. Higher = stronger "
        "driver of salary variation."
    )

    def importance_chart(imp_df, title, color):
        """Horizontal bar chart of feature importances with display names."""
        # Map internal feature names to human-readable labels
        imp_df = imp_df.copy()
        imp_df["Label"] = imp_df["Feature"].map(FEATURE_DISPLAY_NAMES)
        imp_df["Pct"]   = imp_df["Importance"] / imp_df["Importance"].sum() * 100
        imp_df = imp_df.sort_values("Importance")   # ascending so top is highest in chart

        fig = go.Figure(go.Bar(
            x=imp_df["Pct"],
            y=imp_df["Label"],
            orientation="h",
            marker_color=color,
            text=imp_df["Pct"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Importance (%)",
            xaxis=dict(range=[0, imp_df["Pct"].max() * 1.25]),
            height=420,
            margin=dict(t=50, b=40, l=180, r=80),
        )
        return fig

    col_imp1, col_imp2 = st.columns(2)
    with col_imp1:
        st.plotly_chart(
            importance_chart(rf_importance,  "Random Forest Feature Importance", "#1d6fa4"),
            use_container_width=True,
        )
    with col_imp2:
        st.plotly_chart(
            importance_chart(xgb_importance, "XGBoost Feature Importance", "#e07b39"),
            use_container_width=True,
        )

    # Side-by-side comparison bar chart
    st.markdown("### Side-by-Side Comparison")
    merged = rf_importance.merge(
        xgb_importance, on="Feature", suffixes=("_rf", "_xgb")
    )
    merged["Label"]   = merged["Feature"].map(FEATURE_DISPLAY_NAMES)
    merged["RF (%)"]  = merged["Importance_rf"]  / merged["Importance_rf"].sum()  * 100
    merged["XGB (%)"] = merged["Importance_xgb"] / merged["Importance_xgb"].sum() * 100
    merged = merged.sort_values("RF (%)", ascending=False)

    compare_fig = go.Figure()
    compare_fig.add_trace(go.Bar(
        name="Random Forest",
        x=merged["Label"],
        y=merged["RF (%)"],
        marker_color="#1d6fa4",
    ))
    compare_fig.add_trace(go.Bar(
        name="XGBoost",
        x=merged["Label"],
        y=merged["XGB (%)"],
        marker_color="#e07b39",
    ))
    compare_fig.update_layout(
        barmode="group",
        yaxis_title="Importance (%)",
        height=400,
        xaxis_tickangle=-30,
        margin=dict(t=20, b=100),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(compare_fig, use_container_width=True)

    # Insight callout
    top_feature = FEATURE_DISPLAY_NAMES[rf_importance.iloc[0]["Feature"]]
    second_feature = FEATURE_DISPLAY_NAMES[rf_importance.iloc[1]["Feature"]]
    st.info(
        f"**Key takeaway:** According to Random Forest, **{top_feature}** is "
        f"the single biggest driver of salary, followed by **{second_feature}**. "
        f"This aligns with what we know about the data science job market — "
        f"where you work and how senior you are matter far more than other factors."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Explore the Data
# ══════════════════════════════════════════════════════════════════════════════

with tab_explore:
    st.markdown("### Dataset Overview")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Total records",  f"{len(df_raw):,}")
    col_s2.metric("Median salary",  fmt(int(df_raw["salary_usd"].median())))
    col_s3.metric("Mean salary",    fmt(int(df_raw["salary_usd"].mean())))
    col_s4.metric("Salary std dev", fmt(int(df_raw["salary_usd"].std())))

    st.markdown("---")

    # ── Salary distribution overall ──────────────────────────────────────────
    st.markdown("#### Salary Distribution")
    dist_fig = px.histogram(
        df_raw,
        x="salary_usd",
        nbins=60,
        color_discrete_sequence=["#1d6fa4"],
        labels={"salary_usd": "Annual Salary (USD)"},
    )
    dist_fig.add_vline(
        x=df_raw["salary_usd"].median(),
        line_dash="dash", line_color="orange",
        annotation_text=f"Median: {fmt(int(df_raw['salary_usd'].median()))}",
        annotation_position="top right",
    )
    dist_fig.update_layout(
        xaxis_tickformat="$,.0f",
        height=320,
        margin=dict(t=20, b=40),
        yaxis_title="Count",
    )
    st.plotly_chart(dist_fig, use_container_width=True)

    # ── Box plots by categorical variables ───────────────────────────────────
    st.markdown("#### Salary by Category")

    box_choice = st.radio(
        "Break down by:",
        ["Experience Level", "Job Category", "Company Region", "Company Size", "Remote Work"],
        horizontal=True,
    )

    col_map = {
        "Experience Level": ("experience_level", EXPERIENCE_LABEL,
                             ["EN", "MI", "SE", "EX"]),
        "Job Category":     ("job_category",     None, JOB_CATEGORIES),
        "Company Region":   ("company_region",   None, REGIONS),
        "Company Size":     ("company_size",     COMPANY_SIZE_LABEL, ["S", "M", "L"]),
        "Remote Work":      ("remote_ratio",     {str(k): v for k, v in REMOTE_LABEL.items()},
                             [0, 50, 100]),
    }

    col_name, label_map, order = col_map[box_choice]

    box_df = df_raw.copy()
    if label_map:
        box_df[col_name] = box_df[col_name].astype(str).map(
            {str(k): v for k, v in label_map.items()}
        )
        order_labels = [label_map[str(k)] for k in order]
    else:
        order_labels = order

    box_fig = px.box(
        box_df,
        x=col_name,
        y="salary_usd",
        color=col_name,
        category_orders={col_name: order_labels},
        labels={"salary_usd": "Annual Salary (USD)", col_name: box_choice},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    box_fig.update_layout(
        yaxis_tickformat="$,.0f",
        height=420,
        margin=dict(t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(box_fig, use_container_width=True)

    # ── Heatmap: median salary by experience × region ────────────────────────
    st.markdown("#### Median Salary Heatmap — Experience × Region")
    pivot = (
        df_raw
        .groupby(["experience_level", "company_region"])["salary_usd"]
        .median()
        .reset_index()
        .pivot(index="experience_level", columns="company_region", values="salary_usd")
        .reindex(["EN", "MI", "SE", "EX"])
    )

    heat_fig = px.imshow(
        pivot,
        text_auto="$,.0f",
        color_continuous_scale="Blues",
        labels={"color": "Median Salary ($)"},
        aspect="auto",
    )
    heat_fig.update_layout(
        xaxis_title="Company Region",
        yaxis_title="Experience Level",
        height=340,
        margin=dict(t=20, b=40),
        coloraxis_colorbar=dict(tickformat="$,.0f"),
    )
    st.plotly_chart(heat_fig, use_container_width=True)

    # ── Top-paying job categories table ──────────────────────────────────────
    st.markdown("#### Median Salary by Job Category")
    cat_stats = (
        df_raw.groupby("job_category")["salary_usd"]
        .agg(["median", "mean", "std", "count"])
        .rename(columns={"median": "Median ($)", "mean": "Mean ($)",
                         "std": "Std Dev ($)", "count": "# Records"})
        .sort_values("Median ($)", ascending=False)
        .reset_index()
        .rename(columns={"job_category": "Job Category"})
    )
    for c in ["Median ($)", "Mean ($)", "Std Dev ($)"]:
        cat_stats[c] = cat_stats[c].apply(lambda x: f"${x:,.0f}")
    st.dataframe(cat_stats, use_container_width=True, hide_index=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Streamlit · Random Forest + XGBoost · "
    "Data mirrors the Kaggle Data Science Job Salaries dataset · "
    "Made by an MSc Statistics student to demystify data science compensation"
)
