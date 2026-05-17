import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# WMDST - Waste Mapping Decision Support Tool
# Improved Streamlit Web App Version
# ============================================================

st.set_page_config(
    page_title="WMDST | Waste Mapping Decision Support Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# VISUAL STYLE
# ============================================================

CUSTOM_CSS = """
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 52%, #0ea5e9 100%);
    color: white;
    padding: 28px 30px;
    border-radius: 22px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.18);
    margin-bottom: 22px;
}
.hero-card h1 {
    font-size: 2.15rem;
    margin-bottom: 0.25rem;
}
.hero-card p {
    font-size: 1.02rem;
    margin-bottom: 0;
    opacity: 0.95;
}
.section-card {
    background: white;
    padding: 20px 22px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    margin-bottom: 18px;
}
.small-note {
    color: #475569;
    font-size: 0.93rem;
}
.status-good {
    background-color: #dcfce7;
    color: #166534;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
}
.status-warning {
    background-color: #fef3c7;
    color: #92400e;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
}
.status-critical {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
}
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e5e7eb;
    padding: 16px 18px;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.06);
}
div[data-testid="stMetricLabel"] {
    font-size: 0.86rem;
}
div[data-testid="stMetricValue"] {
    font-size: 1.35rem;
}
.stButton button {
    border-radius: 12px;
    font-weight: 700;
}
.stDownloadButton button {
    border-radius: 12px;
    font-weight: 700;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# DEFAULT DATA
# ============================================================

DEPARTMENT_LIST = [
    "Unloading & Storing",
    "Picking",
    "Laying",
    "Cutting",
    "Sewing (CPU Main)",
    "Metal Detection",
    "Pre-Final Inspection",
    "Final Inspection",
    "Finished Goods Storing",
    "Shipment Preparation",
    "Loading"
]

DEFAULT_DEPARTMENTS = pd.DataFrame({
    "Department": DEPARTMENT_LIST,
    "Cycle Time (sec)": [60, 45, 120, 180, 240, 30, 90, 120, 60, 180, 90],
    "Throughput Time (sec)": [3600, 2700, 7200, 5400, 14400, 1800, 3600, 5400, 761000, 7200, 3600],
    "Actual Output (pcs)": [500, 500, 500, 500, 480, 480, 470, 460, 460, 460, 460],
    "FTT (%)": [98, 98, 97, 97, 92, 96, 95, 95, 99, 99, 99],
    "Scrap Qty": [0, 0, 2, 5, 8, 1, 0, 0, 0, 0, 0],
    "Rework Qty": [0, 0, 4, 5, 25, 4, 8, 5, 0, 0, 0],
    "Overproduction Qty": [0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0],
    "Resource Multiplier": [1, 1, 1, 1, 6, 1, 2, 2, 1, 1, 1],
    "Cost Score (1 Low - 5 High)": [2, 2, 3, 3, 4, 2, 3, 3, 2, 2, 2],
    "Effort Score (1 Easy - 5 Hard)": [2, 2, 3, 3, 4, 2, 3, 3, 2, 2, 2]
})

REQUIRED_COLUMNS = list(DEFAULT_DEPARTMENTS.columns)

WASTE_TYPES = ["Waiting", "Inventory", "Defects", "Overproduction"]

DEFAULT_RELATIONSHIP_MATRIX = pd.DataFrame(
    [
        [0, 2, 2, 1],
        [2, 0, 2, 1],
        [2, 2, 0, 1],
        [1, 2, 1, 0],
    ],
    index=WASTE_TYPES,
    columns=WASTE_TYPES
)

DEFAULT_WAM_QUESTIONNAIRE = pd.DataFrame({
    "Waste Type": WASTE_TYPES,
    "Likert Score (1 Low - 5 High)": [5, 4, 4, 3]
})


# ============================================================
# SESSION STATE
# ============================================================

def init_state():
    if "production_df" not in st.session_state:
        st.session_state.production_df = DEFAULT_DEPARTMENTS.copy()

    if "wam_questionnaire" not in st.session_state:
        st.session_state.wam_questionnaire = DEFAULT_WAM_QUESTIONNAIRE.copy()

    if "relationship_matrix" not in st.session_state:
        st.session_state.relationship_matrix = DEFAULT_RELATIONSHIP_MATRIX.copy()

    if "ahp_pairwise" not in st.session_state:
        st.session_state.ahp_pairwise = {
            "Impact_vs_Cost": 3.0,
            "Impact_vs_Effort": 5.0,
            "Cost_vs_Effort": 2.0
        }

    if "cutting_inputs" not in st.session_state:
        st.session_state.cutting_inputs = {
            "Fabric Input (rolls)": 500.0,
            "Marker Efficiency (%)": 80.0,
            "Reusable Off-cut (%)": 15.0
        }

    if "project_info" not in st.session_state:
        st.session_state.project_info = {
            "Style / Product": "Boxer Style 01",
            "Line / Factory": "Line 01",
            "Prepared By": "Production / IE Team",
            "Analysis Date": datetime.now().date()
        }


init_state()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def safe_divide(numerator, denominator):
    denominator = np.where(np.asarray(denominator) == 0, np.nan, denominator)
    result = numerator / denominator
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def normalize(series):
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_value) / (max_value - min_value)


def priority_label(score):
    if score >= 0.75:
        return "Critical"
    if score >= 0.50:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


def clean_production_df(df):
    df = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = DEFAULT_DEPARTMENTS[col].iloc[0] if col == "Department" else 0

    df = df[REQUIRED_COLUMNS]
    df["Department"] = df["Department"].astype(str).str.strip()

    numeric_cols = [col for col in REQUIRED_COLUMNS if col != "Department"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Basic boundary protection
    non_negative_cols = [
        "Cycle Time (sec)", "Throughput Time (sec)", "Actual Output (pcs)",
        "Scrap Qty", "Rework Qty", "Overproduction Qty", "Resource Multiplier"
    ]
    for col in non_negative_cols:
        df[col] = df[col].clip(lower=0)

    df["FTT (%)"] = df["FTT (%)"].clip(lower=0, upper=100)
    df["Cost Score (1 Low - 5 High)"] = df["Cost Score (1 Low - 5 High)"].clip(lower=1, upper=5)
    df["Effort Score (1 Easy - 5 Hard)"] = df["Effort Score (1 Easy - 5 Hard)"].clip(lower=1, upper=5)

    df = df[df["Department"] != ""].reset_index(drop=True)

    return df


def validate_production_df(df):
    warnings = []

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        warnings.append(f"Missing columns: {', '.join(missing)}")

    if df.empty:
        warnings.append("Production table is empty.")

    if "Department" in df.columns and df["Department"].duplicated().any():
        warnings.append("Some department names are duplicated. Ranking may be confusing.")

    if "Cycle Time (sec)" in df.columns and (pd.to_numeric(df["Cycle Time (sec)"], errors="coerce") <= 0).any():
        warnings.append("Some cycle time values are zero or negative. Theoretical output may become zero.")

    if "Throughput Time (sec)" in df.columns and "Cycle Time (sec)" in df.columns:
        bad_rows = df[pd.to_numeric(df["Throughput Time (sec)"], errors="coerce") < pd.to_numeric(df["Cycle Time (sec)"], errors="coerce")]
        if not bad_rows.empty:
            warnings.append("Some throughput times are less than cycle time. Waiting time will be treated as zero for those rows.")

    if "FTT (%)" in df.columns:
        ftt = pd.to_numeric(df["FTT (%)"], errors="coerce")
        if ((ftt < 0) | (ftt > 100)).any():
            warnings.append("Some FTT values are outside 0-100%. They will be clipped to the valid range.")

    return warnings


def explain_priority(priority):
    if priority == "Critical":
        return "Fix first"
    if priority == "High":
        return "Fix soon"
    if priority == "Medium":
        return "Monitor and improve"
    return "Low monitoring priority"


def csv_template_download():
    return DEFAULT_DEPARTMENTS.to_csv(index=False).encode("utf-8")


def upsert_department(df, row_dict):
    df = df.copy()
    department = row_dict["Department"]

    if department in df["Department"].values:
        row_index = df.index[df["Department"] == department][0]
        for key, value in row_dict.items():
            df.loc[row_index, key] = value
    else:
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

    return clean_production_df(df)


def style_priority_dataframe(df):
    def color_priority(value):
        if value == "Critical":
            return "background-color: #fee2e2; color: #991b1b; font-weight: bold"
        if value == "High":
            return "background-color: #ffedd5; color: #9a3412; font-weight: bold"
        if value == "Medium":
            return "background-color: #fef9c3; color: #854d0e; font-weight: bold"
        return "background-color: #dcfce7; color: #166534; font-weight: bold"

    format_dict = {
        "Waiting Time (hr)": "{:.2f}",
        "Inventory Days": "{:.2f}",
        "Value Added Ratio (%)": "{:.4f}",
        "Defect Loss (%)": "{:.2f}",
        "KPI Waste Score": "{:.3f}",
        "AHP Priority Score": "{:.3f}"
    }
    format_dict = {k: v for k, v in format_dict.items() if k in df.columns}

    return df.style.format(format_dict).applymap(color_priority, subset=["Priority Level"])


# ============================================================
# CALCULATION FUNCTIONS
# ============================================================

def calculate_kpis(df, net_available_time_sec=28800):
    df = clean_production_df(df)

    df["Waiting Time (sec)"] = (
        df["Throughput Time (sec)"] - df["Cycle Time (sec)"]
    ).clip(lower=0)

    df["Waiting Time (hr)"] = df["Waiting Time (sec)"] / 3600
    df["Inventory Days"] = df["Throughput Time (sec)"] / 86400

    df["Value Added Ratio (%)"] = safe_divide(
        df["Cycle Time (sec)"],
        df["Throughput Time (sec)"]
    ) * 100

    df["Non-Value Added Ratio (%)"] = 100 - df["Value Added Ratio (%)"]

    df["Defect Loss (%)"] = (100 - df["FTT (%)"]).clip(lower=0)

    df["Scrap Rate (%)"] = safe_divide(
        df["Scrap Qty"],
        df["Actual Output (pcs)"]
    ) * 100

    df["Rework Rate (%)"] = safe_divide(
        df["Rework Qty"],
        df["Actual Output (pcs)"]
    ) * 100

    df["Overproduction Rate (%)"] = safe_divide(
        df["Overproduction Qty"],
        df["Actual Output (pcs)"]
    ) * 100

    df["Theoretical Output (pcs/day)"] = safe_divide(
        net_available_time_sec * df["Resource Multiplier"],
        df["Cycle Time (sec)"]
    )

    df["Efficiency (%)"] = safe_divide(
        df["Actual Output (pcs)"],
        df["Theoretical Output (pcs/day)"]
    ) * 100

    df["Capacity Utilisation (%)"] = df["Efficiency (%)"].clip(upper=100)

    df["Waiting Norm"] = normalize(df["Waiting Time (sec)"])
    df["Inventory Norm"] = normalize(df["Inventory Days"])
    df["Defect Norm"] = normalize(df["Defect Loss (%)"])
    df["Scrap Norm"] = normalize(df["Scrap Rate (%)"])
    df["Rework Norm"] = normalize(df["Rework Rate (%)"])
    df["Overproduction Norm"] = normalize(df["Overproduction Rate (%)"])

    # KPI-based waste score. These weights can be changed if the organization changes its policy.
    df["KPI Waste Score"] = (
        0.25 * df["Waiting Norm"] +
        0.25 * df["Inventory Norm"] +
        0.20 * df["Defect Norm"] +
        0.10 * df["Scrap Norm"] +
        0.10 * df["Rework Norm"] +
        0.10 * df["Overproduction Norm"]
    )

    waste_signal_cols = {
        "Waiting": "Waiting Norm",
        "Inventory": "Inventory Norm",
        "Defects": "Defect Norm",
        "Overproduction": "Overproduction Norm"
    }

    df["Dominant Waste"] = df[list(waste_signal_cols.values())].idxmax(axis=1)
    reverse_map = {v: k for k, v in waste_signal_cols.items()}
    df["Dominant Waste"] = df["Dominant Waste"].map(reverse_map)

    return df


def calculate_wam(questionnaire_df, relationship_matrix):
    q = questionnaire_df.copy()

    q["Likert Score (1 Low - 5 High)"] = pd.to_numeric(
        q["Likert Score (1 Low - 5 High)"],
        errors="coerce"
    ).fillna(0).clip(1, 5)

    relationship_matrix = relationship_matrix.copy()
    relationship_matrix = relationship_matrix.apply(pd.to_numeric, errors="coerce").fillna(0).clip(0, 3)

    relation_strength = relationship_matrix.sum(axis=1).reindex(WASTE_TYPES).fillna(0)
    q = q.set_index("Waste Type").reindex(WASTE_TYPES).fillna(0)

    q["Relationship Strength"] = relation_strength
    q["WAM Raw Score"] = q["Likert Score (1 Low - 5 High)"] * (1 + q["Relationship Strength"])

    total = q["WAM Raw Score"].sum()
    q["WAM Weight (%)"] = safe_divide(q["WAM Raw Score"], total) * 100
    q["Rank"] = q["WAM Weight (%)"].rank(ascending=False, method="dense").astype(int)

    return q.reset_index().sort_values("Rank")


def calculate_ahp_weights(impact_vs_cost, impact_vs_effort, cost_vs_effort):
    criteria = ["Impact", "Cost", "Effort"]

    matrix = np.array([
        [1, impact_vs_cost, impact_vs_effort],
        [1 / impact_vs_cost, 1, cost_vs_effort],
        [1 / impact_vs_effort, 1 / cost_vs_effort, 1]
    ], dtype=float)

    geo_mean = np.prod(matrix, axis=1) ** (1 / len(criteria))
    weights = geo_mean / geo_mean.sum()

    weighted_sum = matrix @ weights
    lambda_max = np.mean(weighted_sum / weights)
    ci = (lambda_max - len(criteria)) / (len(criteria) - 1)
    ri = 0.58
    cr = ci / ri if ri != 0 else 0

    weights_df = pd.DataFrame({
        "Criteria": criteria,
        "Weight": weights,
        "Weight (%)": weights * 100
    })

    pairwise_matrix = pd.DataFrame(matrix, index=criteria, columns=criteria)

    return weights_df, lambda_max, ci, cr, pairwise_matrix


def calculate_ahp_department_ranking(kpi_df, ahp_weights):
    df = kpi_df.copy()

    impact_weight = float(ahp_weights.loc[ahp_weights["Criteria"] == "Impact", "Weight"].iloc[0])
    cost_weight = float(ahp_weights.loc[ahp_weights["Criteria"] == "Cost", "Weight"].iloc[0])
    effort_weight = float(ahp_weights.loc[ahp_weights["Criteria"] == "Effort", "Weight"].iloc[0])

    df["Impact Score"] = normalize(df["KPI Waste Score"])

    # Lower cost and lower effort are more favourable for quick improvement.
    df["Cost Feasibility Score"] = 1 - normalize(df["Cost Score (1 Low - 5 High)"])
    df["Effort Feasibility Score"] = 1 - normalize(df["Effort Score (1 Easy - 5 Hard)"])

    df["AHP Priority Score"] = (
        impact_weight * df["Impact Score"] +
        cost_weight * df["Cost Feasibility Score"] +
        effort_weight * df["Effort Feasibility Score"]
    )

    df["Priority Level"] = df["AHP Priority Score"].apply(priority_label)
    df["Action Meaning"] = df["Priority Level"].apply(explain_priority)
    df["AHP Rank"] = df["AHP Priority Score"].rank(ascending=False, method="dense").astype(int)

    return df.sort_values("AHP Rank")


def recommendation_for(row):
    waste = row["Dominant Waste"]
    priority = row["Priority Level"]

    if waste == "Inventory":
        action = "Introduce FIFO or pull control, align shipment schedule, reduce holding time, and review storage policy."
    elif waste == "Waiting":
        action = "Improve line balancing, material availability, work release planning, and bottleneck control."
    elif waste == "Defects":
        action = "Strengthen inline quality checks, operator training, poka-yoke, setup control, and root-cause analysis."
    elif waste == "Overproduction":
        action = "Control batch size, improve demand alignment, and reduce production ahead of confirmed requirement."
    else:
        action = "Review the department using a lean waste walk and update the data."

    if priority == "Critical":
        timing = "Immediate action required."
    elif priority == "High":
        timing = "Short-term improvement action needed."
    elif priority == "Medium":
        timing = "Monitor and improve gradually."
    else:
        timing = "Low priority; continue monitoring."

    return f"{action} {timing}"


def calculate_cutting_waste(cutting_inputs):
    fabric_input = float(cutting_inputs["Fabric Input (rolls)"])
    marker_efficiency = float(cutting_inputs["Marker Efficiency (%)"])
    reusable_offcut = float(cutting_inputs["Reusable Off-cut (%)"])

    offcut_pct = max(0.0, 100.0 - marker_efficiency)
    final_disposal_pct = max(0.0, offcut_pct - reusable_offcut)
    final_waste_rolls = fabric_input * final_disposal_pct / 100
    recovery_rate = safe_divide(reusable_offcut, offcut_pct) * 100 if offcut_pct > 0 else 0

    return pd.DataFrame({
        "Metric": [
            "Fabric Input (rolls)",
            "Marker Efficiency (%)",
            "Estimated Off-cut (%)",
            "Reusable Off-cut (%)",
            "Final Disposal Waste (%)",
            "Final Waste (rolls)",
            "Recovery Rate (%)"
        ],
        "Value": [
            fabric_input,
            marker_efficiency,
            offcut_pct,
            reusable_offcut,
            final_disposal_pct,
            final_waste_rolls,
            recovery_rate
        ]
    })


def make_excel_download(project_info, validation_notes, kpi_df, wam_df, ahp_weights, ahp_ranked, recommendation_df, cutting_df):
    output = io.BytesIO()

    project_df = pd.DataFrame({
        "Field": list(project_info.keys()),
        "Value": list(project_info.values())
    })

    validation_df = pd.DataFrame({
        "Validation Notes": validation_notes if validation_notes else ["No major data validation warnings."]
    })

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        project_df.to_excel(writer, index=False, sheet_name="Project Info")
        validation_df.to_excel(writer, index=False, sheet_name="Validation Notes")
        kpi_df.to_excel(writer, index=False, sheet_name="KPI Results")
        wam_df.to_excel(writer, index=False, sheet_name="WAM Results")
        ahp_weights.to_excel(writer, index=False, sheet_name="AHP Weights")
        ahp_ranked.to_excel(writer, index=False, sheet_name="AHP Ranking")
        recommendation_df.to_excel(writer, index=False, sheet_name="Recommendations")
        cutting_df.to_excel(writer, index=False, sheet_name="Cutting Waste")

    return output.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 WMDST")
st.sidebar.caption("Waste Mapping Decision Support Tool")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Home",
        "2. Easy Data Entry",
        "3. WAM Input",
        "4. AHP Input",
        "5. Cutting Waste",
        "6. Dashboard",
        "7. Recommendations & Export",
        "8. User Guide"
    ]
)

st.sidebar.markdown("---")

net_available_time = st.sidebar.number_input(
    "Net available time per day (sec)",
    min_value=1,
    value=28800,
    step=600,
    help="Default 28,800 seconds = 8 working hours."
)

with st.sidebar.expander("Project Information", expanded=False):
    st.session_state.project_info["Style / Product"] = st.text_input(
        "Style / Product",
        st.session_state.project_info["Style / Product"]
    )
    st.session_state.project_info["Line / Factory"] = st.text_input(
        "Line / Factory",
        st.session_state.project_info["Line / Factory"]
    )
    st.session_state.project_info["Prepared By"] = st.text_input(
        "Prepared By",
        st.session_state.project_info["Prepared By"]
    )
    st.session_state.project_info["Analysis Date"] = st.date_input(
        "Analysis Date",
        st.session_state.project_info["Analysis Date"]
    )

if st.sidebar.button("Reset all sample data"):
    st.session_state.production_df = DEFAULT_DEPARTMENTS.copy()
    st.session_state.wam_questionnaire = DEFAULT_WAM_QUESTIONNAIRE.copy()
    st.session_state.relationship_matrix = DEFAULT_RELATIONSHIP_MATRIX.copy()
    st.session_state.ahp_pairwise = {
        "Impact_vs_Cost": 3.0,
        "Impact_vs_Effort": 5.0,
        "Cost_vs_Effort": 2.0
    }
    st.session_state.cutting_inputs = {
        "Fabric Input (rolls)": 500.0,
        "Marker Efficiency (%)": 80.0,
        "Reusable Off-cut (%)": 15.0
    }
    rerun_app()


# ============================================================
# COMMON CALCULATIONS
# ============================================================

current_production_df = clean_production_df(st.session_state.production_df)
validation_notes = validate_production_df(current_production_df)

kpi_df = calculate_kpis(current_production_df, net_available_time_sec=net_available_time)

wam_df = calculate_wam(
    st.session_state.wam_questionnaire,
    st.session_state.relationship_matrix
)

weights_df, lambda_max, ci, cr, pairwise_matrix = calculate_ahp_weights(
    st.session_state.ahp_pairwise["Impact_vs_Cost"],
    st.session_state.ahp_pairwise["Impact_vs_Effort"],
    st.session_state.ahp_pairwise["Cost_vs_Effort"]
)

ahp_ranked = calculate_ahp_department_ranking(kpi_df, weights_df)

recommendation_df = ahp_ranked[[
    "AHP Rank", "Department", "Dominant Waste", "KPI Waste Score",
    "AHP Priority Score", "Priority Level", "Action Meaning"
]].copy()
recommendation_df["Recommendation"] = recommendation_df.apply(recommendation_for, axis=1)

cutting_df = calculate_cutting_waste(st.session_state.cutting_inputs)


# ============================================================
# REUSABLE UI BLOCKS
# ============================================================

def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero-card">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_validation_panel():
    if validation_notes:
        with st.expander("⚠️ Data validation warnings", expanded=True):
            for item in validation_notes:
                st.warning(item)
    else:
        st.success("Production input data passed the basic validation check.")


def plot_bar(df, x, y, title, orientation="v", text=None):
    fig = px.bar(
        df,
        x=x,
        y=y,
        orientation=orientation,
        text=text,
        title=title,
        template="plotly_white",
        color=y if orientation == "h" else x,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        title_font_size=18,
        showlegend=False,
        margin=dict(l=20, r=20, t=55, b=20),
        height=430
    )
    return fig


# ============================================================
# PAGE 1 - HOME
# ============================================================

if page == "1. Home":
    hero(
        "WMDST - Waste Mapping Decision Support Tool",
        "A simple web-based tool to identify waste, rank departments, and recommend improvement actions."
    )

    top_department = ahp_ranked.iloc[0]["Department"]
    top_priority = ahp_ranked.iloc[0]["Priority Level"]
    highest_waste = wam_df.iloc[0]["Waste Type"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Departments Analysed", len(kpi_df))
    c2.metric("Top Priority Department", top_department)
    c3.metric("Highest Waste Type", highest_waste)
    c4.metric("AHP Consistency Ratio", f"{cr:.4f}")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("How this tool works")
    st.markdown(
        """
        1. Enter production data using the easy form or full table.
        2. The tool calculates waiting time, inventory days, value-added ratio, defect loss, scrap rate, rework rate, and waste score.
        3. WAM ranks the main waste types.
        4. AHP ranks departments by impact, cost, and effort.
        5. The dashboard shows the top priority area and improvement actions.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    show_validation_panel()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Current first improvement action")
    st.write(f"**Department:** {recommendation_df.iloc[0]['Department']}")
    st.write(f"**Priority:** {recommendation_df.iloc[0]['Priority Level']}")
    st.write(f"**Dominant waste:** {recommendation_df.iloc[0]['Dominant Waste']}")
    st.info(recommendation_df.iloc[0]["Recommendation"])
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 2 - EASY DATA ENTRY
# ============================================================

elif page == "2. Easy Data Entry":
    hero(
        "Easy Production Data Entry",
        "Enter one department at a time, edit the full table, or upload a CSV template."
    )

    input_mode = st.radio(
        "Choose input method",
        ["Quick department form", "Full table editing", "Upload CSV"],
        horizontal=True
    )

    if input_mode == "Quick department form":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Quick form: add or update one department")

        existing_departments = list(current_production_df["Department"].unique())
        department_choice = st.selectbox(
            "Select existing department or type a new one below",
            existing_departments + ["Add new department"]
        )

        if department_choice == "Add new department":
            department_name = st.text_input("New department name", "New Department")
            default_row = DEFAULT_DEPARTMENTS.iloc[0].to_dict()
        else:
            department_name = department_choice
            default_row = current_production_df[current_production_df["Department"] == department_choice].iloc[0].to_dict()

        with st.form("quick_department_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                cycle_time = st.number_input("Cycle Time (sec)", min_value=0.0, value=float(default_row["Cycle Time (sec)"]), step=1.0)
                actual_output = st.number_input("Actual Output (pcs)", min_value=0.0, value=float(default_row["Actual Output (pcs)"]), step=1.0)
                scrap_qty = st.number_input("Scrap Qty", min_value=0.0, value=float(default_row["Scrap Qty"]), step=1.0)
            with c2:
                throughput_time = st.number_input("Throughput Time (sec)", min_value=0.0, value=float(default_row["Throughput Time (sec)"]), step=10.0)
                ftt = st.number_input("FTT (%)", min_value=0.0, max_value=100.0, value=float(default_row["FTT (%)"]), step=1.0)
                rework_qty = st.number_input("Rework Qty", min_value=0.0, value=float(default_row["Rework Qty"]), step=1.0)
            with c3:
                overproduction_qty = st.number_input("Overproduction Qty", min_value=0.0, value=float(default_row["Overproduction Qty"]), step=1.0)
                resource_multiplier = st.number_input("Resource Multiplier", min_value=0.0, value=float(default_row["Resource Multiplier"]), step=1.0)
                cost_score = st.slider("Cost Score (1 low - 5 high)", 1, 5, int(default_row["Cost Score (1 Low - 5 High)"]))
                effort_score = st.slider("Effort Score (1 easy - 5 hard)", 1, 5, int(default_row["Effort Score (1 Easy - 5 Hard)"]))

            submitted = st.form_submit_button("Save this department")

        if submitted:
            row = {
                "Department": department_name,
                "Cycle Time (sec)": cycle_time,
                "Throughput Time (sec)": throughput_time,
                "Actual Output (pcs)": actual_output,
                "FTT (%)": ftt,
                "Scrap Qty": scrap_qty,
                "Rework Qty": rework_qty,
                "Overproduction Qty": overproduction_qty,
                "Resource Multiplier": resource_multiplier,
                "Cost Score (1 Low - 5 High)": cost_score,
                "Effort Score (1 Easy - 5 Hard)": effort_score
            }
            st.session_state.production_df = upsert_department(current_production_df, row)
            st.success(f"{department_name} saved successfully.")
            rerun_app()

        st.markdown('</div>', unsafe_allow_html=True)

    elif input_mode == "Full table editing":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Full table editing")

        edited_df = st.data_editor(
            current_production_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Department": st.column_config.TextColumn("Department", required=True),
                "Cycle Time (sec)": st.column_config.NumberColumn("Cycle Time (sec)", min_value=0),
                "Throughput Time (sec)": st.column_config.NumberColumn("Throughput Time (sec)", min_value=0),
                "Actual Output (pcs)": st.column_config.NumberColumn("Actual Output (pcs)", min_value=0),
                "FTT (%)": st.column_config.NumberColumn("FTT (%)", min_value=0, max_value=100),
                "Scrap Qty": st.column_config.NumberColumn("Scrap Qty", min_value=0),
                "Rework Qty": st.column_config.NumberColumn("Rework Qty", min_value=0),
                "Overproduction Qty": st.column_config.NumberColumn("Overproduction Qty", min_value=0),
                "Resource Multiplier": st.column_config.NumberColumn("Resource Multiplier", min_value=0),
                "Cost Score (1 Low - 5 High)": st.column_config.NumberColumn("Cost Score", min_value=1, max_value=5),
                "Effort Score (1 Easy - 5 Hard)": st.column_config.NumberColumn("Effort Score", min_value=1, max_value=5),
            },
            key="production_editor"
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Save full table"):
                st.session_state.production_df = clean_production_df(edited_df)
                st.success("Full production table saved.")
                rerun_app()
        with c2:
            if st.button("Restore sample table"):
                st.session_state.production_df = DEFAULT_DEPARTMENTS.copy()
                st.success("Sample production table restored.")
                rerun_app()

        st.markdown('</div>', unsafe_allow_html=True)

    elif input_mode == "Upload CSV":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Upload production CSV")

        st.download_button(
            "Download CSV template",
            data=csv_template_download(),
            file_name="WMDST_Production_Data_Template.csv",
            mime="text/csv"
        )

        uploaded = st.file_uploader("Upload completed CSV file", type=["csv"])

        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded)
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in uploaded_df.columns]

            if missing_cols:
                st.error(f"The uploaded file is missing these columns: {', '.join(missing_cols)}")
            else:
                st.session_state.production_df = clean_production_df(uploaded_df)
                st.success("CSV data loaded successfully.")
                rerun_app()

        st.markdown('</div>', unsafe_allow_html=True)

    show_validation_panel()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Calculated KPI preview")

    preview_cols = [
        "Department", "Waiting Time (hr)", "Inventory Days", "Value Added Ratio (%)",
        "Defect Loss (%)", "Scrap Rate (%)", "Rework Rate (%)",
        "KPI Waste Score", "Dominant Waste"
    ]

    st.dataframe(
        kpi_df[preview_cols].style.format({
            "Waiting Time (hr)": "{:.2f}",
            "Inventory Days": "{:.2f}",
            "Value Added Ratio (%)": "{:.4f}",
            "Defect Loss (%)": "{:.2f}",
            "Scrap Rate (%)": "{:.2f}",
            "Rework Rate (%)": "{:.2f}",
            "KPI Waste Score": "{:.3f}"
        }),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 3 - WAM INPUT
# ============================================================

elif page == "3. WAM Input":
    hero(
        "WAM Input",
        "Score the seriousness of each waste type and define how waste types influence each other."
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Waste seriousness questionnaire")

    st.caption("Score 1 = very low problem, 5 = very high problem.")

    wam_scores = []
    current_wam = st.session_state.wam_questionnaire.set_index("Waste Type")

    c1, c2, c3, c4 = st.columns(4)
    columns = [c1, c2, c3, c4]

    for idx, waste in enumerate(WASTE_TYPES):
        with columns[idx]:
            value = st.slider(
                waste,
                min_value=1,
                max_value=5,
                value=int(current_wam.loc[waste, "Likert Score (1 Low - 5 High)"]),
                key=f"wam_{waste}"
            )
            wam_scores.append({"Waste Type": waste, "Likert Score (1 Low - 5 High)": value})

    if st.button("Save WAM questionnaire scores"):
        st.session_state.wam_questionnaire = pd.DataFrame(wam_scores)
        st.success("WAM questionnaire scores saved.")
        rerun_app()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Waste relationship matrix")
    st.caption("Use 0 = no relationship, 1 = weak, 2 = medium, 3 = strong.")

    edited_matrix = st.data_editor(
        st.session_state.relationship_matrix,
        use_container_width=True,
        column_config={col: st.column_config.NumberColumn(col, min_value=0, max_value=3, step=1) for col in WASTE_TYPES},
        key="matrix_editor"
    )

    if st.button("Save relationship matrix"):
        st.session_state.relationship_matrix = edited_matrix.apply(pd.to_numeric, errors="coerce").fillna(0).clip(0, 3)
        st.success("Relationship matrix saved.")
        rerun_app()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("WAM results")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.dataframe(
            wam_df.style.format({
                "Relationship Strength": "{:.0f}",
                "WAM Raw Score": "{:.2f}",
                "WAM Weight (%)": "{:.2f}"
            }),
            use_container_width=True
        )
    with c2:
        fig = plot_bar(
            wam_df,
            x="Waste Type",
            y="WAM Weight (%)",
            title="Waste Type Ranking by WAM",
            text="WAM Weight (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 4 - AHP INPUT
# ============================================================

elif page == "4. AHP Input":
    hero(
        "AHP Input",
        "Choose which decision criteria matter most for improvement priority."
    )

    def guided_pairwise(label, first, second, default_value, key_prefix):
        if default_value > 1:
            default_choice = first
            default_intensity = int(round(default_value))
        elif default_value < 1:
            default_choice = second
            default_intensity = int(round(1 / default_value))
        else:
            default_choice = "Equal"
            default_intensity = 1

        st.markdown(f"**{label}**")
        choice = st.radio(
            f"Which one is more important for {label}?",
            [first, "Equal", second],
            index=[first, "Equal", second].index(default_choice),
            horizontal=True,
            key=f"{key_prefix}_choice"
        )

        intensity = st.slider(
            "Importance strength",
            min_value=1,
            max_value=9,
            value=max(1, min(9, default_intensity)),
            step=1,
            key=f"{key_prefix}_intensity",
            help="1 = equal, 3 = moderate, 5 = strong, 7 = very strong, 9 = extreme"
        )

        if choice == "Equal":
            return 1.0
        if choice == first:
            return float(intensity)
        return 1.0 / float(intensity)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Guided pairwise comparison")

    c1, c2, c3 = st.columns(3)
    with c1:
        impact_vs_cost = guided_pairwise(
            "Impact vs Cost",
            "Impact",
            "Cost",
            st.session_state.ahp_pairwise["Impact_vs_Cost"],
            "impact_cost"
        )
    with c2:
        impact_vs_effort = guided_pairwise(
            "Impact vs Effort",
            "Impact",
            "Effort",
            st.session_state.ahp_pairwise["Impact_vs_Effort"],
            "impact_effort"
        )
    with c3:
        cost_vs_effort = guided_pairwise(
            "Cost vs Effort",
            "Cost",
            "Effort",
            st.session_state.ahp_pairwise["Cost_vs_Effort"],
            "cost_effort"
        )

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Save AHP comparison"):
            st.session_state.ahp_pairwise = {
                "Impact_vs_Cost": impact_vs_cost,
                "Impact_vs_Effort": impact_vs_effort,
                "Cost_vs_Effort": cost_vs_effort
            }
            st.success("AHP comparison saved.")
            rerun_app()
    with c2:
        if st.button("Use recommended default AHP values"):
            st.session_state.ahp_pairwise = {
                "Impact_vs_Cost": 3.0,
                "Impact_vs_Effort": 5.0,
                "Cost_vs_Effort": 2.0
            }
            st.success("Recommended AHP values restored.")
            rerun_app()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("AHP results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Lambda Max", f"{lambda_max:.4f}")
    c2.metric("Consistency Index", f"{ci:.4f}")
    c3.metric("Consistency Ratio", f"{cr:.4f}")

    if cr <= 0.10:
        st.success("AHP consistency is acceptable because CR ≤ 0.10.")
    else:
        st.warning("AHP consistency is not ideal. Adjust the pairwise values until CR ≤ 0.10.")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("**Criteria weights**")
        st.dataframe(
            weights_df.style.format({"Weight": "{:.4f}", "Weight (%)": "{:.2f}"}),
            use_container_width=True
        )
    with c2:
        fig = px.pie(
            weights_df,
            names="Criteria",
            values="Weight",
            title="AHP Criteria Weights",
            hole=0.45,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View pairwise matrix"):
        st.dataframe(pairwise_matrix.style.format("{:.3f}"), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 5 - CUTTING WASTE
# ============================================================

elif page == "5. Cutting Waste":
    hero(
        "Cutting Waste Analysis",
        "Calculate fabric off-cut, reusable off-cut, final disposal waste, and recovery rate."
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        fabric_input = st.number_input(
            "Fabric Input (rolls)",
            min_value=0.0,
            value=float(st.session_state.cutting_inputs["Fabric Input (rolls)"]),
            step=10.0
        )
    with c2:
        marker_efficiency = st.number_input(
            "Marker Efficiency (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.cutting_inputs["Marker Efficiency (%)"]),
            step=1.0
        )
    with c3:
        reusable_offcut = st.number_input(
            "Reusable Off-cut (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.cutting_inputs["Reusable Off-cut (%)"]),
            step=1.0
        )

    if st.button("Save cutting waste inputs"):
        st.session_state.cutting_inputs = {
            "Fabric Input (rolls)": fabric_input,
            "Marker Efficiency (%)": marker_efficiency,
            "Reusable Off-cut (%)": reusable_offcut
        }
        st.success("Cutting waste inputs saved.")
        rerun_app()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Cutting waste output")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(
            cutting_df.style.format({"Value": "{:.2f}"}),
            use_container_width=True
        )
    with c2:
        chart_df = cutting_df[
            cutting_df["Metric"].isin([
                "Estimated Off-cut (%)",
                "Reusable Off-cut (%)",
                "Final Disposal Waste (%)"
            ])
        ]
        fig = plot_bar(
            chart_df,
            x="Metric",
            y="Value",
            title="Material Waste Breakdown",
            text="Value"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 6 - DASHBOARD
# ============================================================

elif page == "6. Dashboard":
    hero(
        "Executive Dashboard",
        "Final management view showing the biggest waste, top priority department, and recommended action."
    )

    top_department = ahp_ranked.iloc[0]["Department"]
    top_priority = ahp_ranked.iloc[0]["Priority Level"]
    highest_waste = wam_df.iloc[0]["Waste Type"]
    flow_efficiency = safe_divide(
        kpi_df["Cycle Time (sec)"].sum(),
        kpi_df["Throughput Time (sec)"].sum()
    ) * 100
    total_waiting_days = kpi_df["Waiting Time (sec)"].sum() / 86400

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Top Priority Department", top_department)
    c2.metric("Priority Level", top_priority)
    c3.metric("Highest Waste Type", highest_waste)
    c4.metric("Flow Efficiency", f"{flow_efficiency:.4f}%")
    c5.metric("Total Waiting Time", f"{total_waiting_days:.2f} days")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Management interpretation")
    st.write(
        f"The first improvement focus should be **{top_department}**. "
        f"The priority level is **{top_priority}**, and the strongest overall waste signal is **{highest_waste}**. "
        f"The total waiting time in the system is approximately **{total_waiting_days:.2f} days**, "
        f"while the flow efficiency is only **{flow_efficiency:.4f}%**."
    )
    st.info(recommendation_df.iloc[0]["Recommendation"])
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = plot_bar(
            ahp_ranked.sort_values("AHP Priority Score", ascending=True),
            x="AHP Priority Score",
            y="Department",
            title="Department Priority Ranking",
            orientation="h"
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = plot_bar(
            wam_df,
            x="Waste Type",
            y="WAM Weight (%)",
            title="Waste Type Ranking by WAM",
            text="WAM Weight (%)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("KPI and priority status table")

    status_cols = [
        "AHP Rank", "Department", "Waiting Time (hr)", "Inventory Days",
        "Value Added Ratio (%)", "Defect Loss (%)", "KPI Waste Score",
        "Dominant Waste", "AHP Priority Score", "Priority Level"
    ]

    st.dataframe(
        style_priority_dataframe(ahp_ranked[status_cols]),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 7 - RECOMMENDATIONS & EXPORT
# ============================================================

elif page == "7. Recommendations & Export":
    hero(
        "Recommendations & Export",
        "Convert analysis into improvement actions and download the results."
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Improvement recommendations")

    st.dataframe(
        recommendation_df.style.format({
            "KPI Waste Score": "{:.3f}",
            "AHP Priority Score": "{:.3f}"
        }),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Download results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    excel_file = make_excel_download(
        st.session_state.project_info,
        validation_notes,
        kpi_df,
        wam_df,
        weights_df,
        ahp_ranked,
        recommendation_df,
        cutting_df
    )

    st.download_button(
        label="Download full result as Excel",
        data=excel_file,
        file_name=f"WMDST_Results_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    csv_file = recommendation_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download recommendations as CSV",
        data=csv_file,
        file_name=f"WMDST_Recommendations_{timestamp}.csv",
        mime="text/csv"
    )

    st.caption(
        "The Excel export includes project information, validation notes, KPI results, WAM results, AHP weights, ranking, recommendations, and cutting waste analysis."
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 8 - USER GUIDE
# ============================================================

elif page == "8. User Guide":
    hero(
        "User Guide",
        "Simple guide for company users and presentation explanation."
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("What data should the user enter?")

    st.markdown(
        """
        **Production data**
        - Department name
        - Cycle time
        - Throughput time
        - Actual output
        - FTT percentage
        - Scrap quantity
        - Rework quantity
        - Overproduction quantity
        - Resource multiplier
        - Cost score
        - Effort score

        **WAM data**
        - Waste seriousness scores from 1 to 5
        - Waste relationship scores from 0 to 3

        **AHP data**
        - Impact vs Cost
        - Impact vs Effort
        - Cost vs Effort

        **Cutting waste data**
        - Fabric input
        - Marker efficiency
        - Reusable off-cut percentage
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("How to explain the dashboard")

    st.markdown(
        """
        The dashboard converts production data into waste decision outputs.  
        First, it calculates KPI values such as waiting time, inventory days, value-added ratio, defect loss, scrap rate, and rework rate.  
        Then, WAM identifies the most critical waste type, while AHP ranks the departments by considering impact, cost, and effort.  
        Finally, the tool displays the top priority department and recommends improvement actions.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Why this is better than Excel")

    st.markdown(
        """
        - The user enters data through guided forms instead of editing many Excel sheets.
        - The formulas are protected inside the code.
        - The tool reduces accidental formula changes.
        - The dashboard is easier to understand for managers and supervisors.
        - It can be shared as a web link.
        - Results can still be downloaded as Excel for reporting.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)