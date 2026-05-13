import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# WMDST - Waste Mapping Decision Support Tool
# Streamlit Web App Version
# ============================================================

st.set_page_config(
    page_title="WMDST - Waste Mapping Decision Support Tool",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Default data
# -----------------------------
DEFAULT_DEPARTMENTS = pd.DataFrame({
    "Department": [
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
    ],
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


# -----------------------------
# Session state
# -----------------------------
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


# -----------------------------
# Helper functions
# -----------------------------
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


def calculate_kpis(df, net_available_time_sec=28800):
    df = df.copy()

    numeric_cols = [
        "Cycle Time (sec)", "Throughput Time (sec)", "Actual Output (pcs)",
        "FTT (%)", "Scrap Qty", "Rework Qty", "Overproduction Qty",
        "Resource Multiplier", "Cost Score (1 Low - 5 High)",
        "Effort Score (1 Easy - 5 Hard)"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Waiting Time (sec)"] = (df["Throughput Time (sec)"] - df["Cycle Time (sec)"]).clip(lower=0)
    df["Inventory Days"] = df["Throughput Time (sec)"] / 86400
    df["Value Added Ratio (%)"] = safe_divide(df["Cycle Time (sec)"], df["Throughput Time (sec)"]) * 100
    df["Defect Loss (%)"] = (100 - df["FTT (%)"]).clip(lower=0)
    df["Scrap Rate (%)"] = safe_divide(df["Scrap Qty"], df["Actual Output (pcs)"]) * 100
    df["Rework Rate (%)"] = safe_divide(df["Rework Qty"], df["Actual Output (pcs)"]) * 100
    df["Overproduction Rate (%)"] = safe_divide(df["Overproduction Qty"], df["Actual Output (pcs)"]) * 100
    df["Theoretical Output (pcs)"] = safe_divide(
        net_available_time_sec * df["Resource Multiplier"],
        df["Cycle Time (sec)"]
    )
    df["Efficiency (%)"] = safe_divide(df["Actual Output (pcs)"], df["Theoretical Output (pcs)"]) * 100
    df["Capacity Utilisation (%)"] = df["Efficiency (%)"].clip(upper=100)

    df["Waiting Norm"] = normalize(df["Waiting Time (sec)"])
    df["Inventory Norm"] = normalize(df["Inventory Days"])
    df["Defect Norm"] = normalize(df["Defect Loss (%)"])
    df["Scrap Norm"] = normalize(df["Scrap Rate (%)"])
    df["Rework Norm"] = normalize(df["Rework Rate (%)"])
    df["Overproduction Norm"] = normalize(df["Overproduction Rate (%)"])

    # KPI-based waste score. Weights can be changed to match company policy.
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
    ).fillna(0)

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

    # Geometric mean method
    geo_mean = np.prod(matrix, axis=1) ** (1 / len(criteria))
    weights = geo_mean / geo_mean.sum()

    weighted_sum = matrix @ weights
    lambda_max = np.mean(weighted_sum / weights)
    ci = (lambda_max - len(criteria)) / (len(criteria) - 1)
    ri = 0.58
    cr = ci / ri if ri != 0 else 0

    weights_df = pd.DataFrame({
        "Criteria": criteria,
        "Weight": weights
    })

    return weights_df, lambda_max, ci, cr, pd.DataFrame(matrix, index=criteria, columns=criteria)


def calculate_ahp_department_ranking(kpi_df, ahp_weights):
    df = kpi_df.copy()

    impact_weight = float(ahp_weights.loc[ahp_weights["Criteria"] == "Impact", "Weight"].iloc[0])
    cost_weight = float(ahp_weights.loc[ahp_weights["Criteria"] == "Cost", "Weight"].iloc[0])
    effort_weight = float(ahp_weights.loc[ahp_weights["Criteria"] == "Effort", "Weight"].iloc[0])

    df["Impact Score"] = normalize(df["KPI Waste Score"])

    # Lower cost and lower effort are more favourable for improvement priority.
    df["Cost Feasibility Score"] = 1 - normalize(df["Cost Score (1 Low - 5 High)"])
    df["Effort Feasibility Score"] = 1 - normalize(df["Effort Score (1 Easy - 5 Hard)"])

    df["AHP Priority Score"] = (
        impact_weight * df["Impact Score"] +
        cost_weight * df["Cost Feasibility Score"] +
        effort_weight * df["Effort Feasibility Score"]
    )

    df["Priority Level"] = df["AHP Priority Score"].apply(priority_label)
    df["AHP Rank"] = df["AHP Priority Score"].rank(ascending=False, method="dense").astype(int)

    return df.sort_values("AHP Rank")


def recommendation_for(row):
    waste = row["Dominant Waste"]
    priority = row["Priority Level"]

    if waste == "Inventory":
        action = "Introduce FIFO/pull control, align shipment schedule, reduce holding time, and review storage policy."
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


def make_excel_download(kpi_df, wam_df, ahp_weights, ahp_ranked, recommendation_df, cutting_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        kpi_df.to_excel(writer, index=False, sheet_name="KPI Results")
        wam_df.to_excel(writer, index=False, sheet_name="WAM Results")
        ahp_weights.to_excel(writer, index=False, sheet_name="AHP Weights")
        ahp_ranked.to_excel(writer, index=False, sheet_name="AHP Ranking")
        recommendation_df.to_excel(writer, index=False, sheet_name="Recommendations")
        cutting_df.to_excel(writer, index=False, sheet_name="Cutting Waste")
    return output.getvalue()


# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("WMDST Tool")
page = st.sidebar.radio(
    "Go to",
    [
        "1. Home",
        "2. Production Data Form",
        "3. WAM Questionnaire",
        "4. AHP Input",
        "5. Cutting Waste Analysis",
        "6. Dashboard",
        "7. Recommendations & Export"
    ]
)

st.sidebar.markdown("---")
net_available_time = st.sidebar.number_input(
    "Net available time per day (sec)",
    min_value=1,
    value=28800,
    step=600
)


# -----------------------------
# Common calculations
# -----------------------------
kpi_df = calculate_kpis(st.session_state.production_df, net_available_time_sec=net_available_time)

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
    "Department", "Dominant Waste", "KPI Waste Score", "AHP Priority Score", "Priority Level"
]].copy()
recommendation_df["Recommendation"] = recommendation_df.apply(recommendation_for, axis=1)

cutting_df = calculate_cutting_waste(st.session_state.cutting_inputs)


# -----------------------------
# Pages
# -----------------------------
if page == "1. Home":
    st.title("📊 WMDST - Waste Mapping Decision Support Tool")
    st.subheader("Form-based web application version")

    st.markdown("""
    This tool converts the Excel-based WMDST into a simple web application.

    **Main workflow:**

    1. Enter department-wise production data.
    2. Calculate waste KPIs automatically.
    3. Use WAM to rank waste types.
    4. Use AHP to rank departments.
    5. Generate improvement recommendations.
    6. Export the final results.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Departments", len(kpi_df))
    c2.metric("Top Department", ahp_ranked.iloc[0]["Department"])
    c3.metric("Highest Waste Type", wam_df.iloc[0]["Waste Type"])
    c4.metric("AHP CR", f"{cr:.4f}")

    st.info(
        "Use the sidebar to move through the tool. Start from Production Data Form, then WAM, AHP, Dashboard, and Export."
    )


elif page == "2. Production Data Form":
    st.title("2. Production Data Form")
    st.markdown("Enter or edit department-wise production data. This replaces the Excel **Data Input** sheet.")

    uploaded = st.file_uploader("Optional: upload production data CSV", type=["csv"])
    if uploaded is not None:
        uploaded_df = pd.read_csv(uploaded)
        st.session_state.production_df = uploaded_df
        st.success("CSV data loaded into the tool.")

    edited_df = st.data_editor(
        st.session_state.production_df,
        num_rows="dynamic",
        use_container_width=True,
        key="production_editor"
    )

    if st.button("Save Production Data"):
        st.session_state.production_df = edited_df
        st.success("Production data saved.")

    st.subheader("Calculated KPI Preview")
    preview_cols = [
        "Department", "Waiting Time (sec)", "Inventory Days", "Value Added Ratio (%)",
        "Defect Loss (%)", "Scrap Rate (%)", "Rework Rate (%)", "KPI Waste Score", "Dominant Waste"
    ]
    st.dataframe(kpi_df[preview_cols], use_container_width=True)


elif page == "3. WAM Questionnaire":
    st.title("3. WAM Questionnaire")
    st.markdown("This section replaces the Excel **WAM Questionnaire** and **WAM Relationship Matrix** sheets.")

    st.subheader("Likert Scale Waste Questionnaire")
    edited_wam = st.data_editor(
        st.session_state.wam_questionnaire,
        num_rows="fixed",
        use_container_width=True,
        key="wam_editor"
    )

    st.subheader("Waste Relationship Matrix")
    st.caption("Use 0 = no relationship, 1 = weak, 2 = medium, 3 = strong.")
    edited_matrix = st.data_editor(
        st.session_state.relationship_matrix,
        use_container_width=True,
        key="matrix_editor"
    )

    if st.button("Save WAM Inputs"):
        st.session_state.wam_questionnaire = edited_wam
        st.session_state.relationship_matrix = edited_matrix
        st.success("WAM inputs saved.")

    st.subheader("WAM Results")
    st.dataframe(wam_df, use_container_width=True)

    fig = px.bar(
        wam_df,
        x="Waste Type",
        y="WAM Weight (%)",
        text="WAM Weight (%)",
        title="WAM Waste Type Ranking"
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "4. AHP Input":
    st.title("4. AHP Input")
    st.markdown("This section replaces the Excel **AHP Input** sheet.")

    st.info(
        "Use Saaty's 1-9 scale. Example: if Impact is moderately more important than Cost, select 3."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        impact_vs_cost = st.slider(
            "Impact vs Cost",
            min_value=1.0,
            max_value=9.0,
            value=float(st.session_state.ahp_pairwise["Impact_vs_Cost"]),
            step=1.0
        )

    with c2:
        impact_vs_effort = st.slider(
            "Impact vs Effort",
            min_value=1.0,
            max_value=9.0,
            value=float(st.session_state.ahp_pairwise["Impact_vs_Effort"]),
            step=1.0
        )

    with c3:
        cost_vs_effort = st.slider(
            "Cost vs Effort",
            min_value=1.0,
            max_value=9.0,
            value=float(st.session_state.ahp_pairwise["Cost_vs_Effort"]),
            step=1.0
        )

    if st.button("Save AHP Inputs"):
        st.session_state.ahp_pairwise = {
            "Impact_vs_Cost": impact_vs_cost,
            "Impact_vs_Effort": impact_vs_effort,
            "Cost_vs_Effort": cost_vs_effort
        }
        st.success("AHP inputs saved.")

    st.subheader("Pairwise Comparison Matrix")
    st.dataframe(pairwise_matrix.style.format("{:.3f}"), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Lambda Max", f"{lambda_max:.4f}")
    c2.metric("Consistency Index", f"{ci:.4f}")
    c3.metric("Consistency Ratio", f"{cr:.4f}")

    if cr <= 0.10:
        st.success("AHP consistency is acceptable because CR ≤ 0.10.")
    else:
        st.warning("AHP consistency is not ideal. Adjust pairwise comparison values until CR ≤ 0.10.")

    st.subheader("Criteria Weights")
    st.dataframe(weights_df, use_container_width=True)

    fig = px.pie(
        weights_df,
        names="Criteria",
        values="Weight",
        title="AHP Criteria Weights"
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "5. Cutting Waste Analysis":
    st.title("5. Cutting Waste Analysis")
    st.markdown("This section replaces the Excel **Cutting Waste Analysis** sheet.")

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

    if st.button("Save Cutting Waste Inputs"):
        st.session_state.cutting_inputs = {
            "Fabric Input (rolls)": fabric_input,
            "Marker Efficiency (%)": marker_efficiency,
            "Reusable Off-cut (%)": reusable_offcut
        }
        st.success("Cutting waste inputs saved.")

    st.subheader("Cutting Waste Result")
    st.dataframe(cutting_df, use_container_width=True)

    fig = px.bar(
        cutting_df[cutting_df["Metric"].isin([
            "Estimated Off-cut (%)", "Reusable Off-cut (%)", "Final Disposal Waste (%)"
        ])],
        x="Metric",
        y="Value",
        text="Value",
        title="Material Waste Breakdown"
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "6. Dashboard":
    st.title("6. Executive Dashboard")

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

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            ahp_ranked.sort_values("AHP Priority Score", ascending=True),
            x="AHP Priority Score",
            y="Department",
            orientation="h",
            title="Department Priority Ranking"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            wam_df,
            x="Waste Type",
            y="WAM Weight (%)",
            text="WAM Weight (%)",
            title="Waste Type Ranking by WAM"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("KPI Status Table")
    status_cols = [
        "Department", "Waiting Time (sec)", "Inventory Days", "Value Added Ratio (%)",
        "KPI Waste Score", "Dominant Waste", "AHP Priority Score", "Priority Level"
    ]
    st.dataframe(ahp_ranked[status_cols], use_container_width=True)


elif page == "7. Recommendations & Export":
    st.title("7. Recommendations & Export")

    st.subheader("Improvement Recommendations")
    st.dataframe(recommendation_df, use_container_width=True)

    st.subheader("Download Results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    excel_file = make_excel_download(
        kpi_df,
        wam_df,
        weights_df,
        ahp_ranked,
        recommendation_df,
        cutting_df
    )

    st.download_button(
        label="Download Full Result as Excel",
        data=excel_file,
        file_name=f"WMDST_Results_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    csv_file = recommendation_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Recommendations as CSV",
        data=csv_file,
        file_name=f"WMDST_Recommendations_{timestamp}.csv",
        mime="text/csv"
    )

    st.markdown("""
    **Report explanation:**

    The exported result contains KPI results, WAM results, AHP weights, department ranking,
    recommendations, and cutting material waste analysis. This allows the company user to keep a record
    of each production style analysis.
    """)