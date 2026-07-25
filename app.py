'''The dashboard allows users to upload battery datasets,
predict battery health, optimize replacement decisions,
and visualize fleet-level analytics.'''
import streamlit as st
import tempfile
import zipfile
import os
import pandas as pd
from optimization.optimizer import run_pipeline

# %%
# Page configuration
st.set_page_config(
    page_title = 'Battery Predictive Maintenance',
    layout = 'wide'
)

# %%
st.title("Battery Predictive Maintenance Dashboard")

st.markdown("""
This dashboard predicts the **State of Health (SOH)** of lithium-ion batteries,
estimates **Remaining Useful Life (RUL)**, evaluates battery health, and
optimizes replacement decisions using **Operations Research (OR-Tools)**.
""")

st.divider()

# %%
# Sidebar
st.sidebar.header("Optimization Constraints")

budget = st.sidebar.number_input(
    "Budget",
    min_value=0,
    value=5000,
    step=500,)

max_hours = st.sidebar.number_input(
    "Maximum Maintenance Hours",
    min_value=1,
    value=20,)

max_replacements = st.sidebar.number_input(
    "Maximum Batteries to Replace",
    min_value=1,
    value=5,)

# %%
# File Upload
st.header("Upload Input Files")

battery_zip = st.file_uploader(
    "Upload NASA Battery Dataset (.zip) (Check readme for the format)",
    type="zip")

parameter_csv = st.file_uploader(
    "Upload Operational Parameters (.csv)  (Check readme for the format)",
    type="csv")

# %%
# Analysis
analyze = st.button(
    "Analyze Fleet",
    use_container_width=True,)


# %%
if analyze:

    if battery_zip is None or parameter_csv is None:
        st.error("Please upload both the battery dataset (.zip) and the operational parameter CSV.")
        st.stop()

    with tempfile.TemporaryDirectory() as temp_dir:# Create a temporary workspace for uploaded files.

        zip_path = os.path.join(temp_dir, "battery_dataset.zip")

        with open(zip_path, "wb") as f:
            f.write(battery_zip.getbuffer())

        extract_folder = os.path.join(temp_dir, "battery_data")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

        csv_path = os.path.join(temp_dir, parameter_csv.name)

        with open(csv_path, "wb") as f:
            f.write(parameter_csv.getbuffer())

        with st.spinner("Running battery health analysis..."):

            entries = os.listdir(extract_folder)
            if (len(entries) == 1 and os.path.isdir(os.path.join(extract_folder, entries[0]))):
                data_folder = os.path.join(extract_folder, entries[0])
            else:
                data_folder = extract_folder

            results = run_pipeline(
                folder_path=data_folder,
                csv_path=csv_path,
                budget=budget,
                max_hours=max_hours,
                max_replacements=max_replacements,
            )

        st.success("Analysis completed successfully")
    # Dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Fleet Summary",
    "Health Report",
    "Maintenance Plan",
    "Predictions",
    "Visualizations"]
    )
    with tab1:
        summary = results["fleet_summary"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
        "Average SOH",
        f"{summary['Average_SOH']:.3f}")
        col2.metric(
        "Average RUL",
        f"{summary['Average_Remaining_Useful_Life']:.1f}")
        col3.metric(
        "Critical Batteries",
        summary["Critical"])
        col4.metric(
        "Replacements",
        summary["Batteries_Replaced"])
        st.divider()

        st.subheader("Fleet Statistics")
        st.dataframe(
        pd.DataFrame([summary]),
        use_container_width=True)
    with tab2:
        st.subheader("Battery Health Report")
        st.dataframe(
        results["health_report"],
        use_container_width=True)
    with tab3:
        st.subheader("Recommended Battery Replacements")
        st.dataframe(
        results["maintenance_plan"],
        use_container_width=True)
    with tab4:
        st.subheader("SOH Predictions")
        st.dataframe(
        results["prediction"],
        use_container_width=True)
    with tab5:
        st.subheader("State of Health Curves")
        for battery, fig in results["plots"]["soh"].items():
            st.markdown(f"{battery}")
            st.pyplot(fig)
        st.divider()

        st.subheader("Remaining Useful Life")

        st.pyplot(results["plots"]["rul"])

        st.divider()

        st.subheader("Fleet Risk")
        st.pyplot(results["plots"]["risk"])

# %%


    


