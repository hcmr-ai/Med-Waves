import streamlit as st
from pathlib import Path
import pandas as pd

# --- Configuration ---
OUTPUTS_ROOT = Path("/data/tsolis/AI_project/output/eda")
SPATIAL_ROOT = OUTPUTS_ROOT / "spatial_heatmaps"
feature_cols = ["VHM0", "VMDR", "VTM02", "WDIR", "WSPD"]
feature_cols = [
        'WSPD', 'VHM0', 'VTM02', 'corrected_VHM0', 'corrected_VTM02',
        'U10', 'V10', 'wave_dir_sin', 'wave_dir_cos',
        'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
        'sin_month', 'cos_month', 'lat_norm', 'lon_norm'
    ]
years = ["2021", "2022", "2023"]
data_origins = ["Ground Truth", "Training (without correction)", "Training (without correction) + GT"]
stat_types = ["mean", "std"]
seasons = ["", "WINTER", "SPRING", "SUMMER", "AUTUMN"]  # '' is for annual/whole year

# --- Sidebar selections ---
st.sidebar.title("Med-WAV Dashboard")
origin = st.sidebar.selectbox("Data Origin", data_origins)
year = st.sidebar.selectbox("Year", years)
feature = st.sidebar.selectbox("Feature", feature_cols)
origin_raw = "with_reduced" if origin == "Ground Truth" else "augmented_with_labels" if "Training (without correction) + GT" else "without_reduced"

# --- Main Display ---
st.title("Med-WAV EDA Dashboard")

# --- Top-level tabs ---
main_tabs = st.tabs(["Data Splits", "Time Series & Distribution", "Spatial Heatmaps", "Sampled Data", "Experiments", "Features Description"])

# === TAB 1: Data Splits Overview ===
with main_tabs[0]:
    st.subheader("🗂️ Dataset Splits Overview")
    st.image(str(OUTPUTS_ROOT/"dataset_split.png"), caption="dataset_split", use_container_width=True)

    st.markdown("- **Date range:** 1 Jan 2017 → 31 Dec 2023 (7 years)")
    st.markdown("- **Days available:** 2568 total days")
    st.markdown("- **Grid points per day:** 380 × 1307 = 496,660")
    st.markdown("- **Time steps per day:** 24 hourly measurements")

    split_data = [
        {
            "Name": "🔍 Debug",
            "Train": "2017-01-01 → 2017-01-02",
            "Val": "2017-01-03",
            "Test": "2017-01-04",
            "Rows (approx)": "~2M"
        },
        {
            "Name": "🧪 Prototype",
            "Train": "2021-01-01 → 2021-12-31",
            "Val": "2022-01-01 → 2022-03-31",
            "Test": "2022-04-01 → 2022-06-30",
            "Rows (approx)": "~273M"
        },
        {
            "Name": "🧠 Full Experiment",
            "Train": "2017-01-01 → 2021-12-31",
            "Val": "2022-01-01 → 2022-12-31",
            "Test": "2023-01-01 → 2023-12-31",
            "Rows (approx)": "~1.27B"
        },
    ]
    st.dataframe(pd.DataFrame(split_data))
    
    with st.expander("ℹ️ Splitting Strategies"):
        st.markdown("""
            - **Debug**: Smallest possible, for quick dev/tests.
            - **Prototype**: For model/feature analytics, medium scale.
            - **Full Experiment**: Publication-grade, all data.

            ---

            #### 🔄 Bonus: Rolling & Expanding Window Splits _(planned)_

            - **Expanding Window:**  
            Train on all data up to time _t_, test on the next block.  
            Example:  
            Train: [2017 → 2018], Test: [2019]
            Train: [2017 → 2019], Test: [2020]

            _Useful for showing how performance improves as more data becomes available._

            - **Rolling Window:**  
            Use a fixed-size training window that "rolls" forward with time, always testing on future data.
            Example:  
            Train: [2017-2018], Test: [2019]
            Train: [2018-2019], Test: [2020]

            _Good for benchmarking in operational settings._

            - Both strategies are important for realistic **time series ML** evaluation:  
            - Prevent "future data leakage"
            - Mimic how models would be deployed live
            - Reveal performance stability over time

        """)
        # st.markdown("""
        #     **Visual Example:**

        #     ```
        #     Expanding Window:
        #     Train: |----|----|----|         |----|----|----|----|
        #     Test:                    |----|                  |----|

        #     Rolling Window:
        #     Train:      |----|----|----|
        #     Test:                    |----|
        #     (window slides forward)
        #     ```
        # """)
            
        st.markdown("""
            ---
            **Why use these?**

            - Avoids leaking future info into training
            - Better mimics real-world ML deployment for time series
            - Useful for robust error analysis, model selection, and publications

        """)
with main_tabs[1]:
    st.subheader(f"{feature} — {year} ({origin})")
    feat_dir = OUTPUTS_ROOT /origin_raw / year / feature

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Distribution", "Time Series", "Decomposition", "Seasonality", "Descriptive Analytics", "Feature Correlation"])

    with tab1:
        st.markdown("""
        This plot shows the overall distribution of the selected variable across the entire time period. 
        It helps us quickly spot the presence of outliers, skewness, and typical value ranges, 
        giving an initial sense of data quality and variability.
        """)
        plot_names = [
            ("actual_time_series.png", "Actual Time Series"),
            ("distribution.png", "Feature Distribution"),
        ]
        for file, caption in plot_names:
            path = feat_dir / file
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.info(f"{caption} not found.")
    
    with tab2:
        st.markdown("""
        The time series plot visualizes how the variable changes over time. It’s essential for detecting trends, 
        abrupt changes, and gaps or anomalies in the data. This plot also helps us identify patterns like periodicity 
        or unusual events.
        """)
        plot_names = [
            ("filled_outliers.png", "Outlier Detection"),
            ("smoothed_differenced.png", "Smoothed & Differenced Series"),
        ]
        for file, caption in plot_names:
            path = feat_dir / file
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.info(f"{caption} not found.")
    with tab3:
        st.markdown("""
        Time series decomposition breaks down the observed series into trend, seasonal, and residual components. 
        This is crucial for understanding underlying patterns and choosing the right modeling approach 
        (e.g., handling seasonality explicitly).
        """)
        plot_names = [
            ("time_series_decomposition_add.png", "Ts Additive Decomposition"),
            ("time_series_decomposition_multi.png", "Ts Multivariate Decomposition"),
        ]
        for file, caption in plot_names:
            path = feat_dir / file
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.info(f"{caption} not found.")
    
    with tab4:
        st.markdown("""
        Seasonal plots (monthly, weekly, hourly) reveal repeating patterns within the data. 
        These insights inform how much the variable depends on time-of-year, which can be vital for accurate modeling 
        and interpretation.
        """)
        plot_names = [
            ("monthly_seasonality.png", "Monthly Seasonality"),
            ("hourly_by_weekday_seasonality.png", "Ts Multivariate Decomposition"),
        ]
        for file, caption in plot_names:
            path = feat_dir / file
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.info(f"{caption} not found.")
    with tab5:
        path = OUTPUTS_ROOT / origin_raw / year / "descriptive_stats.csv"
        df = pd.read_csv(path)[["statistic", f"{feature}_mean"]]
        st.dataframe(df, use_container_width=True)
    
    with tab6:
        path = OUTPUTS_ROOT / origin_raw / year / "correlation_matrix.png"
        st.image(str(path), caption="Correlation Heatmap Pearson", use_container_width=True)

with main_tabs[2]:
    st.subheader(f"Spatial Heatmaps — {feature} ({year}, {origin})")
    spatial_dir = SPATIAL_ROOT / origin_raw / year / feature

    tab1, tab2, tab3 = st.tabs(["Annual Spatial Heatmaps", "Seasonal Heatmaps", "Missing Values Heatmaps"])
    
    with tab1:
        # --- Annual plots ---
        st.subheader("Annual Spatial Heatmaps")
        st.markdown("""
        Annual spatial heatmaps display the mean and standard deviation percentage for each grid cell. 
        They help us quickly spot spatial patterns, persistent hot/cold spots, and areas with data quality issues.
        """)
        mean_map = spatial_dir / f"{feature.lower()}_mean_map.png"
        std_map = spatial_dir / f"{feature.lower()}_std_map.png"
        # missing_map = spatial_dir / "missing_heatmap.png"

        cols = st.columns(2)
        if mean_map.exists():
            st.image(str(mean_map), caption="Mean", use_container_width=True)
        if std_map.exists():
            st.image(str(std_map), caption="Std Dev", use_container_width=True)
        # if missing_map.exists():
        #     st.image(str(missing_map), caption="Missing (%)", use_container_width=True)

    with tab2:
        # --- Seasonal plots ---
        st.subheader("Seasonal Heatmaps")
        st.markdown("""
        Seasonal spatial heatmaps show how the mean and variability of the variable change across locations and seasons. 
        This helps identify regions with strong seasonal effects, guiding both scientific analysis and model development.
        """)
        for s in seasons[1:]:
            st.markdown(f"#### {s.title()}")
            row = st.columns(2)
            mean_season = spatial_dir / f"{feature.lower()}_mean_map_{s}_mean.png"
            std_season = spatial_dir / f"{feature.lower()}_std_map_{s}_mean.png"
            # Plot if exists
            if mean_season.exists():
                row[0].image(str(mean_season), caption=f"Mean {s}", use_container_width=True)
            if std_season.exists():
                row[1].image(str(std_season), caption=f"Std Dev {s}", use_container_width=True)
    
    with tab3:
        # --- Annual plots ---
        st.subheader("Missing Values Heatmaps")
        st.markdown("""
        Missing spatial heatmaps display the missing value percentage for each grid cell. 
        They help us quickly spot areas with missing values.
        """)
        missing_map = spatial_dir / "missing_heatmap.png"

        if missing_map.exists():
            st.image(str(missing_map), caption="Missing (%)", use_container_width=True)

        st.subheader("Seasonal Missing Heatmaps")
        for s in seasons[1:]:
            st.markdown(f"#### {s.title()}")
            miss_season = spatial_dir / f"missing_heatmap_{s}_mean.png"

            if miss_season.exists():
                st.image(str(miss_season), caption=f"Missing {s}", use_container_width=True)

with main_tabs[3]:
    st.subheader(f"Sampled Data — {feature} ({year})")
    sampled_dir = OUTPUTS_ROOT / origin_raw / year
    st.markdown("""
    📊 Sampled Data Overview

    To ensure efficient yet meaningful analytics on large-scale wave forecast data, a 1% stratified sample was extracted from daily hourly prediction files (~460K grid points per day). The plots below summarize the distribution and relationships among key oceanographic and meteorological variables after bias correction.
    """)

    tab1, tab2 = st.tabs(["Descriptive Statistics", "Feature Correlation"])
    with tab1:
        st.markdown("""
        📌 Descriptive Statistics

        This table shows summary statistics (mean, std, min, max, etc.) for each feature across the sampled dataset. It helps assess the scale, variability, and range of each parameter (e.g., wind speed, significant wave height).
        """)
        path = OUTPUTS_ROOT / origin_raw / year / "sampled_descriptive_stats.csv"
        df = pd.read_csv(path)[["statistic", f"{feature}"]]
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.markdown("""
        📈 Correlation Matrix (Pearson)
        
        - The heatmap visualize pairwise correlations between variables:
        - Pearson Correlation captures linear relationships.
        - These help identify strongly correlated features such as:
        - Wind speed (WSPD) with wind components (U10, V10)
        - Corrected wave heights (corrected_VHM0, corrected_VTM02) with original values
        - Temporal encodings like sin_doy and cos_doy with seasonal effects
        """)
        path = OUTPUTS_ROOT / origin_raw / year / "sampled_correlation_pearson.png"
        st.image(str(path), caption="Correlation Heatmap Pearson", use_container_width=True)

with main_tabs[4]:
    # Define your experiment records
    experiments = [
        {
            "Experiment": "Random Regressor - VHM0/VTM02",
            "Model": "RandomRegressor",
            "Dataset": "2021–2022 (Train+Error), 2023 (Eval)",
            "Comet Link": "https://www.comet.com/ioannisgkinis/hcmr-ai/1575e3591c534bc5841bc7cfba07d10c?&prevPath=%2Fioannisgkinis%2Fhcmr-ai%2Fview%2Fnew%2Fpanels"
        },
    ]

    # Convert to DataFrame
    df = pd.DataFrame(experiments)

    # Convert Comet Link to clickable HTML
    df["Comet Link"] = df["Comet Link"].apply(
        lambda url: f'<a href="{url}" target="_blank">🔗 View</a>'
    )
    st.subheader("📊 Experiment Tracker")

    st.write(
        df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

with main_tabs[-1]:
    import streamlit.components.v1 as components

    notion_url = "https://evergreen-seeker-8ec.notion.site/Feature-engineering-doc-23c60e7db91380c39344f0f6d0b30948?pvs=143"

    st.markdown(
        f"""
        ### 📘 View Feature Description Documentation
        [👉 Click here to open in Notion]({notion_url})
        """,
        unsafe_allow_html=True
    )
# --- Footer
st.markdown("---")
st.markdown(
    "[Project Repository](https://github.com/hcmr-ai/Med-WAV)"
)
