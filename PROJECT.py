import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    confusion_matrix, 
    classification_report
)

# =============================================
# 🎨 CUSTOM STYLING
# =============================================
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #1e1e1e;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #282828 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    
    /* Text and headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background-color: #333333;
        color: white;
    }
    
    /* Slider */
    .stSlider .thumb {
        background-color: #4CAF50 !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================
# 🏗️ APP STRUCTURE
# =============================================

# 🎯 App Header
st.title("🚦 Traffic Data Analysis Dashboard")
st.markdown("""
    <div style="background-color: #282828; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        Explore traffic patterns, peak hours, holiday effects, and more with this interactive dashboard.
    </div>
""", unsafe_allow_html=True)

# 📂 Sidebar - File Upload
with st.sidebar:
    st.header("📂 Data Upload & Cleaning")
    st.markdown("---")
    
    # File uploader with improved styling
    uploaded_file = st.file_uploader(
        "Upload your traffic data (CSV)",
        type=["csv"],
        help="Please upload a CSV file containing traffic data with DateTime and Vehicles columns"
    )

# =============================================
# 📊 MAIN CONTENT - WHEN FILE UPLOADED
# =============================================
if uploaded_file:
    # Load data with progress indicator
    with st.spinner("Loading and processing data..."):
        traffic_data = pd.read_csv(uploaded_file)
    
    # 🔧 Data Cleaning Section
    with st.sidebar.expander("🔧 Data Cleaning Options", expanded=True):
        # DateTime conversion
        try:
            traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'], errors='coerce')
            st.success("✅ DateTime conversion successful!")
        except Exception as e:
            st.error(f"❌ Error converting DateTime: {e}")
        
        # Missing values handling
        initial_rows = traffic_data.shape[0]
        traffic_data.dropna(subset=['DateTime', 'Vehicles'], inplace=True)
        cleaned_rows = traffic_data.shape[0]
        st.info(f"🧹 Removed {initial_rows - cleaned_rows} rows with missing data")
        
        # Duplicates removal
        if st.checkbox("Remove duplicate rows", value=True):
            initial_rows = traffic_data.shape[0]
            traffic_data.drop_duplicates(inplace=True)
            st.success(f"♻️ Removed {initial_rows - traffic_data.shape[0]} duplicates")
        
        # Data validation
        if 'Vehicles' in traffic_data.columns:
            traffic_data = traffic_data[traffic_data['Vehicles'] >= 0]
            st.success("🚦 Removed negative vehicle counts")
    
    st.sidebar.success("✅ Data cleaning complete!")

    # 📑 Tab System
    tab1, tab2, tab3 = st.tabs([
        "📊 General Analysis", 
        "📅 Time Series", 
        "⚙️ Predictive Models"
    ])

    # =========================================
    # 📊 TAB 1: GENERAL ANALYSIS
    # =========================================
    with tab1:
        st.header("📊 Traffic Pattern Analysis")
        st.write("Explore fundamental traffic patterns and distributions")
        
        analysis_options = st.multiselect(
            "Select analyses to display:",
            options=[
                "Peak Hours",
                "Holiday Effects",
                "Traffic by Junction",
                "Traffic Volume Distribution"
            ],
            default=["Peak Hours", "Traffic by Junction"]
        )
        
        # 📈 Peak Hours Analysis
        if "Peak Hours" in analysis_options:
            with st.expander("⏰ Peak Hour Analysis", expanded=True):
                traffic_data['Hour'] = traffic_data['DateTime'].dt.hour
                hourly_traffic = traffic_data.groupby('Hour')['Vehicles'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    x='Hour', 
                    y='Vehicles', 
                    data=hourly_traffic, 
                    ax=ax,
                    palette="viridis"
                )
                ax.set_title('Average Traffic Volume by Hour', pad=20)
                ax.set_xlabel('Hour of Day', labelpad=10)
                ax.set_ylabel('Average Vehicles', labelpad=10)
                plt.tight_layout()
                st.pyplot(fig)
        
        # 🎉 Holiday Effects
        if "Holiday Effects" in analysis_options:
            with st.expander("🎉 Holiday Effects", expanded=True):
                public_holidays = st.text_input(
                    "Enter holidays (comma-separated dates):",
                    value="2016-01-01,2016-07-04,2016-12-25",
                    help="Format: YYYY-MM-DD,YYYY-MM-DD,..."
                )
                
                try:
                    public_holidays = pd.to_datetime(public_holidays.split(","))
                    traffic_data['IsHoliday'] = traffic_data['DateTime'].dt.normalize().isin(public_holidays)
                    holiday_traffic = traffic_data.groupby('IsHoliday')['Vehicles'].mean().reset_index()
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(
                        x='IsHoliday', 
                        y='Vehicles', 
                        data=holiday_traffic, 
                        ax=ax,
                        palette="mako"
                    )
                    ax.set_title("Holiday vs Non-Holiday Traffic", pad=15)
                    ax.set_xlabel("Is Holiday?", labelpad=10)
                    ax.set_ylabel("Average Vehicles", labelpad=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error processing holidays: {e}")
        
        # 🏗️ Traffic by Junction
        if "Traffic by Junction" in analysis_options:
            with st.expander("🏗️ Traffic by Junction", expanded=True):
                if 'Junction' in traffic_data.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(
                        x='Junction', 
                        y='Vehicles', 
                        data=traffic_data, 
                        ax=ax,
                        palette="rocket"
                    )
                    ax.set_title("Traffic Distribution by Junction", pad=15)
                    ax.set_xlabel("Junction ID", labelpad=10)
                    ax.set_ylabel("Number of Vehicles", labelpad=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Junction data not found in the dataset")
        
        # 📊 Traffic Distribution
        if "Traffic Volume Distribution" in analysis_options:
            with st.expander("📈 Traffic Volume Distribution", expanded=True):
                bins = st.slider(
                    "Select histogram bins:",
                    min_value=10,
                    max_value=50,
                    value=30,
                    help="Adjust the granularity of the distribution"
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(
                    traffic_data['Vehicles'], 
                    kde=True,
                    bins=bins, 
                    ax=ax,
                    color='#4CAF50'
                )
                ax.set_title('Traffic Volume Distribution', pad=15)
                ax.set_xlabel('Number of Vehicles', labelpad=10)
                ax.set_ylabel('Frequency', labelpad=10)
                plt.tight_layout()
                st.pyplot(fig)

    # =========================================
    # 📅 TAB 2: TIME SERIES ANALYSIS
    # =========================================
    with tab2:
        st.header("📅 Time Series Analysis")
        st.write("Examine temporal patterns and seasonal trends")
        
        # 📆 Traffic Over Time
        with st.expander("🛣️ Traffic Over Time", expanded=True):
            try:
                traffic_data_resampled = traffic_data.set_index('DateTime').resample('D').mean()
                
                fig, ax = plt.subplots(figsize=(14, 6))
                sns.lineplot(
                    data=traffic_data_resampled, 
                    x=traffic_data_resampled.index, 
                    y='Vehicles',
                    ax=ax,
                    color='#4CAF50',
                    linewidth=2
                )
                ax.set_title('Daily Traffic Volume Trend', pad=15)
                ax.set_xlabel('Date', labelpad=10)
                ax.set_ylabel('Average Vehicles', labelpad=10)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating time series: {e}")
        
        # 🔍 Time Series Decomposition
        with st.expander("🔍 Seasonal Decomposition", expanded=True):
            if st.checkbox("Perform decomposition (requires ≥14 days data)"):
                try:
                    traffic_data_daily = traffic_data.set_index('DateTime').resample('D').mean()
                    
                    if len(traffic_data_daily) >= 14:
                        decomposition = seasonal_decompose(
                            traffic_data_daily['Vehicles'], 
                            model='additive', 
                            period=7
                        )
                        
                        fig = decomposition.plot()
                        fig.set_size_inches(12, 8)
                        fig.suptitle('Time Series Decomposition', y=1.02)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Insufficient data for decomposition (need ≥14 days)")
                except Exception as e:
                    st.error(f"Decomposition failed: {e}")

    # =========================================
    # ⚙️ TAB 3: PREDICTIVE MODELS
    # =========================================
    with tab3:
        st.header("⚙️ Predictive Modeling")
        st.write("Build and evaluate traffic prediction models")
        
        # 📈 Linear Regression
        with st.expander("📈 Traffic Volume Prediction", expanded=True):
            if st.checkbox("Run Linear Regression", key="lin_reg"):
                try:
                    # Feature engineering
                    traffic_data['Hour'] = traffic_data['DateTime'].dt.hour
                    traffic_data['DayOfWeek'] = traffic_data['DateTime'].dt.dayofweek
                    
                    # Model setup
                    X = traffic_data[['Hour', 'DayOfWeek']]
                    y = traffic_data['Vehicles']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Model training
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Results display
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
                    with col2:
                        st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.1f}")
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test, y_pred, alpha=0.6, color='#4CAF50')
                    ax.plot(
                        [y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        color='red', 
                        linestyle='--',
                        linewidth=2
                    )
                    ax.set_title("Actual vs Predicted Traffic", pad=15)
                    ax.set_xlabel("Actual Vehicles", labelpad=10)
                    ax.set_ylabel("Predicted Vehicles", labelpad=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Regression failed: {e}")
        
        # 🔍 Holiday Classification
        with st.expander("🔍 Holiday Traffic Classifier", expanded=True):
            if st.checkbox("Run Holiday Classifier", key="log_reg"):
                if 'IsHoliday' in traffic_data.columns:
                    try:
                        # Feature engineering
                        traffic_data['Hour'] = traffic_data['DateTime'].dt.hour
                        traffic_data['DayOfWeek'] = traffic_data['DateTime'].dt.dayofweek
                        
                        # Model setup
                        X = traffic_data[['Hour', 'DayOfWeek', 'Vehicles']]
                        y = traffic_data['IsHoliday'].astype(int)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Model training
                        model = LogisticRegression(max_iter=1000)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Results display
                        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Confusion Matrix:**")
                            fig1, ax1 = plt.subplots(figsize=(5, 4))
                            sns.heatmap(
                                confusion_matrix(y_test, y_pred), 
                                annot=True, 
                                cmap='Blues', 
                                fmt='d',
                                ax=ax1
                            )
                            ax1.set_title("Confusion Matrix", pad=10)
                            ax1.set_xlabel("Predicted")
                            ax1.set_ylabel("Actual")
                            st.pyplot(fig1)
                        
                        with col2:
                            st.write("**Classification Report:**")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.style.highlight_max(axis=0))
                    except Exception as e:
                        st.error(f"Classification failed: {e}")
                else:
                    st.warning("Please enable Holiday Effects analysis first")

# =============================================
# 🏁 FOOTER
# =============================================
st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 15px; border-top: 1px solid #444;">
        <p>Designed with ❤️ by Raunak and Shobhit</p>
    </div>
""", unsafe_allow_html=True)