import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Superstore Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 30px;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px;
}
.sidebar-header {
    font-size: 1.2rem;
    color: #333;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_and_process_data():
    """Load dan proses data superstore dengan feature engineering"""
    
    # Baca data
    df = pd.read_csv('superstore.csv', encoding='iso-8859-1')
    
    # Data cleaning
    df['Order.Date'] = pd.to_datetime(df['Order.Date'])
    df['Ship.Date'] = pd.to_datetime(df['Ship.Date'])
    df = df.drop_duplicates()
    
    # Simulasi data waktu yang realistis
    np.random.seed(42)
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2024-12-31')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df['Order.Date'] = np.random.choice(date_range, size=len(df))
    
    # Delivery time simulation
    delivery_mapping = {
        'Same Day': np.random.randint(0, 2, size=len(df)),
        'First Class': np.random.randint(1, 4, size=len(df)),
        'Second Class': np.random.randint(2, 6, size=len(df)),
        'Standard Class': np.random.randint(3, 8, size=len(df))
    }
    
    df['Delivery_Days'] = 0
    for mode in df['Ship.Mode'].unique():
        mask = df['Ship.Mode'] == mode
        if mode in delivery_mapping:
            df.loc[mask, 'Delivery_Days'] = delivery_mapping[mode][:mask.sum()]
    
    df['Ship.Date'] = df['Order.Date'] + pd.to_timedelta(df['Delivery_Days'], unit='D')
    
    # Feature Engineering
    # 1. Fitur waktu
    df['Year'] = df['Order.Date'].dt.year
    df['Month'] = df['Order.Date'].dt.month
    df['Quarter'] = df['Order.Date'].dt.quarter
    df['DayOfWeek'] = df['Order.Date'].dt.dayofweek
    df['DayName'] = df['Order.Date'].dt.day_name()
    df['MonthName'] = df['Order.Date'].dt.month_name()
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # 2. Fitur keuangan
    df['Profit_Margin'] = df['Profit'] / df['Sales']
    df['Profit_Margin'] = df['Profit_Margin'].replace([np.inf, -np.inf], 0)
    df['Is_Loss'] = (df['Profit'] < 0).astype(int)
    df['Revenue_per_Quantity'] = df['Sales'] / df['Quantity']
    
    # 3. Fitur pengiriman
    df['Delivery_Time'] = (df['Ship.Date'] - df['Order.Date']).dt.days
    df['Delivery_Efficiency'] = np.where(df['Delivery_Time'] <= 2, 'Fast',
                                        np.where(df['Delivery_Time'] <= 5, 'Medium', 'Slow'))
    
    # 4. Fitur diskon
    def categorize_discount(discount):
        if discount == 0:
            return 'No Discount'
        elif discount <= 0.2:
            return 'Low Discount'
        elif discount <= 0.5:
            return 'Medium Discount'
        else:
            return 'High Discount'
    
    df['Discount_Category'] = df['Discount'].apply(categorize_discount)
    
    # 5. Fitur tambahan
    df['Order_Size'] = np.where(df['Quantity'] >= 5, 'Large',
                               np.where(df['Quantity'] >= 2, 'Medium', 'Small'))
    df['High_Value_Transaction'] = (df['Sales'] > df['Sales'].quantile(0.75)).astype(int)
    df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                   3: 'Spring', 4: 'Spring', 5: 'Spring',
                                   6: 'Summer', 7: 'Summer', 8: 'Summer',
                                   9: 'Fall', 10: 'Fall', 11: 'Fall'})
    
    # 6. RFM Analysis
    reference_date = df['Order.Date'].max()
    rfm = df.groupby('Customer.ID').agg({
        'Order.Date': lambda x: (reference_date - x.max()).days,
        'Order.ID': 'count',
        'Sales': 'sum'
    }).reset_index()
    rfm.columns = ['Customer.ID', 'Recency', 'Frequency', 'Monetary']
    
    # RFM scoring
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    
    # Customer segmentation
    def segment_customers(row):
        score = str(row['R_Score']) + str(row['F_Score']) + str(row['M_Score'])
        if score in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif score in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal Customers'
        elif score in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
            return 'Potential Loyalists'
        elif score in ['512', '511', '422', '421', '412', '411', '311']:
            return 'New Customers'
        elif score in ['155', '154', '144', '214', '215', '115', '114']:
            return 'At Risk'
        else:
            return 'Others'
    
    rfm['Customer_Segment'] = rfm.apply(segment_customers, axis=1)
    df = df.merge(rfm[['Customer.ID', 'Customer_Segment']], on='Customer.ID', how='left')
    
    return df

# Fungsi untuk training model
@st.cache_data
def train_models(df):
    """Training multiple models dan return hasil perbandingan"""
    
    # Prepare data
    df_model = df.copy()
    
    # Encode categorical variables
    categorical_cols = ['Category', 'Ship.Mode', 'Segment', 'Region', 'Customer_Segment',
                       'Delivery_Efficiency', 'Discount_Category', 'Order_Size', 'Season']
    
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col + '_encoded'] = le.fit_transform(df_model[col].astype(str))
    
    # Select features
    feature_cols = ['Quantity', 'Discount', 'Profit_Margin', 'Year', 'Month', 'Quarter',
                   'DayOfWeek', 'IsWeekend', 'Delivery_Time', 'High_Value_Transaction',
                   'Category_encoded', 'Ship.Mode_encoded', 'Segment_encoded', 'Region_encoded',
                   'Customer_Segment_encoded', 'Delivery_Efficiency_encoded', 
                   'Discount_Category_encoded', 'Order_Size_encoded', 'Season_encoded']
    
    available_features = [col for col in feature_cols if col in df_model.columns]
    X = df_model[available_features].fillna(0)
    y = df_model['Sales']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': np.sqrt(mse),
            'R2': r2,
            'predictions': y_pred
        }
    
    return results, y_test, available_features

# Main App
def main():
    st.markdown('<h1 class="main-header">📊 Superstore Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_and_process_data()
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">📋 Navigation</div>', unsafe_allow_html=True)
    
    pages = [
        "📈 Overview & KPIs",
        "🔍 Data Exploration",
        "📊 Sales Analysis",
        "🎯 Customer Segmentation",
        "🚚 Delivery Analysis",
        "🤖 Machine Learning",
        "📋 Data Summary"
    ]
    
    selected_page = st.sidebar.selectbox("Select Page", pages)
    
    # Filters
    st.sidebar.markdown("### 🔧 Filters")
    
    # Date filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Order.Date'].min(), df['Order.Date'].max()),
        min_value=df['Order.Date'].min(),
        max_value=df['Order.Date'].max()
    )
    
    # Category filter
    categories = st.sidebar.multiselect(
        "Select Categories",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )
    
    # Apply filters
    mask = (
        (df['Order.Date'].dt.date >= date_range[0]) & 
        (df['Order.Date'].dt.date <= date_range[1]) &
        (df['Category'].isin(categories)) &
        (df['Region'].isin(regions))
    )
    filtered_df = df[mask]
    
    # Page content
    if selected_page == "📈 Overview & KPIs":
        show_overview(filtered_df)
    elif selected_page == "🔍 Data Exploration":
        show_data_exploration(filtered_df)
    elif selected_page == "📊 Sales Analysis":
        show_sales_analysis(filtered_df)
    elif selected_page == "🎯 Customer Segmentation":
        show_customer_segmentation(filtered_df)
    elif selected_page == "🚚 Delivery Analysis":
        show_delivery_analysis(filtered_df)
    elif selected_page == "🤖 Machine Learning":
        show_machine_learning(filtered_df)
    elif selected_page == "📋 Data Summary":
        show_data_summary(filtered_df)

def show_overview(df):
    st.header("📈 Business Overview & KPIs")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Sales",
            value=f"${df['Sales'].sum():,.0f}",
            delta=f"{df['Sales'].sum() / 1000000:.1f}M"
        )
    
    with col2:
        st.metric(
            label="📈 Total Profit",
            value=f"${df['Profit'].sum():,.0f}",
            delta=f"{df['Profit_Margin'].mean():.1%} margin"
        )
    
    with col3:
        st.metric(
            label="🛒 Total Orders",
            value=f"{df['Order.ID'].nunique():,}",
            delta=f"{df['Quantity'].sum():,} items"
        )
    
    with col4:
        st.metric(
            label="👥 Customers",
            value=f"{df['Customer.ID'].nunique():,}",
            delta=f"{df.groupby('Customer.ID')['Sales'].sum().mean():.0f} avg/customer"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales trend
        monthly_sales = df.groupby(df['Order.Date'].dt.to_period('M'))['Sales'].sum()
        fig = px.line(
            x=monthly_sales.index.astype(str),
            y=monthly_sales.values,
            title="📈 Monthly Sales Trend"
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Sales ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category performance
        category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        fig = px.bar(
            x=category_sales.index,
            y=category_sales.values,
            title="💼 Sales by Category"
        )
        fig.update_layout(xaxis_title="Category", yaxis_title="Sales ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional performance
    st.subheader("🌍 Regional Performance")
    regional_metrics = df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order.ID': 'nunique',
        'Customer.ID': 'nunique'
    }).round(2)
    
    regional_metrics.columns = ['Sales ($)', 'Profit ($)', 'Orders', 'Customers']
    regional_metrics['Profit Margin (%)'] = (regional_metrics['Profit ($)'] / regional_metrics['Sales ($)'] * 100).round(1)
    
    st.dataframe(regional_metrics, use_container_width=True)

def show_data_exploration(df):
    st.header("🔍 Data Exploration")
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Information")
        st.write(f"**Total Rows:** {len(df):,}")
        st.write(f"**Total Columns:** {len(df.columns)}")
        st.write(f"**Date Range:** {df['Order.Date'].min().strftime('%Y-%m-%d')} to {df['Order.Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
    
    with col2:
        st.subheader("📈 Statistical Summary")
        st.dataframe(df[['Sales', 'Profit', 'Quantity', 'Discount']].describe())
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(20))
    
    # Column analysis
    st.subheader("📋 Column Analysis")
    
    analysis_col = st.selectbox("Select column to analyze", df.columns)
    
    if df[analysis_col].dtype in ['int64', 'float64']:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=analysis_col, title=f"Distribution of {analysis_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=analysis_col, title=f"Box Plot of {analysis_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        value_counts = df[analysis_col].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Value Counts of {analysis_col}")
        st.plotly_chart(fig, use_container_width=True)

def show_sales_analysis(df):
    st.header("📊 Sales Analysis")
    
    # Time series analysis
    st.subheader("⏰ Time Series Analysis")
    
    time_granularity = st.selectbox("Select time granularity", ["Daily", "Weekly", "Monthly", "Quarterly"])
    
    if time_granularity == "Daily":
        time_series = df.groupby(df['Order.Date'].dt.date)['Sales'].sum()
    elif time_granularity == "Weekly":
        time_series = df.groupby(df['Order.Date'].dt.to_period('W'))['Sales'].sum()
    elif time_granularity == "Monthly":
        time_series = df.groupby(df['Order.Date'].dt.to_period('M'))['Sales'].sum()
    else:  # Quarterly
        time_series = df.groupby(df['Order.Date'].dt.to_period('Q'))['Sales'].sum()
    
    fig = px.line(
        x=time_series.index.astype(str),
        y=time_series.values,
        title=f"{time_granularity} Sales Trend"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    col1, col2 = st.columns(2)
    
    with col1:
        seasonal_sales = df.groupby('Season')['Sales'].sum()
        fig = px.pie(values=seasonal_sales.values, names=seasonal_sales.index, title="🌸 Seasonal Sales Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        day_sales = df.groupby('DayName')['Sales'].sum()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_sales = day_sales.reindex(day_order)
        fig = px.bar(x=day_sales.index, y=day_sales.values, title="📅 Sales by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    
    # Product analysis
    st.subheader("🛍️ Product Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_products = df.groupby('Product.Name')['Sales'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            title="🏆 Top 10 Products by Sales"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        category_profit = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
        fig = px.bar(x=category_profit.index, y=category_profit.values, title="💰 Profit by Category")
        st.plotly_chart(fig, use_container_width=True)
    
    # Discount analysis
    st.subheader("🎯 Discount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        discount_sales = df.groupby('Discount_Category')['Sales'].mean()
        fig = px.bar(x=discount_sales.index, y=discount_sales.values, title="📊 Average Sales by Discount Category")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='Discount', y='Sales', title="🔍 Discount vs Sales Correlation")
        st.plotly_chart(fig, use_container_width=True)

def show_customer_segmentation(df):
    st.header("🎯 Customer Segmentation")
    
    # Customer segment overview
    if 'Customer_Segment' in df.columns:
        segment_counts = df['Customer_Segment'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=segment_counts.values, names=segment_counts.index, title="👥 Customer Segments")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            segment_value = df.groupby('Customer_Segment')['Sales'].sum().sort_values(ascending=False)
            fig = px.bar(x=segment_value.index, y=segment_value.values, title="💰 Sales by Customer Segment")
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment analysis table
        st.subheader("📊 Segment Analysis")
        segment_analysis = df.groupby('Customer_Segment').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Profit': ['sum', 'mean'],
            'Customer.ID': 'nunique'
        }).round(2)
        
        segment_analysis.columns = ['Total Sales', 'Avg Sales', 'Transactions', 'Total Profit', 'Avg Profit', 'Customers']
        st.dataframe(segment_analysis)
    
    # Top customers
    st.subheader("🏆 Top Customers")
    top_customers = df.groupby('Customer.Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order.ID': 'nunique'
    }).sort_values('Sales', ascending=False).head(10)
    
    top_customers.columns = ['Total Sales', 'Total Profit', 'Orders']
    st.dataframe(top_customers)

def show_delivery_analysis(df):
    st.header("🚚 Delivery Analysis")
    
    # Delivery performance
    col1, col2 = st.columns(2)
    
    with col1:
        delivery_mode = df.groupby('Ship.Mode')['Delivery_Time'].mean().sort_values()
        fig = px.bar(x=delivery_mode.index, y=delivery_mode.values, title="⏱️ Average Delivery Time by Mode")
        fig.update_layout(yaxis_title="Days")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        efficiency_counts = df['Delivery_Efficiency'].value_counts()
        fig = px.pie(values=efficiency_counts.values, names=efficiency_counts.index, title="📊 Delivery Efficiency Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional delivery analysis
    st.subheader("🌍 Regional Delivery Performance")
    regional_delivery = df.groupby('Region').agg({
        'Delivery_Time': 'mean',
        'Sales': 'sum'
    }).round(2)
    
    regional_delivery.columns = ['Avg Delivery Time (Days)', 'Total Sales']
    st.dataframe(regional_delivery)
    
    # Delivery time distribution
    st.subheader("📈 Delivery Time Distribution")
    fig = px.histogram(df, x='Delivery_Time', nbins=20, title="Distribution of Delivery Times")
    fig.update_layout(xaxis_title="Delivery Time (Days)", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

def show_machine_learning(df):
    st.header("🤖 Machine Learning Models")
    
    with st.spinner('Training models...'):
        results, y_test, features = train_models(df)
    
    # Model comparison
    st.subheader("📊 Model Performance Comparison")
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE': [results[model]['RMSE'] for model in results.keys()],
        'R²': [results[model]['R2'] for model in results.keys()]
    }).sort_values('R²', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(comparison_df, x='Model', y='R²', title="Model R² Scores")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(comparison_df, x='Model', y='RMSE', title="Model RMSE")
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model details
    best_model = comparison_df.iloc[0]['Model']
    st.success(f"🏆 Best Model: {best_model}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R² Score", f"{results[best_model]['R2']:.4f}")
    
    with col2:
        st.metric("RMSE", f"${results[best_model]['RMSE']:,.0f}")
    
    # Prediction vs Actual
    st.subheader("🎯 Prediction vs Actual")
    
    best_predictions = results[best_model]['predictions']
    fig = px.scatter(x=y_test, y=best_predictions, title=f"Predictions vs Actual - {best_model}")
    fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(dash="dash"))
    fig.update_layout(xaxis_title="Actual Sales", yaxis_title="Predicted Sales")
    st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison table
    st.subheader("📋 Detailed Model Comparison")
    st.dataframe(comparison_df.round(4))

def show_data_summary(df):
    st.header("📋 Data Summary")
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    insights = [
        f"📊 Total dataset contains {len(df):,} transactions",
        f"💰 Total sales revenue: ${df['Sales'].sum():,.0f}",
        f"📈 Total profit: ${df['Profit'].sum():,.0f}",
        f"🎯 Average profit margin: {df['Profit_Margin'].mean():.1%}",
        f"👥 Unique customers: {df['Customer.ID'].nunique():,}",
        f"🛍️ Unique products: {df['Product.ID'].nunique():,}",
        f"🏆 Best performing category: {df.groupby('Category')['Sales'].sum().idxmax()}",
        f"🌟 Best performing region: {df.groupby('Region')['Sales'].sum().idxmax()}",
        f"⏱️ Average delivery time: {df['Delivery_Time'].mean():.1f} days",
        f"🎁 Average discount: {df['Discount'].mean():.1%}"
    ]
    
    for insight in insights:
        st.write(insight)
    
    # Feature engineering summary
    st.subheader("🔧 Feature Engineering Summary")
    
    engineered_features = [
        "Time-based features: Year, Month, Quarter, DayOfWeek, Season",
        "Financial metrics: Profit Margin, Revenue per Quantity",
        "Delivery metrics: Delivery Time, Delivery Efficiency",
        "Customer segmentation: RFM Analysis",
        "Product categorization: Order Size, Discount Category",
        "Business indicators: High Value Transaction, Weekend flag"
    ]
    
    for feature in engineered_features:
        st.write(f"• {feature}")
    
    # Data quality report
    st.subheader("✅ Data Quality Report")
    
    quality_report = {
        'Total Records': len(df),
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Records': df.duplicated().sum(),
        'Data Completeness': f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%"
    }
    
    for metric, value in quality_report.items():
        st.write(f"**{metric}:** {value}")

if __name__ == "__main__":
    main