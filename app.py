import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat data
@st.cache_data
def load_and_process_data():
    """Load dan proses data superstore"""
    try:
        # Baca data
        df = pd.read_csv('superstore.csv', encoding='iso-8859-1')
        
        # Basic cleaning
        df['Order.Date'] = pd.to_datetime(df['Order.Date'])
        df['Ship.Date'] = pd.to_datetime(df['Ship.Date'])
        df = df.drop_duplicates()
        
        # Feature engineering basic
        df['Year'] = df['Order.Date'].dt.year
        df['Month'] = df['Order.Date'].dt.month
        df['Quarter'] = df['Order.Date'].dt.quarter
        df['DayOfWeek'] = df['Order.Date'].dt.dayofweek
        df['DayName'] = df['Order.Date'].dt.day_name()
        df['MonthName'] = df['Order.Date'].dt.month_name()
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Financial metrics
        df['Profit_Margin'] = df['Profit'] / df['Sales']
        df['Profit_Margin'] = df['Profit_Margin'].replace([np.inf, -np.inf], 0)
        df['Is_Loss'] = (df['Profit'] < 0).astype(int)
        df['Revenue_per_Quantity'] = df['Sales'] / df['Quantity']
        
        # Delivery time
        df['Delivery_Time'] = (df['Ship.Date'] - df['Order.Date']).dt.days
        
        # Discount categories
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
        
        # Order size
        df['Order_Size'] = np.where(df['Quantity'] >= 5, 'Large',
                                   np.where(df['Quantity'] >= 2, 'Medium', 'Small'))
        
        # Season
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def show_overview(df):
    """Halaman overview"""
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
            delta=f"{df.groupby('Customer.ID')['Sales'].sum().mean():.0f} avg"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by Category
        category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        fig = px.bar(
            x=category_sales.index,
            y=category_sales.values,
            title="💼 Sales by Category",
            color=category_sales.values,
            color_continuous_scale="Blues"
        )
        fig.update_layout(xaxis_title="Category", yaxis_title="Sales ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales by Region
        region_sales = df.groupby('Region')['Sales'].sum()
        fig = px.pie(
            values=region_sales.values,
            names=region_sales.index,
            title="🌍 Sales by Region"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_sales_analysis(df):
    """Halaman analisis sales"""
    st.header("📊 Sales Analysis")
    
    # Time series analysis
    st.subheader("⏰ Time Series Analysis")
    
    time_granularity = st.selectbox("Select time granularity", ["Daily", "Weekly", "Monthly", "Quarterly"])
    
    if time_granularity == "Daily":
        time_series = df.groupby(df['Order.Date'].dt.date)['Sales'].sum()
        x_title = "Date"
    elif time_granularity == "Weekly":
        time_series = df.groupby(df['Order.Date'].dt.to_period('W'))['Sales'].sum()
        x_title = "Week"
    elif time_granularity == "Monthly":
        time_series = df.groupby(df['Order.Date'].dt.to_period('M'))['Sales'].sum()
        x_title = "Month"
    else:  # Quarterly
        time_series = df.groupby(df['Order.Date'].dt.to_period('Q'))['Sales'].sum()
        x_title = "Quarter"
    
    fig = px.line(
        x=time_series.index.astype(str),
        y=time_series.values,
        title=f"{time_granularity} Sales Trend"
    )
    fig.update_layout(xaxis_title=x_title, yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal and day analysis
    col1, col2 = st.columns(2)
    
    with col1:
        seasonal_sales = df.groupby('Season')['Sales'].sum()
        fig = px.bar(
            x=seasonal_sales.index,
            y=seasonal_sales.values,
            title="🌸 Seasonal Sales",
            color=seasonal_sales.values,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        day_sales = df.groupby('DayName')['Sales'].sum()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_sales = day_sales.reindex(day_order)
        fig = px.bar(
            x=day_sales.index,
            y=day_sales.values,
            title="📅 Sales by Day of Week",
            color=day_sales.values,
            color_continuous_scale="Plasma"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_customer_analysis(df):
    """Halaman analisis customer"""
    st.header("🎯 Customer Analysis")
    
    # Customer metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Top customers
        top_customers = df.groupby('Customer.Name').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order.ID': 'nunique'
        }).sort_values('Sales', ascending=False).head(10)
        
        fig = px.bar(
            x=top_customers['Sales'],
            y=top_customers.index,
            orientation='h',
            title="🏆 Top 10 Customers by Sales"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer segment analysis
        segment_sales = df.groupby('Segment')['Sales'].sum()
        fig = px.pie(
            values=segment_sales.values,
            names=segment_sales.index,
            title="👥 Sales by Customer Segment"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer behavior
    st.subheader("📊 Customer Behavior Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Order size distribution
        order_size_counts = df['Order_Size'].value_counts()
        fig = px.bar(
            x=order_size_counts.index,
            y=order_size_counts.values,
            title="📦 Order Size Distribution",
            color=order_size_counts.values,
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Discount impact
        discount_impact = df.groupby('Discount_Category')['Sales'].mean()
        fig = px.bar(
            x=discount_impact.index,
            y=discount_impact.values,
            title="🎯 Average Sales by Discount Category",
            color=discount_impact.values,
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_product_analysis(df):
    """Halaman analisis produk"""
    st.header("🛍️ Product Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top products
        top_products = df.groupby('Product.Name')['Sales'].sum().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            title="🏆 Top 15 Products by Sales"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sub-category performance
        subcat_sales = df.groupby('Sub.Category')['Sales'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=subcat_sales.index,
            y=subcat_sales.values,
            title="📊 Top 10 Sub-Categories by Sales"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Profitability analysis
    st.subheader("💰 Profitability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Profit margin by category
        profit_margin_cat = df.groupby('Category')['Profit_Margin'].mean()
        fig = px.bar(
            x=profit_margin_cat.index,
            y=profit_margin_cat.values,
            title="📈 Average Profit Margin by Category",
            color=profit_margin_cat.values,
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(yaxis_title="Profit Margin")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales vs Profit scatter
        fig = px.scatter(
            df,
            x='Sales',
            y='Profit',
            color='Category',
            title="💹 Sales vs Profit by Category",
            hover_data=['Product.Name']
        )
        st.plotly_chart(fig, use_container_width=True)

def show_data_summary(df):
    """Halaman summary data"""
    st.header("📋 Data Summary")
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("📊 **Dataset Overview:**")
        st.write(f"• Total records: {len(df):,}")
        st.write(f"• Date range: {df['Order.Date'].min().strftime('%Y-%m-%d')} to {df['Order.Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"• Unique customers: {df['Customer.ID'].nunique():,}")
        st.write(f"• Unique products: {df['Product.ID'].nunique():,}")
        st.write(f"• Missing values: {df.isnull().sum().sum()}")
    
    with col2:
        st.write("💰 **Business Metrics:**")
        st.write(f"• Total sales: ${df['Sales'].sum():,.0f}")
        st.write(f"• Total profit: ${df['Profit'].sum():,.0f}")
        st.write(f"• Average profit margin: {df['Profit_Margin'].mean():.1%}")
        st.write(f"• Loss transactions: {df['Is_Loss'].mean():.1%}")
        st.write(f"• Average delivery time: {df['Delivery_Time'].mean():.1f} days")
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10))
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df[['Sales', 'Profit', 'Quantity', 'Discount', 'Profit_Margin']].describe())

# Fungsi utama
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Superstore Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_and_process_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check if 'superstore.csv' exists.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.markdown("### 📋 Navigation")
    
    pages = [
        "📈 Overview",
        "📊 Sales Analysis",
        "🎯 Customer Analysis",
        "🛍️ Product Analysis",
        "📋 Data Summary"
    ]
    
    selected_page = st.sidebar.selectbox("Select Page", pages)
    
    # Filters
    st.sidebar.markdown("### 🔧 Filters")
    
    # Date filter
    min_date = df['Order.Date'].min().date()
    max_date = df['Order.Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
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
    if len(date_range) == 2:
        mask = (
            (df['Order.Date'].dt.date >= date_range[0]) & 
            (df['Order.Date'].dt.date <= date_range[1]) &
            (df['Category'].isin(categories)) &
            (df['Region'].isin(regions))
        )
        filtered_df = df[mask]
    else:
        filtered_df = df[
            (df['Category'].isin(categories)) &
            (df['Region'].isin(regions))
        ]
    
    # Show data info
    st.sidebar.markdown("### 📊 Data Info")
    st.sidebar.write(f"Total records: {len(filtered_df):,}")
    st.sidebar.write(f"Date range: {len(date_range)} days" if len(date_range) == 2 else "Date range: All")
    
    # Page routing
    if selected_page == "📈 Overview":
        show_overview(filtered_df)
    elif selected_page == "📊 Sales Analysis":
        show_sales_analysis(filtered_df)
    elif selected_page == "🎯 Customer Analysis":
        show_customer_analysis(filtered_df)
    elif selected_page == "🛍️ Product Analysis":
        show_product_analysis(filtered_df)
    elif selected_page == "📋 Data Summary":
        show_data_summary(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | Superstore Analytics Dashboard")

# Run the app
if __name__ == "__main__":
    main()
