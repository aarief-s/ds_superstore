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
    layout="wide"
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
</style>
""", unsafe_allow_html=True)

def load_and_process_data(uploaded_file=None):
    """Load dan proses data superstore"""
    try:
        # Jika ada file upload, gunakan itu
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
        else:
            # Coba berbagai nama file yang mungkin
            try:
                df = pd.read_csv('superstore.csv', encoding='iso-8859-1')
            except:
                try:
                    df = pd.read_csv('superstore.csv', encoding='utf-8')
                except:
                    return None
        
        # Data cleaning dan processing
        df['Order.Date'] = pd.to_datetime(df['Order.Date'])
        df['Ship.Date'] = pd.to_datetime(df['Ship.Date'])
        df = df.drop_duplicates()
        
        # Feature engineering
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
        st.error(f"Error loading data: {str(e)}")
        return None

def create_sample_data():
    """Buat sample data jika file tidak tersedia"""
    np.random.seed(42)
    
    # Generate sample data
    n_rows = 1000
    
    categories = ['Technology', 'Furniture', 'Office Supplies']
    regions = ['East', 'West', 'Central', 'South']
    segments = ['Consumer', 'Corporate', 'Home Office']
    ship_modes = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
    
    data = {
        'Order.Date': pd.date_range('2021-01-01', '2024-12-31', periods=n_rows),
        'Ship.Date': pd.date_range('2021-01-01', '2024-12-31', periods=n_rows),
        'Category': np.random.choice(categories, n_rows),
        'Sub.Category': np.random.choice(['Phones', 'Chairs', 'Storage', 'Tables', 'Accessories'], n_rows),
        'Product.Name': [f'Product_{i}' for i in range(n_rows)],
        'Sales': np.random.uniform(10, 10000, n_rows),
        'Quantity': np.random.randint(1, 10, n_rows),
        'Discount': np.random.uniform(0, 0.8, n_rows),
        'Profit': np.random.uniform(-1000, 3000, n_rows),
        'Customer.ID': [f'CU-{i:05d}' for i in np.random.randint(1, 100, n_rows)],
        'Customer.Name': [f'Customer_{i}' for i in np.random.randint(1, 100, n_rows)],
        'Segment': np.random.choice(segments, n_rows),
        'Region': np.random.choice(regions, n_rows),
        'State': np.random.choice(['California', 'New York', 'Texas', 'Florida'], n_rows),
        'Ship.Mode': np.random.choice(ship_modes, n_rows),
        'Order.ID': [f'CA-{i:05d}' for i in range(n_rows)],
        'Product.ID': [f'TEC-{i:05d}' for i in range(n_rows)]
    }
    
    df = pd.DataFrame(data)
    return load_and_process_data_sample(df)

def load_and_process_data_sample(df):
    """Process sample data"""
    # Feature engineering untuk sample data
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
    
    # Monthly trend
    st.subheader("📈 Monthly Sales Trend")
    monthly_sales = df.groupby(df['Order.Date'].dt.to_period('M'))['Sales'].sum()
    fig = px.line(
        x=monthly_sales.index.astype(str),
        y=monthly_sales.values,
        title="Monthly Sales Trend"
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

def show_sales_analysis(df):
    """Halaman analisis sales"""
    st.header("📊 Sales Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top products
        top_products = df.groupby('Product.Name')['Sales'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            title="🏆 Top 10 Products by Sales"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Seasonal analysis
        seasonal_sales = df.groupby('Season')['Sales'].sum()
        fig = px.bar(
            x=seasonal_sales.index,
            y=seasonal_sales.values,
            title="🌸 Seasonal Sales Distribution",
            color=seasonal_sales.values,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Discount analysis
    st.subheader("🎯 Discount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        discount_sales = df.groupby('Discount_Category')['Sales'].mean()
        fig = px.bar(
            x=discount_sales.index,
            y=discount_sales.values,
            title="📊 Average Sales by Discount Category",
            color=discount_sales.values,
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df.sample(min(500, len(df))),  # Sample untuk performance
            x='Discount',
            y='Sales',
            color='Category',
            title="🔍 Discount vs Sales Correlation"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_customer_analysis(df):
    """Halaman analisis customer"""
    st.header("🎯 Customer Analysis")
    
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
    
    # Customer behavior table
    st.subheader("📊 Customer Segment Analysis")
    segment_analysis = df.groupby('Segment').agg({
        'Sales': ['sum', 'mean', 'count'],
        'Profit': ['sum', 'mean'],
        'Customer.ID': 'nunique'
    }).round(2)
    
    segment_analysis.columns = ['Total Sales', 'Avg Sales', 'Transactions', 'Total Profit', 'Avg Profit', 'Customers']
    st.dataframe(segment_analysis, use_container_width=True)

def show_simple_prediction(df):
    """Simple prediction interface"""
    st.header("🔮 Simple Sales Prediction")
    
    st.write("Enter product details to get a sales prediction:")
    
    with st.form("simple_prediction"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quantity = st.number_input("Quantity", min_value=1, max_value=50, value=1)
            discount = st.slider("Discount", 0.0, 1.0, 0.0, 0.01)
        
        with col2:
            category = st.selectbox("Category", df['Category'].unique())
            region = st.selectbox("Region", df['Region'].unique())
        
        with col3:
            segment = st.selectbox("Customer Segment", df['Segment'].unique())
            ship_mode = st.selectbox("Ship Mode", df['Ship.Mode'].unique())
        
        submitted = st.form_submit_button("🔮 Predict Sales")
        
        if submitted:
            # Simple prediction based on historical averages
            base_sales = df[
                (df['Category'] == category) & 
                (df['Region'] == region) & 
                (df['Segment'] == segment)
            ]['Sales'].mean()
            
            if pd.isna(base_sales):
                base_sales = df['Sales'].mean()
            
            # Adjust for quantity and discount
            predicted_sales = base_sales * quantity * (1 - discount * 0.5)
            
            # Display prediction
            st.success("🎯 Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Sales", f"${predicted_sales:.2f}")
            
            with col2:
                estimated_profit = predicted_sales * 0.2
                st.metric("Estimated Profit", f"${estimated_profit:.2f}")
            
            with col3:
                st.metric("Profit Margin", "20%")

def show_data_summary(df):
    """Halaman summary data"""
    st.header("📋 Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Overview")
        st.write(f"**Total records:** {len(df):,}")
        st.write(f"**Date range:** {df['Order.Date'].min().strftime('%Y-%m-%d')} to {df['Order.Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"**Unique customers:** {df['Customer.ID'].nunique():,}")
        st.write(f"**Unique products:** {df['Product.ID'].nunique():,}")
        st.write(f"**Missing values:** {df.isnull().sum().sum()}")
    
    with col2:
        st.subheader("💰 Business Metrics")
        st.write(f"**Total sales:** ${df['Sales'].sum():,.0f}")
        st.write(f"**Total profit:** ${df['Profit'].sum():,.0f}")
        st.write(f"**Average profit margin:** {df['Profit_Margin'].mean():.1%}")
        st.write(f"**Loss transactions:** {df['Is_Loss'].mean():.1%}")
        st.write(f"**Average delivery time:** {df['Delivery_Time'].mean():.1f} days")
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df[['Sales', 'Profit', 'Quantity', 'Discount', 'Profit_Margin']].describe(), use_container_width=True)

def main():
    # Header
    st.title("📊 Superstore Analytics Dashboard")
    
    # File upload option
    st.sidebar.markdown("### 📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="Upload your superstore.csv file or use sample data"
    )
    
    use_sample = st.sidebar.checkbox("Use Sample Data", value=False)
    
    # Load data
    with st.spinner('Loading data...'):
        if use_sample:
            df = create_sample_data()
            st.sidebar.success("✅ Using sample data")
        elif uploaded_file is not None:
            df = load_and_process_data(uploaded_file)
            if df is not None:
                st.sidebar.success("✅ File uploaded successfully")
        else:
            df = load_and_process_data()
            if df is not None:
                st.sidebar.success("✅ Data loaded from file")
    
    if df is None:
        st.error("❌ No data available. Please upload a file or use sample data.")
        st.stop()
    
    # Navigation
    st.sidebar.markdown("### 📋 Navigation")
    
    pages = [
        "📈 Overview",
        "📊 Sales Analysis",
        "🎯 Customer Analysis",
        "🔮 Simple Prediction",
        "📋 Data Summary"
    ]
    
    selected_page = st.sidebar.selectbox("Select Page", pages)
    
    # Filters
    st.sidebar.markdown("### 🔧 Filters")
    
    categories = st.sidebar.multiselect(
        "Select Categories",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )
    
    regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )
    
    # Apply filters
    filtered_df = df[
        (df['Category'].isin(categories)) &
        (df['Region'].isin(regions))
    ]
    
    # Show data info
    st.sidebar.markdown("### 📊 Data Info")
    st.sidebar.write(f"Total records: {len(filtered_df):,}")
    
    # Page routing
    if selected_page == "📈 Overview":
        show_overview(filtered_df)
    elif selected_page == "📊 Sales Analysis":
        show_sales_analysis(filtered_df)
    elif selected_page == "🎯 Customer Analysis":
        show_customer_analysis(filtered_df)
    elif selected_page == "🔮 Simple Prediction":
        show_simple_prediction(filtered_df)
    elif selected_page == "📋 Data Summary":
        show_data_summary(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | Superstore Analytics Dashboard")

if __name__ == "__main__":
    main()
