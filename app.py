import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Tambahan import untuk ML
try:
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   from sklearn.linear_model import LinearRegression, Ridge, Lasso
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
   SKLEARN_AVAILABLE = True
except ImportError:
   SKLEARN_AVAILABLE = False

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
.prediction-form {
   background-color: #f0f2f6;
   padding: 20px;
   border-radius: 10px;
   margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

def load_and_process_data(uploaded_file=None):
   """Load dan proses data superstore"""
   try:
       if uploaded_file is not None:
           df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
       else:
           try:
               df = pd.read_csv('superstore.csv', encoding='iso-8859-1')
           except:
               try:
                   df = pd.read_csv('superstore.csv', encoding='utf-8')
               except:
                   return None
       
       # Data cleaning
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
   """Buat sample data"""
   np.random.seed(42)
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
   df['Year'] = df['Order.Date'].dt.year
   df['Month'] = df['Order.Date'].dt.month
   df['Quarter'] = df['Order.Date'].dt.quarter
   df['DayOfWeek'] = df['Order.Date'].dt.dayofweek
   df['DayName'] = df['Order.Date'].dt.day_name()
   df['MonthName'] = df['Order.Date'].dt.month_name()
   df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
   
   df['Profit_Margin'] = df['Profit'] / df['Sales']
   df['Profit_Margin'] = df['Profit_Margin'].replace([np.inf, -np.inf], 0)
   df['Is_Loss'] = (df['Profit'] < 0).astype(int)
   df['Revenue_per_Quantity'] = df['Sales'] / df['Quantity']
   
   df['Delivery_Time'] = (df['Ship.Date'] - df['Order.Date']).dt.days
   
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
   
   df['Order_Size'] = np.where(df['Quantity'] >= 5, 'Large',
                              np.where(df['Quantity'] >= 2, 'Medium', 'Small'))
   
   df['Season'] = df['Month'].map({
       12: 'Winter', 1: 'Winter', 2: 'Winter',
       3: 'Spring', 4: 'Spring', 5: 'Spring',
       6: 'Summer', 7: 'Summer', 8: 'Summer',
       9: 'Fall', 10: 'Fall', 11: 'Fall'
   })
   
   return df

def prepare_ml_features(df):
   """Prepare features untuk machine learning"""
   if not SKLEARN_AVAILABLE:
       return None, None, None, None
   
   df_ml = df.copy()
   
   encoders = {}
   categorical_features = ['Category', 'Segment', 'Region', 'Ship.Mode', 'Season', 'Discount_Category', 'Order_Size']
   
   for col in categorical_features:
       if col in df_ml.columns:
           le = LabelEncoder()
           df_ml[col + '_encoded'] = le.fit_transform(df_ml[col].astype(str))
           encoders[col] = le
   
   numerical_features = ['Quantity', 'Discount', 'Year', 'Month', 'Quarter', 'DayOfWeek', 'IsWeekend', 'Delivery_Time']
   feature_columns = numerical_features + [col + '_encoded' for col in categorical_features if col in df_ml.columns]
   available_features = [col for col in feature_columns if col in df_ml.columns]
   
   X = df_ml[available_features].fillna(0)
   y = df_ml['Sales']
   
   return X, y, available_features, encoders

def train_ml_models(df):
   """Train multiple ML models"""
   if not SKLEARN_AVAILABLE:
       return None
   
   X, y, features, encoders = prepare_ml_features(df)
   
   if X is None:
       return None
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   models = {
       'Linear Regression': LinearRegression(),
       'Ridge Regression': Ridge(alpha=1.0),
       'Lasso Regression': Lasso(alpha=0.1),
       'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
   }
   
   results = {}
   
   for name, model in models.items():
       if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
           model.fit(X_train_scaled, y_train)
           y_pred = model.predict(X_test_scaled)
           use_scaled = True
       else:
           model.fit(X_train, y_train)
           y_pred = model.predict(X_test)
           use_scaled = False
       
       mse = mean_squared_error(y_test, y_pred)
       r2 = r2_score(y_test, y_pred)
       mae = mean_absolute_error(y_test, y_pred)
       
       results[name] = {
           'model': model,
           'mse': mse,
           'rmse': np.sqrt(mse),
           'r2': r2,
           'mae': mae,
           'predictions': y_pred,
           'use_scaled': use_scaled
       }
   
   return {
       'results': results,
       'y_test': y_test,
       'features': features,
       'encoders': encoders,
       'scaler': scaler,
       'X_train': X_train,
       'X_train_scaled': X_train_scaled
   }

def show_overview(df):
   """Halaman overview"""
   st.header("📈 Business Overview & KPIs")
   
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
   
   col1, col2 = st.columns(2)
   
   with col1:
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
       region_sales = df.groupby('Region')['Sales'].sum()
       fig = px.pie(
           values=region_sales.values,
           names=region_sales.index,
           title="🌍 Sales by Region"
       )
       st.plotly_chart(fig, use_container_width=True)
   
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
       top_products = df.groupby('Product.Name')['Sales'].sum().sort_values(ascending=False).head(10)
       fig = px.bar(
           x=top_products.values,
           y=top_products.index,
           orientation='h',
           title="🏆 Top 10 Products by Sales"
       )
       st.plotly_chart(fig, use_container_width=True)
   
   with col2:
       seasonal_sales = df.groupby('Season')['Sales'].sum()
       fig = px.bar(
           x=seasonal_sales.index,
           y=seasonal_sales.values,
           title="🌸 Seasonal Sales Distribution",
           color=seasonal_sales.values,
           color_continuous_scale="Viridis"
       )
       st.plotly_chart(fig, use_container_width=True)
   
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
           df.sample(min(500, len(df))),
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
       segment_sales = df.groupby('Segment')['Sales'].sum()
       fig = px.pie(
           values=segment_sales.values,
           names=segment_sales.index,
           title="👥 Sales by Customer Segment"
       )
       st.plotly_chart(fig, use_container_width=True)
   
   st.subheader("📊 Customer Segment Analysis")
   segment_analysis = df.groupby('Segment').agg({
       'Sales': ['sum', 'mean', 'count'],
       'Profit': ['sum', 'mean'],
       'Customer.ID': 'nunique'
   }).round(2)
   
   segment_analysis.columns = ['Total Sales', 'Avg Sales', 'Transactions', 'Total Profit', 'Avg Profit', 'Customers']
   st.dataframe(segment_analysis, use_container_width=True)

def show_machine_learning(df):
   """Halaman Machine Learning"""
   st.header("🤖 Machine Learning Models")
   
   if not SKLEARN_AVAILABLE:
       st.error("❌ Scikit-learn is not available. Please install it to use ML features.")
       st.code("pip install scikit-learn")
       return None
   
   with st.spinner('Training machine learning models...'):
       ml_data = train_ml_models(df)
   
   if ml_data is None:
       st.error("❌ Failed to train models")
       return None
   
   results = ml_data['results']
   y_test = ml_data['y_test']
   
   st.subheader("📊 Model Performance Comparison")
   
   comparison_data = []
   for model_name, result in results.items():
       comparison_data.append({
           'Model': model_name,
           'R² Score': result['r2'],
           'RMSE': result['rmse'],
           'MAE': result['mae']
       })
   
   comparison_df = pd.DataFrame(comparison_data)
   comparison_df = comparison_df.sort_values('R² Score', ascending=False)
   
   st.dataframe(comparison_df.round(4), use_container_width=True)
   
   best_model_name = comparison_df.iloc[0]['Model']
   best_result = results[best_model_name]
   
   st.success(f"🏆 Best Model: {best_model_name}")
   
   col1, col2, col3 = st.columns(3)
   with col1:
       st.metric("R² Score", f"{best_result['r2']:.4f}")
   with col2:
       st.metric("RMSE", f"${best_result['rmse']:,.0f}")
   with col3:
       st.metric("MAE", f"${best_result['mae']:,.0f}")
   
   col1, col2 = st.columns(2)
   
   with col1:
       fig = px.bar(
           comparison_df,
           x='Model',
           y='R² Score',
           title="Model R² Comparison",
           color='R² Score',
           color_continuous_scale="Blues"
       )
       st.plotly_chart(fig, use_container_width=True)
   
   with col2:
       pred_df = pd.DataFrame({
           'Actual': y_test,
           'Predicted': best_result['predictions']
       })
       
       fig = px.scatter(
           pred_df,
           x='Actual',
           y='Predicted',
           title=f"Prediction vs Actual - {best_model_name}"
       )
       fig.add_shape(
           type="line",
           x0=y_test.min(), y0=y_test.min(),
           x1=y_test.max(), y1=y_test.max(),
           line=dict(dash="dash", color="red")
       )
       st.plotly_chart(fig, use_container_width=True)
   
   if 'Random Forest' in results:
       st.subheader("🔍 Feature Importance (Random Forest)")
       rf_model = results['Random Forest']['model']
       features = ml_data['features']
       
       importance_df = pd.DataFrame({
           'Feature': features,
           'Importance': rf_model.feature_importances_
       }).sort_values('Importance', ascending=False).head(10)
       
       fig = px.bar(
           importance_df,
           x='Importance',
           y='Feature',
           orientation='h',
           title="Top 10 Feature Importance"
       )
       st.plotly_chart(fig, use_container_width=True)
   
   return ml_data

def show_sales_prediction(df, ml_data=None):
   """Halaman prediksi sales"""
   st.header("🔮 Sales Prediction")
   
   if not SKLEARN_AVAILABLE:
       st.error("❌ Scikit-learn is not available. Please install it to use prediction features.")
       return
   
   if ml_data is None:
       with st.spinner('Training models for prediction...'):
           ml_data = train_ml_models(df)
       
       if ml_data is None:
           st.error("❌ Failed to train prediction models")
           return
   
   results = ml_data['results']
   features = ml_data['features']
   encoders = ml_data['encoders']
   scaler = ml_data['scaler']
   
   model_names = list(results.keys())
   selected_model = st.selectbox("🤖 Select Model for Prediction", model_names)
   
   model_info = results[selected_model]
   model = model_info['model']
   use_scaled = model_info['use_scaled']
   
   st.subheader(f"📊 Make Prediction using {selected_model}")
   
   with st.form("prediction_form"):
       st.markdown('<div class="prediction-form">', unsafe_allow_html=True)
       
       col1, col2, col3 = st.columns(3)
       
       with col1:
           st.write("**📦 Product Details**")
           quantity = st.number_input("Quantity", min_value=1, max_value=50, value=1)
           discount = st.slider("Discount (%)", 0.0, 100.0, 0.0, 1.0) / 100
           category = st.selectbox("Category", df['Category'].unique())
       
       with col2:
           st.write("**👤 Customer Details**")
           segment = st.selectbox("Customer Segment", df['Segment'].unique())
           region = st.selectbox("Region", df['Region'].unique())
           ship_mode = st.selectbox("Ship Mode", df['Ship.Mode'].unique())
       
       with col3:
           st.write("**📅 Time Details**")
           year = st.selectbox("Year", [2021, 2022, 2023, 2024, 2025])
           month = st.selectbox("Month", list(range(1, 13)))
           quarter = st.selectbox("Quarter", [1, 2, 3, 4])
           day_of_week = st.selectbox("Day of Week", list(range(7)))
           is_weekend = st.checkbox("Weekend?")
           delivery_time = st.number_input("Delivery Time (days)", min_value=0, max_value=30, value=3)
       
       st.markdown('</div>', unsafe_allow_html=True)
       
       submitted = st.form_submit_button("🔮 Predict Sales", use_container_width=True)
       
       if submitted:
           input_data = {
               'Quantity': quantity,
               'Discount': discount,
               'Year': year,
               'Month': month,
               'Quarter': quarter,
               'DayOfWeek': day_of_week,
               'IsWeekend': int(is_weekend),
               'Delivery_Time': delivery_time
           }
           
           categorical_inputs = {
               'Category': category,
               'Segment': segment,
               'Region': region,
               'Ship.Mode': ship_mode,
               'Season': ['Winter', 'Spring', 'Summer', 'Fall'][quarter-1],
               'Discount_Category': 'No Discount' if discount == 0 else (
                   'Low Discount' if discount <= 0.2 else (
                       'Medium Discount' if discount <= 0.5 else 'High Discount')),
               'Order_Size': 'Large' if quantity >= 5 else (
                   'Medium' if quantity >= 2 else 'Small')
           }
           
           for col, value in categorical_inputs.items():
               if col in encoders and col + '_encoded' in features:
                   try:
                       encoded_value = encoders[col].transform([value])[0]
                       input_data[col + '_encoded'] = encoded_value
                   except:
                       input_data[col + '_encoded'] = 0
           
           input_array = []
           for feature in features:
               if feature in input_data:
                   input_array.append(input_data[feature])
               else:
                   input_array.append(0)
           
           input_array = np.array(input_array).reshape(1, -1)
           
           if use_scaled:
               input_array = scaler.transform(input_array)
           
           try:
               prediction = model.predict(input_array)[0]
               
               st.success("🎯 Prediction Complete!")
               
               col1, col2, col3 = st.columns(3)
               
               with col1:
                   st.metric("💰 Predicted Sales", f"${prediction:,.2f}")
               
               with col2:
                   estimated_profit = prediction * 0.2
                   st.metric("📈 Estimated Profit", f"${estimated_profit:,.2f}")
               
               with col3:
                   confidence = model_info['r2']
                   st.metric("🎯 Model Confidence", f"{confidence:.1%}")
               
               st.subheader("💡 Business Insights")
               
               insights = []
               
               if discount > 0.3:
                   insights.append("⚠️ High discount may reduce profit margin significantly")
               elif discount > 0.1:
                   insights.append("📊 Moderate discount applied - monitor profit impact")
               else:
                   insights.append("✅ Low/No discount - good for profit margins")
               
               if quantity >= 5:
                   insights.append("📦 Large order quantity - excellent for bulk sales")
               elif quantity >= 2:
                   insights.append("📦 Medium order - good sales volume")
               
               if category == 'Technology':
                   insights.append("💻 Technology products typically have higher margins")
               elif category == 'Furniture':
                   insights.append("🪑 Furniture sales often have longer delivery times")
               
               if is_weekend:
                   insights.append("📅 Weekend order - may have different customer behavior patterns")
               
               for insight in insights:
                   st.write(f"• {insight}")
               
               st.subheader("🔍 Scenario Analysis")
               st.write("See how different discount levels would affect the prediction:")
               
               scenarios = {
                   "No Discount (0%)": 0.0,
                   "Small Discount (10%)": 0.1,
                   "Medium Discount (25%)": 0.25,
                   "Large Discount (50%)": 0.5
               }
               
               scenario_results = []
               
               for scenario_name, discount_val in scenarios.items():
                   input_data_scenario = input_data.copy()
                   input_data_scenario['Discount'] = discount_val
                   
                   discount_cat = 'No Discount' if discount_val == 0 else (
                       'Low Discount' if discount_val <= 0.2 else (
                           'Medium Discount' if discount_val <= 0.5 else 'High Discount'))
                   
                   if 'Discount_Category' in encoders and 'Discount_Category_encoded' in features:
                       try:
                           encoded_discount_cat = encoders['Discount_Category'].transform([discount_cat])[0]
                           input_data_scenario['Discount_Category_encoded'] = encoded_discount_cat
                       except:
                           input_data_scenario['Discount_Category_encoded'] = 0
                   
                   input_array_scenario = []
                   for feature in features:
                       if feature in input_data_scenario:
                           input_array_scenario.append(input_data_scenario[feature])
                       else:
                           input_array_scenario.append(0)
                   
                   input_array_scenario = np.array(input_array_scenario).reshape(1, -1)
                   
                   if use_scaled:
                       input_array_scenario = scaler.transform(input_array_scenario)
                   
                   scenario_pred = model.predict(input_array_scenario)[0]
                   scenario_profit = scenario_pred * 0.2
                   
                   scenario_results.append({
                       'Scenario': scenario_name,
                       'Predicted Sales': f"${scenario_pred:,.0f}",
                       'Est. Profit': f"${scenario_profit:,.0f}",
                       'Profit Margin': "20%"
                   })
               
               scenario_df = pd.DataFrame(scenario_results)
               st.dataframe(scenario_df, use_container_width=True)
               
               scenario_sales = [float(result['Predicted Sales'].replace('$', '').replace(',', '')) for result in scenario_results]
               scenario_names = [result['Scenario'] for result in scenario_results]
               
               fig = px.bar(
                   x=scenario_names,
                   y=scenario_sales,
      title="📊 Sales Prediction by Discount Scenario",
                   color=scenario_sales,
                   color_continuous_scale="Blues"
               )
               fig.update_layout(xaxis_title="Discount Scenario", yaxis_title="Predicted Sales ($)")
               st.plotly_chart(fig, use_container_width=True)
               
           except Exception as e:
               st.error(f"❌ Prediction failed: {str(e)}")

def show_batch_prediction(df, ml_data=None):
   """Halaman batch prediction"""
   st.header("📊 Batch Sales Prediction")
   
   if not SKLEARN_AVAILABLE:
       st.error("❌ Scikit-learn is not available. Please install it to use batch prediction features.")
       return
   
   if ml_data is None:
       with st.spinner('Training models for batch prediction...'):
           ml_data = train_ml_models(df)
       
       if ml_data is None:
           st.error("❌ Failed to train prediction models")
           return
   
   st.write("Upload a CSV file with product data to get batch sales predictions.")
   
   with st.expander("📋 View Required CSV Format"):
       sample_data = {
           'Quantity': [1, 2, 5],
           'Discount': [0.0, 0.1, 0.25],
           'Category': ['Technology', 'Furniture', 'Office Supplies'],
           'Segment': ['Consumer', 'Corporate', 'Home Office'],
           'Region': ['East', 'West', 'Central'],
           'Ship.Mode': ['Standard Class', 'First Class', 'Same Day'],
           'Year': [2024, 2024, 2024],
           'Month': [1, 6, 12],
           'Quarter': [1, 2, 4],
           'DayOfWeek': [0, 3, 6],
           'IsWeekend': [0, 0, 1],
           'Delivery_Time': [3, 2, 1]
       }
       sample_df = pd.DataFrame(sample_data)
       st.dataframe(sample_df)
       
       csv = sample_df.to_csv(index=False)
       st.download_button(
           label="📥 Download Sample CSV",
           data=csv,
           file_name="sample_batch_prediction.csv",
           mime="text/csv"
       )
   
   uploaded_file = st.file_uploader("📤 Upload CSV for Batch Prediction", type=['csv'])
   
   if uploaded_file is not None:
       try:
           batch_df = pd.read_csv(uploaded_file)
           
           st.success(f"✅ File uploaded successfully! {len(batch_df)} rows found.")
           
           st.subheader("👀 Data Preview")
           st.dataframe(batch_df.head(10))
           
           results = ml_data['results']
           features = ml_data['features']
           encoders = ml_data['encoders']
           scaler = ml_data['scaler']
           
           model_names = list(results.keys())
           selected_model = st.selectbox("🤖 Select Model for Batch Prediction", model_names)
           
           if st.button("🚀 Run Batch Prediction", use_container_width=True):
               with st.spinner("Making predictions..."):
                   model_info = results[selected_model]
                   model = model_info['model']
                   use_scaled = model_info['use_scaled']
                   
                   predictions = []
                   errors = []
                   
                   for idx, row in batch_df.iterrows():
                       try:
                           input_data = {}
                           
                           numerical_cols = ['Quantity', 'Discount', 'Year', 'Month', 'Quarter', 'DayOfWeek', 'IsWeekend', 'Delivery_Time']
                           for col in numerical_cols:
                               if col in row:
                                   input_data[col] = row[col]
                               else:
                                   input_data[col] = 0
                           
                           categorical_cols = ['Category', 'Segment', 'Region', 'Ship.Mode']
                           for col in categorical_cols:
                               if col in row and col in encoders:
                                   try:
                                       encoded_value = encoders[col].transform([str(row[col])])[0]
                                       input_data[col + '_encoded'] = encoded_value
                                   except:
                                       input_data[col + '_encoded'] = 0
                           
                           if 'Season' in encoders:
                               quarter = input_data.get('Quarter', 1)
                               season = ['Winter', 'Spring', 'Summer', 'Fall'][quarter-1]
                               try:
                                   input_data['Season_encoded'] = encoders['Season'].transform([season])[0]
                               except:
                                   input_data['Season_encoded'] = 0
                           
                           if 'Discount_Category' in encoders:
                               discount = input_data.get('Discount', 0)
                               discount_cat = 'No Discount' if discount == 0 else (
                                   'Low Discount' if discount <= 0.2 else (
                                       'Medium Discount' if discount <= 0.5 else 'High Discount'))
                               try:
                                   input_data['Discount_Category_encoded'] = encoders['Discount_Category'].transform([discount_cat])[0]
                               except:
                                   input_data['Discount_Category_encoded'] = 0
                           
                           if 'Order_Size' in encoders:
                               quantity = input_data.get('Quantity', 1)
                               order_size = 'Large' if quantity >= 5 else (
                                   'Medium' if quantity >= 2 else 'Small')
                               try:
                                   input_data['Order_Size_encoded'] = encoders['Order_Size'].transform([order_size])[0]
                               except:
                                   input_data['Order_Size_encoded'] = 0
                           
                           input_array = []
                           for feature in features:
                               if feature in input_data:
                                   input_array.append(input_data[feature])
                               else:
                                   input_array.append(0)
                           
                           input_array = np.array(input_array).reshape(1, -1)
                           
                           if use_scaled:
                               input_array = scaler.transform(input_array)
                           
                           prediction = model.predict(input_array)[0]
                           predictions.append(prediction)
                           errors.append(None)
                           
                       except Exception as e:
                           predictions.append(0)
                           errors.append(str(e))
                   
                   batch_df['Predicted_Sales'] = predictions
                   batch_df['Estimated_Profit'] = [p * 0.2 for p in predictions]
                   batch_df['Prediction_Error'] = errors
                   
                   st.subheader("🎯 Prediction Results")
                   
                   col1, col2, col3 = st.columns(3)
                   
                   with col1:
                       st.metric("Total Predictions", len(predictions))
                   
                   with col2:
                       successful_predictions = sum(1 for e in errors if e is None)
                       st.metric("Successful Predictions", successful_predictions)
                   
                   with col3:
                       total_predicted_sales = sum(predictions)
                       st.metric("Total Predicted Sales", f"${total_predicted_sales:,.0f}")
                   
                   st.dataframe(batch_df, use_container_width=True)
                   
                   csv = batch_df.to_csv(index=False)
                   st.download_button(
                       label="📥 Download Predictions",
                       data=csv,
                       file_name="batch_predictions.csv",
                       mime="text/csv"
                   )
                   
                   st.subheader("📊 Prediction Analysis")
                   
                   col1, col2 = st.columns(2)
                   
                   with col1:
                       fig = px.histogram(
                           batch_df,
                           x='Predicted_Sales',
                           title="Distribution of Predicted Sales",
                           nbins=20
                       )
                       st.plotly_chart(fig, use_container_width=True)
                   
                   with col2:
                       if 'Category' in batch_df.columns:
                           category_pred = batch_df.groupby('Category')['Predicted_Sales'].sum()
                           fig = px.bar(
                               x=category_pred.index,
                               y=category_pred.values,
                               title="Predicted Sales by Category"
                           )
                           st.plotly_chart(fig, use_container_width=True)
       
       except Exception as e:
           st.error(f"❌ Error processing file: {str(e)}")

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
   
   st.subheader("👀 Data Preview")
   st.dataframe(df.head(10), use_container_width=True)
   
   st.subheader("📈 Statistical Summary")
   st.dataframe(df[['Sales', 'Profit', 'Quantity', 'Discount', 'Profit_Margin']].describe(), use_container_width=True)

def main():
   st.title("📊 Superstore Analytics Dashboard")
   
   st.sidebar.markdown("### 📁 Data Source")
   uploaded_file = st.sidebar.file_uploader(
       "Upload CSV file", 
       type=['csv'],
       help="Upload your superstore.csv file or use sample data"
   )
   
   use_sample = st.sidebar.checkbox("Use Sample Data", value=False)
   
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
   
   st.sidebar.markdown("### 📋 Navigation")
   
   pages = [
       "📈 Overview",
       "📊 Sales Analysis",
       "🎯 Customer Analysis",
       "🤖 Machine Learning",
       "🔮 Sales Prediction",
       "📊 Batch Prediction",
       "📋 Data Summary"
   ]
   
   selected_page = st.sidebar.selectbox("Select Page", pages)
   
   if 'ml_data' not in st.session_state:
       st.session_state.ml_data = None
   
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
   
   filtered_df = df[
       (df['Category'].isin(categories)) &
       (df['Region'].isin(regions))
   ]
   
   st.sidebar.markdown("### 📊 Data Info")
   st.sidebar.write(f"Total records: {len(filtered_df):,}")
   
   if selected_page == "📈 Overview":
       show_overview(filtered_df)
   elif selected_page == "📊 Sales Analysis":
       show_sales_analysis(filtered_df)
   elif selected_page == "🎯 Customer Analysis":
       show_customer_analysis(filtered_df)
   elif selected_page == "🤖 Machine Learning":
       ml_data = show_machine_learning(filtered_df)
       if ml_data:
           st.session_state.ml_data = ml_data
   elif selected_page == "🔮 Sales Prediction":
       show_sales_prediction(filtered_df, st.session_state.ml_data)
   elif selected_page == "📊 Batch Prediction":
       show_batch_prediction(filtered_df, st.session_state.ml_data)
   elif selected_page == "📋 Data Summary":
       show_data_summary(filtered_df)
   
   st.markdown("---")
   st.markdown("Built with ❤️ using Streamlit | Superstore Analytics Dashboard")

if __name__ == "__main__":
   main()
