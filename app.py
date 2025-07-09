# Tambahkan import ini di bagian atas app.py
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import base64

# Tambahkan fungsi-fungsi ML ini setelah fungsi load_and_process_data

@st.cache_data
def prepare_ml_data(df):
    """Prepare data untuk machine learning"""
    df_model = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Category', 'Ship.Mode', 'Segment', 'Region', 'Customer_Segment',
                       'Delivery_Efficiency', 'Discount_Category', 'Order_Size', 'Season']
    
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col + '_encoded'] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le
    
    # Select features untuk prediksi
    feature_cols = ['Quantity', 'Discount', 'Year', 'Month', 'Quarter',
                   'DayOfWeek', 'IsWeekend', 'Delivery_Time']
    
    # Tambah encoded categorical features jika ada
    encoded_features = [col for col in df_model.columns if col.endswith('_encoded')]
    feature_cols.extend(encoded_features)
    
    # Filter fitur yang tersedia
    available_features = [col for col in feature_cols if col in df_model.columns]
    
    X = df_model[available_features].fillna(0)
    y = df_model['Sales']
    
    return X, y, available_features, label_encoders

@st.cache_data
def train_models_ml(df):
    """Training multiple models untuk prediksi"""
    X, y, features, encoders = prepare_ml_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'use_scaling': True,
            'params': {}
        },
        'Ridge Regression': {
            'model': Ridge(alpha=1.0),
            'use_scaling': True,
            'params': {'alpha': [0.1, 1.0, 10.0]}
        },
        'Lasso Regression': {
            'model': Lasso(alpha=0.1, max_iter=2000),
            'use_scaling': True,
            'params': {'alpha': [0.01, 0.1, 1.0]}
        },
        'Random Forest': {
            'model': RandomForestRegressor(n_estimators=100, random_state=42),
            'use_scaling': False,
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        }
    }
    
    results = {}
    trained_models = {}
    
    for name, config in models.items():
        model = config['model']
        use_scaling = config['use_scaling']
        params = config['params']
        
        # Pilih data yang tepat
        X_train_use = X_train_scaled if use_scaling else X_train
        X_test_use = X_test_scaled if use_scaling else X_test
        
        # Hyperparameter tuning jika ada parameter
        if params:
            grid_search = GridSearchCV(
                model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train_use, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            best_model = model
            best_model.fit(X_train_use, y_train)
            best_params = "No tuning"
        
        # Prediksi dan evaluasi
        y_pred_train = best_model.predict(X_train_use)
        y_pred_test = best_model.predict(X_test_use)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'model': best_model,
            'scaler': scaler if use_scaling else None,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'best_params': best_params,
            'use_scaling': use_scaling,
            'predictions': y_pred_test
        }
        
        trained_models[name] = best_model
    
    return results, y_test, features, encoders, scaler

def show_machine_learning(df):
    """Halaman Machine Learning"""
    st.header("🤖 Machine Learning Models")
    
    with st.spinner('Training models...'):
        results, y_test, features, encoders, scaler = train_models_ml(df)
    
    # Model comparison
    st.subheader("📊 Model Performance Comparison")
    
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Train R²': result['train_r2'],
            'Test R²': result['test_r2'],
            'Train RMSE': np.sqrt(result['train_mse']),
            'Test RMSE': np.sqrt(result['test_mse']),
            'Overfitting': result['train_r2'] - result['test_r2']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test R²', ascending=False)
    
    # Display comparison table
    st.dataframe(comparison_df.round(4), use_container_width=True)
    
    # Charts untuk perbandingan
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            comparison_df, 
            x='Model', 
            y='Test R²',
            title="Model R² Scores",
            color='Test R²',
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            comparison_df, 
            x='Model', 
            y='Test RMSE',
            title="Model RMSE",
            color='Test RMSE',
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model info
    best_model_name = comparison_df.iloc[0]['Model']
    best_result = results[best_model_name]
    
    st.success(f"🏆 Best Model: {best_model_name}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{best_result['test_r2']:.4f}")
    with col2:
        st.metric("RMSE", f"${np.sqrt(best_result['test_mse']):,.0f}")
    with col3:
        st.metric("MAE", f"${mean_absolute_error(y_test, best_result['predictions']):,.0f}")
    
    # Prediction vs Actual plot
    st.subheader("🎯 Prediction vs Actual (Best Model)")
    
    pred_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': best_result['predictions']
    })
    
    fig = px.scatter(
        pred_df, 
        x='Actual', 
        y='Predicted',
        title=f"Predictions vs Actual - {best_model_name}",
        trendline="ols"
    )
    fig.add_shape(
        type="line",
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(dash="dash", color="red")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance untuk Random Forest
    if 'Random Forest' in results:
        st.subheader("🔍 Feature Importance (Random Forest)")
        rf_model = results['Random Forest']['model']
        
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
    
    return results, features, encoders, scaler

def show_prediction_interface(df, results=None, features=None, encoders=None, scaler=None):
    """Interface untuk prediksi sales"""
    st.header("🔮 Sales Prediction")
    
    # Jika belum ada model, train dulu
    if results is None:
        st.info("Please visit the Machine Learning page first to train models.")
        with st.spinner('Training models...'):
            results, y_test, features, encoders, scaler = train_models_ml(df)
    
    # Pilih model untuk prediksi
    model_names = list(results.keys())
    selected_model = st.selectbox("Select Model for Prediction", model_names)
    
    model_info = results[selected_model]
    model = model_info['model']
    use_scaling = model_info['use_scaling']
    
    st.subheader(f"📊 Predict Sales using {selected_model}")
    
    # Input form untuk prediksi
    with st.form("prediction_form"):
        st.write("### Enter Product Details:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quantity = st.number_input("Quantity", min_value=1, max_value=50, value=1)
            discount = st.slider("Discount", 0.0, 1.0, 0.0, 0.01)
            year = st.selectbox("Year", [2021, 2022, 2023, 2024, 2025])
        
        with col2:
            month = st.selectbox("Month", list(range(1, 13)))
            quarter = st.selectbox("Quarter", [1, 2, 3, 4])
            day_of_week = st.selectbox("Day of Week", list(range(7)), 
                                     format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 
                                                           'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
        
        with col3:
            is_weekend = st.checkbox("Weekend?")
            delivery_time = st.number_input("Delivery Time (days)", min_value=0, max_value=30, value=3)
        
        # Categorical inputs
        st.write("### Product & Customer Details:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            category = st.selectbox("Category", df['Category'].unique())
            ship_mode = st.selectbox("Ship Mode", df['Ship.Mode'].unique())
        
        with col2:
            segment = st.selectbox("Customer Segment", df['Segment'].unique())
            region = st.selectbox("Region", df['Region'].unique())
        
        with col3:
            order_size = st.selectbox("Order Size", ['Small', 'Medium', 'Large'])
            season = st.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter'])
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Sales")
        
        if submitted:
            # Prepare input data
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
            
            # Encode categorical variables
            categorical_inputs = {
                'Category': category,
                'Ship.Mode': ship_mode,
                'Segment': segment,
                'Region': region,
                'Order_Size': order_size,
                'Season': season
            }
            
            for col, value in categorical_inputs.items():
                if col in encoders and col + '_encoded' in features:
                    try:
                        encoded_value = encoders[col].transform([value])[0]
                        input_data[col + '_encoded'] = encoded_value
                    except:
                        # Handle unseen categories
                        input_data[col + '_encoded'] = 0
            
            # Create input array
            input_array = []
            for feature in features:
                if feature in input_data:
                    input_array.append(input_data[feature])
                else:
                    input_array.append(0)  # Default value
            
            input_array = np.array(input_array).reshape(1, -1)
            
            # Scale if needed
            if use_scaling and scaler is not None:
                input_array = scaler.transform(input_array)
            
            # Make prediction
            prediction = model.predict(input_array)[0]
            
            # Display result
            st.success("🎯 Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Sales", f"${prediction:,.2f}")
            
            with col2:
                # Calculate profit estimate (assuming 20% margin)
                estimated_profit = prediction * 0.2
                st.metric("Estimated Profit", f"${estimated_profit:,.2f}")
            
            with col3:
                # Model confidence
                confidence = model_info['test_r2']
                st.metric("Model Confidence", f"{confidence:.1%}")
            
            # Additional insights
            st.subheader("📊 Prediction Insights")
            
            # Prediction breakdown
            insights = []
            
            if discount > 0.3:
                insights.append("⚠️ High discount may reduce profit margin")
            
            if quantity >= 5:
                insights.append("✅ Large order quantity - good for bulk sales")
            
            if is_weekend:
                insights.append("📅 Weekend order - may have different patterns")
            
            if delivery_time <= 2:
                insights.append("🚀 Fast delivery - premium service")
            
            if category == 'Technology':
                insights.append("💻 Technology products typically have higher margins")
            
            for insight in insights:
                st.write(insight)
            
            # Scenario analysis
            st.subheader("🔍 Scenario Analysis")
            
            scenarios = {
                "No Discount": 0.0,
                "Small Discount (10%)": 0.1,
                "Medium Discount (25%)": 0.25,
                "Large Discount (50%)": 0.5
            }
            
            scenario_results = []
            
            for scenario_name, discount_val in scenarios.items():
                # Update discount in input
                input_data_scenario = input_data.copy()
                input_data_scenario['Discount'] = discount_val
                
                # Create input array
                input_array_scenario = []
                for feature in features:
                    if feature in input_data_scenario:
                        input_array_scenario.append(input_data_scenario[feature])
                    else:
                        input_array_scenario.append(0)
                
                input_array_scenario = np.array(input_array_scenario).reshape(1, -1)
                
                # Scale if needed
                if use_scaling and scaler is not None:
                    input_array_scenario = scaler.transform(input_array_scenario)
                
                # Predict
                scenario_pred = model.predict(input_array_scenario)[0]
                scenario_results.append({
                    'Scenario': scenario_name,
                    'Discount': f"{discount_val:.0%}",
                    'Predicted Sales': f"${scenario_pred:,.0f}",
                    'Est. Profit': f"${scenario_pred * 0.2:,.0f}"
                })
            
            scenario_df = pd.DataFrame(scenario_results)
            st.dataframe(scenario_df, use_container_width=True)

def show_batch_prediction(df, results=None, features=None, encoders=None, scaler=None):
    """Interface untuk batch prediction"""
    st.header("📊 Batch Prediction")
    
    # Jika belum ada model, train dulu
    if results is None:
        st.info("Please visit the Machine Learning page first to train models.")
        return
    
    st.write("Upload a CSV file with the same format as training data for batch predictions.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded file has {len(batch_df)} rows")
            
            # Preview data
            st.subheader("Data Preview")
            st.dataframe(batch_df.head())
            
            # Select model
            model_names = list(results.keys())
            selected_model = st.selectbox("Select Model", model_names)
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Making predictions..."):
                    # Process batch data (simplified)
                    model_info = results[selected_model]
                    model = model_info['model']
                    
                    # Simple prediction (would need full preprocessing in real scenario)
                    if 'Sales' in batch_df.columns:
                        # Remove target if present
                        X_batch = batch_df.drop('Sales', axis=1)
                    else:
                        X_batch = batch_df
                    
                    # For demo, just predict on first few numeric columns
                    numeric_cols = X_batch.select_dtypes(include=[np.number]).columns[:len(features)]
                    if len(numeric_cols) >= len(features):
                        X_batch_numeric = X_batch[numeric_cols[:len(features)]]
                        
                        if model_info['use_scaling']:
                            X_batch_scaled = scaler.transform(X_batch_numeric)
                            predictions = model.predict(X_batch_scaled)
                        else:
                            predictions = model.predict(X_batch_numeric)
                        
                        # Add predictions to dataframe
                        batch_df['Predicted_Sales'] = predictions
                        batch_df['Estimated_Profit'] = predictions * 0.2
                        
                        st.success("Predictions completed!")
                        st.dataframe(batch_df)
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    else:
                        st.error("Uploaded file doesn't have enough numeric columns for prediction.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Update fungsi main() untuk menambah halaman prediksi
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
            df = load_and_process_data()
            if df is None:
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
        st.error("❌ No data available.")
        st.stop()
    
    # Navigation - TAMBAH HALAMAN PREDIKSI
    st.sidebar.markdown("### 📋 Navigation")
    
    pages = [
        "📈 Overview",
        "📊 Sales Analysis",
        "🎯 Customer Analysis",
        "🤖 Machine Learning",      # BARU
        "🔮 Sales Prediction",      # BARU
        "📊 Batch Prediction",      # BARU
        "📋 Data Summary"
    ]
    
    selected_page = st.sidebar.selectbox("Select Page", pages)
    
    # Initialize session state untuk menyimpan model
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
        st.session_state.ml_features = None
        st.session_state.ml_encoders = None
        st.session_state.ml_scaler = None
    
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
    elif selected_page == "🤖 Machine Learning":
        results, features, encoders, scaler = show_machine_learning(filtered_df)
        # Simpan hasil ke session state
        st.session_state.ml_results = results
        st.session_state.ml_features = features
        st.session_state.ml_encoders = encoders
        st.session_state.ml_scaler = scaler
    elif selected_page == "🔮 Sales Prediction":
        show_prediction_interface(
            filtered_df,
            st.session_state.ml_results,
            st.session_state.ml_features,
            st.session_state.ml_encoders,
            st.session_state.ml_scaler
        )
    elif selected_page == "📊 Batch Prediction":
        show_batch_prediction(
            filtered_df,
            st.session_state.ml_results,
            st.session_state.ml_features,
            st.session_state.ml_encoders,
            st.session_state.ml_scaler
        )
    elif selected_page == "📋 Data Summary":
        show_data_summary(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | Superstore Analytics Dashboard")

if __name__ == "__main__":
    main()
