"""
Bank Churn Prediction - Streamlit Web Application

Interactive web app for predicting customer churn using trained ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from src.feature_engineering import engineer_all_features
from src.data_preprocessing import DataPreprocessor

# Page configuration
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_preprocessor():
    """Load the saved model and preprocessor."""
    try:
        # Try to load best model
        model_path = 'models/best_model_xgboost.pkl'
        if not os.path.exists(model_path):
            # Fallback to other models
            if os.path.exists('models/xgboost.pkl'):
                model_path = 'models/xgboost.pkl'
            elif os.path.exists('models/random_forest.pkl'):
                model_path = 'models/random_forest.pkl'
            elif os.path.exists('models/logistic_regression.pkl'):
                model_path = 'models/logistic_regression.pkl'
            else:
                return None, None, "No model found"
        
        model_data = joblib.load(model_path)
        
        # Extract model (could be in dict or direct)
        if isinstance(model_data, dict):
            if 'model' in model_data:
                model = model_data['model']
            else:
                model = model_data
        else:
            model = model_data
        
        # Load preprocessor
        preprocessor_path = 'models/preprocessor.pkl'
        if os.path.exists(preprocessor_path):
            preprocessor_dict = joblib.load(preprocessor_path)
        else:
            preprocessor_dict = None
        
        model_name = Path(model_path).stem.replace('best_model_', '').replace('_', ' ').title()
        return model, preprocessor_dict, model_name
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def create_customer_dataframe(input_data: dict) -> pd.DataFrame:
    """Create a DataFrame from user input."""
    # Create DataFrame with single row
    df = pd.DataFrame([input_data])
    return df


def preprocess_customer_data(df: pd.DataFrame, preprocessor_dict: dict) -> pd.DataFrame:
    """Preprocess customer data using the saved preprocessor."""
    try:
        # Apply feature engineering
        df_engineered = engineer_all_features(
            df,
            include_age_groups=True,
            include_balance_groups=True,
            include_tenure_groups=True,
            include_interactions=True,
            include_aggregated=True,
            include_polynomial=False
        )
        
        # Transform using preprocessor
        if preprocessor_dict and 'preprocessor' in preprocessor_dict:
            preprocessor_obj = preprocessor_dict['preprocessor']
            feature_names = preprocessor_dict.get('feature_names', [])
            
            # Remove target column if it exists
            if 'Exited' in df_engineered.columns:
                X = df_engineered.drop(columns=['Exited'])
            else:
                X = df_engineered
            
            # Use transform method
            if hasattr(preprocessor_obj, 'transform'):
                X_transformed = preprocessor_obj.transform(X)
                
                # Convert to DataFrame with correct feature names
                if feature_names and len(feature_names) == X_transformed.shape[1]:
                    X_df = pd.DataFrame(X_transformed, columns=feature_names)
                else:
                    # Fallback: use column names from X
                    X_df = pd.DataFrame(X_transformed, columns=X.columns)
                
                return X_df
            else:
                # If no transform method, return engineered features
                return X
        else:
            # If no preprocessor, just use engineered features
            if 'Exited' in df_engineered.columns:
                X = df_engineered.drop(columns=['Exited'])
            else:
                X = df_engineered
            return X
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def get_risk_level(probability: float) -> tuple:
    """Determine risk level based on churn probability."""
    if probability >= 0.7:
        return "High", "🔴"
    elif probability >= 0.4:
        return "Medium", "🟡"
    else:
        return "Low", "🟢"


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Bank Churn Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🔮 Predict Churn", "📈 Model Information", "💡 Insights & Analysis"]
    )
    
    # Load model and preprocessor
    model, preprocessor, model_name = load_model_and_preprocessor()
    
    if model is None:
        st.error("❌ Could not load the trained model. Please ensure models are saved in the 'models/' directory.")
        st.info("💡 Run the modeling notebook first to train and save models.")
        return
    
    # Main content based on selected page
    if page == "🔮 Predict Churn":
        predict_churn_page(model, preprocessor, model_name)
    elif page == "📈 Model Information":
        model_info_page(model_name)
    elif page == "💡 Insights & Analysis":
        insights_page()


def predict_churn_page(model, preprocessor, model_name):
    """Prediction page."""
    st.markdown('<h2 class="sub-header">Customer Churn Prediction</h2>', unsafe_allow_html=True)
    st.markdown("Enter customer information below to predict churn probability.")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Personal Information")
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", min_value=18, max_value=100, value=40, step=1)
        
        st.markdown("### Account Information")
        tenure = st.slider("Tenure (years with bank)", min_value=0, max_value=10, value=5, step=1)
        balance = st.number_input("Balance ($)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
        num_products = st.slider("Number of Products", min_value=1, max_value=4, value=1, step=1)
    
    with col2:
        st.markdown("### Financial Information")
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650, step=10)
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
        
        st.markdown("### Account Status")
        has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_credit_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active_member == "Yes" else 0,
            'EstimatedSalary': estimated_salary,
            'Exited': 0  # Placeholder, will be removed
        }
        
        # Create DataFrame and preprocess
        df = create_customer_dataframe(input_data)
        X_processed = preprocess_customer_data(df, preprocessor)
        
        if X_processed is not None:
            try:
                # Make prediction
                prediction_proba = model.predict_proba(X_processed)[0]
                churn_probability = prediction_proba[1]  # Probability of churn
                prediction = model.predict(X_processed)[0]
                
                # Display results
                st.markdown("---")
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                # Main prediction result
                col_result1, col_result2 = st.columns([2, 1])
                
                with col_result1:
                    risk_level, risk_emoji = get_risk_level(churn_probability)
                    st.markdown(f"### {risk_emoji} Churn Risk: **{risk_level}**")
                    st.markdown(f"**Churn Probability:** {churn_probability:.1%}")
                    st.markdown(f"**Prediction:** {'⚠️ Customer will CHURN' if prediction == 1 else '✅ Customer will RETAIN'}")
                
                with col_result2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = churn_probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Risk %"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed breakdown
                st.markdown("### 📊 Detailed Analysis")
                
                col_break1, col_break2, col_break3 = st.columns(3)
                
                with col_break1:
                    st.metric("Retention Probability", f"{(1 - churn_probability):.1%}")
                
                with col_break2:
                    st.metric("Churn Probability", f"{churn_probability:.1%}")
                
                with col_break3:
                    confidence = abs(churn_probability - 0.5) * 2  # How far from 50/50
                    st.metric("Prediction Confidence", f"{confidence:.1%}")
                
                # Feature importance insights
                st.markdown("### 💡 Key Insights")
                
                insights = []
                if age > 50:
                    insights.append(f"⚠️ Age ({age}) is above average, which may increase churn risk")
                if balance == 0:
                    insights.append("⚠️ Zero balance may indicate inactive account")
                if is_active_member == "No":
                    insights.append("⚠️ Inactive membership is a strong churn indicator")
                if num_products == 1:
                    insights.append("💡 Low product engagement - consider upselling")
                if credit_score < 600:
                    insights.append("⚠️ Lower credit score may correlate with churn")
                if tenure < 2:
                    insights.append("⚠️ New customer (low tenure) - higher churn risk")
                if geography == "Germany":
                    insights.append("💡 German customers historically show different churn patterns")
                
                if insights:
                    for insight in insights:
                        st.info(insight)
                else:
                    st.success("✅ Customer profile shows positive retention indicators")
                
                # Probability distribution visualization
                st.markdown("### 📈 Probability Distribution")
                prob_df = pd.DataFrame({
                    'Outcome': ['Retain', 'Churn'],
                    'Probability': [1 - churn_probability, churn_probability]
                })
                
                fig_bar = px.bar(
                    prob_df,
                    x='Outcome',
                    y='Probability',
                    color='Outcome',
                    color_discrete_map={'Retain': '#27ae60', 'Churn': '#e74c3c'},
                    text='Probability',
                    title="Churn vs Retention Probability"
                )
                fig_bar.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check that all required features are present in the model.")


def model_info_page(model_name):
    """Model information page."""
    st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
    
    st.markdown("### 🤖 Model Details")
    st.info(f"**Current Model:** {model_name}")
    st.info("This model was trained on historical customer data to predict churn probability.")
    
    st.markdown("### 📊 Model Performance")
    
    # Performance metrics (you can update these with actual values from your evaluation)
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    
    with col_met1:
        st.metric("Accuracy", "~86%", "High")
    
    with col_met2:
        st.metric("ROC-AUC", "~0.86", "Excellent")
    
    with col_met3:
        st.metric("Precision", "~72%", "Good")
    
    with col_met4:
        st.metric("Recall", "~49%", "Moderate")
    
    st.markdown("### 🔍 How It Works")
    st.markdown("""
    1. **Input Collection**: Customer data is collected through the prediction form
    2. **Feature Engineering**: Additional features are created (age groups, interactions, etc.)
    3. **Preprocessing**: Data is encoded and scaled using the same pipeline as training
    4. **Prediction**: The trained model predicts churn probability
    5. **Interpretation**: Results are displayed with risk levels and insights
    """)
    
    st.markdown("### 📋 Features Used")
    st.markdown("""
    The model uses the following customer features:
    - **Demographics**: Age, Geography, Gender
    - **Financial**: Credit Score, Balance, Estimated Salary
    - **Account**: Tenure, Number of Products, Credit Card Status
    - **Engagement**: Active Member Status
    - **Engineered**: Age Groups, Balance Groups, Interaction Features, Aggregated Scores
    """)


def insights_page():
    """Insights and analysis page."""
    st.markdown('<h2 class="sub-header">Insights & Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Understanding Churn Prediction")
    
    st.markdown("""
    #### What is Customer Churn?
    Customer churn refers to customers who stop using a company's products or services.
    Predicting churn helps banks:
    - **Retain valuable customers** through targeted interventions
    - **Reduce revenue loss** by identifying at-risk customers early
    - **Optimize marketing efforts** by focusing on high-risk segments
    """)
    
    st.markdown("### 🎯 Risk Levels Explained")
    
    col_risk1, col_risk2, col_risk3 = st.columns(3)
    
    with col_risk1:
        st.markdown("#### 🟢 Low Risk (0-40%)")
        st.success("Customer is likely to stay. Standard retention strategies.")
    
    with col_risk2:
        st.markdown("#### 🟡 Medium Risk (40-70%)")
        st.warning("Customer may churn. Consider engagement campaigns or special offers.")
    
    with col_risk3:
        st.markdown("#### 🔴 High Risk (70-100%)")
        st.error("High churn probability. Immediate intervention recommended (retention offers, account review).")
    
    st.markdown("### 💡 Key Factors Affecting Churn")
    
    factors = {
        "Age": "Older customers (50+) tend to have higher churn rates",
        "Active Membership": "Inactive members are significantly more likely to churn",
        "Tenure": "New customers (< 2 years) have higher churn risk",
        "Product Engagement": "Customers with only 1 product are more likely to churn",
        "Geography": "Different regions show varying churn patterns",
        "Balance": "Zero balance accounts indicate potential inactivity"
    }
    
    for factor, explanation in factors.items():
        with st.expander(f"📌 {factor}"):
            st.write(explanation)
    
    st.markdown("### 🎓 Best Practices")
    st.markdown("""
    1. **Monitor High-Risk Customers**: Regularly check predictions for customers with >70% churn probability
    2. **Personalized Interventions**: Use insights to tailor retention strategies
    3. **Track Predictions**: Compare predictions with actual outcomes to improve model
    4. **Combine with Domain Knowledge**: Use predictions as one tool alongside business expertise
    """)


if __name__ == "__main__":
    main()

