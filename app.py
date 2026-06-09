import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# PAGE CONFIG
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")
st.title("🚗 Car Price Prediction & Analysis")
st.markdown("---")


# LOAD DATA
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("car data.csv")
        df['discount_price'] = df['Present_Price'] - df['Selling_Price']
        return df
    except FileNotFoundError:
        return None


df = load_data()

if df is None:
    st.error("❌ Error: 'car data.csv' not found. Place it in the same directory as script.")
    st.stop()

st.success("✅ Data loaded successfully!")

# SIDEBAR NAVIGATION
with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio("Select Page:",
                    ["📊 Data Overview", "📈 EDA", "🤖 Model Training", "📉 Results", "🔮 Predictions"])

# PAGE 1: DATA OVERVIEW
if page == "📊 Data Overview":
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.2f} KB")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Types")
        st.write(df.dtypes)

    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

# PAGE 2: EXPLORATORY DATA ANALYSIS
elif page == "📈 EDA":
    st.header("📈 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Categorical", "Numerical", "Relationships"])

    # CATEGORICAL ANALYSIS
    with tab1:
        st.subheader("Categorical Variables Analysis")
        categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission"]

        for col in categorical_cols:
            st.markdown(f"### {col}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(df[col].value_counts())

            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[col].value_counts().plot(kind="bar", color="skyblue", ax=ax)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

    # NUMERICAL ANALYSIS
    with tab2:
        st.subheader("Numerical Variables Analysis")
        numerical_cols = ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'discount_price']

        for col in numerical_cols:
            st.markdown(f"### {col}")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Mean", f"{df[col].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[col].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[col].std():.2f}")
            with col4:
                st.metric("Min", f"{df[col].min():.2f}")
            with col5:
                st.metric("Max", f"{df[col].max():.2f}")

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            axes[0].hist(df[col], bins=30, color='skyblue', edgecolor='black')
            axes[0].set_title(f"Distribution of {col}")
            axes[0].set_xlabel(col)

            axes[1].boxplot(df[col])
            axes[1].set_title(f"Boxplot of {col}")
            axes[1].set_ylabel(col)

            plt.tight_layout()
            st.pyplot(fig)

    # RELATIONSHIPS
    with tab3:
        st.subheader("Relationships & Aggregations")

        # Sales by Fuel Type
        st.markdown("#### Sales by Fuel Type")
        fuel_sales = df.groupby("Fuel_Type")["Selling_Price"].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(fuel_sales)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(fuel_sales.values, labels=fuel_sales.index, autopct="%1.1f%%")
            ax.set_title("Sales by Fuel Type")
            st.pyplot(fig)

        # KMs Driven by Fuel Type
        st.markdown("#### KMs Driven by Fuel Type")
        kms_driven = df.groupby("Fuel_Type")["Kms_Driven"].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(kms_driven)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(kms_driven.values, labels=kms_driven.index, autopct="%1.1f%%")
            ax.set_title("KMs Driven by Fuel Type")
            st.pyplot(fig)

        # Sales by Seller Type
        st.markdown("#### Sales by Seller Type")
        seller_sales = df.groupby("Seller_Type")["Selling_Price"].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(seller_sales)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(seller_sales.values, labels=seller_sales.index, autopct="%1.1f%%")
            ax.set_title("Sales by Seller Type")
            st.pyplot(fig)

        # Sales by Transmission
        st.markdown("#### Sales by Transmission Type")
        trans_sales = df.groupby("Transmission")["Selling_Price"].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(trans_sales)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(trans_sales.values, labels=trans_sales.index, autopct="%1.1f%%")
            ax.set_title("Sales by Transmission")
            st.pyplot(fig)

# PAGE 3: MODEL TRAINING
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")

    # Data Preprocessing
    st.subheader("📋 Data Preprocessing")

    df_model = df.copy()

    # Encode categorical variables
    st.info("Encoding categorical variables: Car_Name, Fuel_Type, Seller_Type, Transmission")

    oe = OrdinalEncoder()
    categorical_features = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]
    df_model[categorical_features] = oe.fit_transform(df_model[categorical_features])

    st.success("✅ Categorical variables encoded!")

    # Feature selection
    st.subheader("🎯 Feature Selection")
    features = ["Car_Name", "Year", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner", "discount_price"]
    target = "Selling_Price"

    st.write(f"**Features:** {features}")
    st.write(f"**Target:** {target}")

    X = df_model[features]
    y = df_model[target]

    # Train-Test Split
    st.subheader("📊 Train-Test Split")
    test_size = st.slider("Test Size (%):", 10, 40, 20) / 100
    random_state = st.number_input("Random State:", value=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train Size", len(X_train))
    with col2:
        st.metric("Test Size", len(X_test))
    with col3:
        st.metric("Train (%)", f"{(len(X_train) / len(X) * 100):.1f}%")
    with col4:
        st.metric("Test (%)", f"{(len(X_test) / len(X) * 100):.1f}%")

    # Model Training
    st.subheader("🚀 Training Models")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Linear Regression**")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        st.success("✅ Linear Regression trained!")

    with col2:
        st.write("**Random Forest Regressor**")
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        st.success("✅ Random Forest trained!")

    # Feature Importance
    st.subheader("📊 Feature Importance (Random Forest)")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance - Random Forest')
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(feature_importance, use_container_width=True)

    # Store in session state for later use
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.y_pred_lr = y_pred_lr
    st.session_state.y_pred_rf = y_pred_rf
    st.session_state.lr_model = lr
    st.session_state.rf_model = rf

# PAGE 4: RESULTS
elif page == "📉 Results":
    st.header("📉 Model Evaluation Results")

    if 'y_pred_lr' not in st.session_state:
        st.warning("⚠️ Please train models first in the 'Model Training' page")
        st.stop()

    y_test = st.session_state.y_test
    y_pred_lr = st.session_state.y_pred_lr
    y_pred_rf = st.session_state.y_pred_rf

    # Calculate metrics
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    # Display metrics
    st.subheader("📊 Linear Regression Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MSE", f"{mse_lr:.2f}")
    with col2:
        st.metric("MAE", f"{mae_lr:.2f}")
    with col3:
        st.metric("R² Score", f"{r2_lr:.4f}")

    st.subheader("📊 Random Forest Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MSE", f"{mse_rf:.2f}")
    with col2:
        st.metric("MAE", f"{mae_rf:.2f}")
    with col3:
        st.metric("R² Score", f"{r2_rf:.4f}")

    # Comparison
    st.subheader("📊 Model Comparison")
    metrics_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest'],
        'MSE': [mse_lr, mse_rf],
        'MAE': [mae_lr, mae_rf],
        'R² Score': [r2_lr, r2_rf]
    })
    st.dataframe(metrics_df, use_container_width=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred_rf, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Random Forest: Actual vs Predicted')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred_lr, alpha=0.5, color='orange')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Linear Regression: Actual vs Predicted')
        plt.tight_layout()
        st.pyplot(fig)

# PAGE 5: PREDICTIONS
elif page == "🔮 Predictions":
    st.header("🔮 Make Predictions")

    if 'rf_model' not in st.session_state:
        st.warning("⚠️ Please train models first in the 'Model Training' page")
        st.stop()

    rf_model = st.session_state.rf_model

    st.subheader("Enter Car Details")

    col1, col2 = st.columns(2)

    with col1:
        car_name = st.selectbox("Car Name:", df['Car_Name'].unique())
        year = st.number_input("Year:", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()),
                               value=int(df['Year'].median()))
        kms_driven = st.number_input("KMs Driven:", min_value=0, value=50000)

    with col2:
        fuel_type = st.selectbox("Fuel Type:", df['Fuel_Type'].unique())
        seller_type = st.selectbox("Seller Type:", df['Seller_Type'].unique())
        transmission = st.selectbox("Transmission:", df['Transmission'].unique())

    col3, col4 = st.columns(2)

    with col3:
        owner = st.number_input("Number of Owners:", min_value=0, max_value=3, value=0)

    with col4:
        present_price = st.number_input("Present Price:", min_value=0.0, value=5.0)

    discount_price = present_price - 0  # Will be calculated

    if st.button("🔮 Predict Price", key="predict_button"):
        # Prepare data for prediction
        from sklearn.preprocessing import OrdinalEncoder

        df_temp = df.copy()
        df_temp['discount_price'] = df_temp['Present_Price'] - df_temp['Selling_Price']

        oe = OrdinalEncoder()
        categorical_features = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]
        df_temp[categorical_features] = oe.fit_transform(df_temp[categorical_features])

        # Create prediction input
        input_data = pd.DataFrame({
            'Car_Name': [oe.categories_[0].tolist().index(car_name) if car_name in oe.categories_[0] else 0],
            'Year': [year],
            'Kms_Driven': [kms_driven],
            'Fuel_Type': [oe.categories_[1].tolist().index(fuel_type) if fuel_type in oe.categories_[1] else 0],
            'Seller_Type': [oe.categories_[2].tolist().index(seller_type) if seller_type in oe.categories_[2] else 0],
            'Transmission': [
                oe.categories_[3].tolist().index(transmission) if transmission in oe.categories_[3] else 0],
            'Owner': [owner],
            'discount_price': [discount_price]
        })

        prediction = rf_model.predict(input_data)[0]

        st.success(f"✅ Predicted Selling Price: ₹{prediction:.2f} Lakhs")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Price", f"₹{prediction:.2f}L")
        with col2:
            st.metric("Present Price", f"₹{present_price:.2f}L")
        with col3:
            if present_price > prediction:
                discount = ((present_price - prediction) / present_price) * 100
                st.metric("Expected Discount", f"{discount:.2f}%")

st.markdown("---")
st.markdown("© 2024 Car Price Prediction System | Built with Streamlit")