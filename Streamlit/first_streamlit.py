import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split


# ---------------------------------------------------
# XGBoost Model 
# ---------------------------------------------------
def run_xgboost(df, target):
    # Convert date column but don't use it
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # Ensure target is numeric
    if target not in numeric_df.columns:
        raise ValueError(f"Target column '{target}' must be numeric for XGBoost.")

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    # Time-ordered split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": preds
    })


# ---------------------------------------------------
# SARIMAX Model 
# ---------------------------------------------------
def run_sarimax(df, target, order=(1,1,1), seasonal_order=(1,1,1,12)):
    if "date" not in df.columns:
        raise ValueError("SARIMAX requires a 'date' column.")

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    y = df[target].astype(float)

    model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
    fit = model.fit(disp=False)

    forecast = fit.forecast(steps=30)

    return pd.DataFrame({
        "actual": y,
        "forecast": forecast
    })


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("Forecast Maker")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # ---------------------------------------------------
    # Determine Columns
    # ---------------------------------------------------
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    non_date_cols = [c for c in df.columns if c != 'date']

    model_choice = st.radio("Choose a model:", ["XGBoost", "SARIMAX"])

    # Determine default option (target)
    default_target = "target" if "target" in df.columns else None

    # Choose valid options for each model
    if model_choice == "XGBoost":
        selectable_cols = numeric_cols
    else:
        selectable_cols = non_date_cols

    # Ensure default is valid
    if default_target not in selectable_cols:
        default_target = selectable_cols[0]

    target = st.selectbox(
        "Select target column",
        selectable_cols,
        index=selectable_cols.index(default_target)
    )

    # ---------------------------------------------------
    # Run Model Button
    # ---------------------------------------------------
    if st.button("Run Model"):
        try:
            if model_choice == "XGBoost":
                st.subheader("Running XGBoost")
                results = run_xgboost(df, target)
                st.line_chart(results)

            else:
                st.subheader("Running SARIMAX")
                results = run_sarimax(df, target)
                st.line_chart(results)

        except Exception as e:
            st.error(f"Error: {e}")
