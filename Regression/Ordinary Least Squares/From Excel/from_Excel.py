import pandas as pd
import statsmodels.api as sm

# Load the data
file_path = 'regression_data.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Independent and dependent variables
X = df['age']
y = df['spend']

# Add a constant (intercept) to the predictor
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Output results
print(model.summary())