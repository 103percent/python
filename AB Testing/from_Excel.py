# pip install pandas statsmodels openpyxl


import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# Load data
file_path = "ab_test_data.xlsx"  # Change to your actual file path
df = pd.read_excel(file_path)

# Ensure expected columns
assert {'group', 'converted'}.issubset(df.columns), "Missing required columns"

# Count conversions and total users per group
summary = df.groupby('group')['converted'].agg(['sum', 'count'])
conversions = summary['sum'].values
n_obs = summary['count'].values

# A/B Test using two-proportion z-test
stat, pval = proportions_ztest(count=conversions, nobs=n_obs)

# Print results
print("Conversion Summary:")
print(summary)
print("\nZ-test statistic: {:.4f}".format(stat))
print("P-value: {:.4f}".format(pval))

# Interpretation
alpha = 0.05
if pval < alpha:
    print("Reject the null hypothesis - there's a significant difference.")
else:
    print("Fail to reject the null - no significant difference detected.")