# Databricks notebook source
#-- Environment Setup
from pyspark.sql.functions import when, col
import pandas as pd
from pyspark.sql.functions import when
from pyspark.sql.functions import regexp_replace
from datetime import datetime

target_table = 'sandbox.cg_inverness.{}'
target_view = 'sandbox.cg_inverness.{}'

# COMMAND ----------

#-- Load in the managed view Campaign Table (Grab Only Single Player Games, that have either finished or appear to be abandoned)
df = spark.table(target_view.format('campaign_clustering_data_v2_civ7_turn100_adj'))
df = df.select(
    'CAMPAIGN_ID',
    'scouts',
    'settlers',
    'all_other_units',
    'max_gold_balance',
    'max_influence_balance',
    'max_settlementcap',
    'max_science',
    'max_production',
    'max_influence',
    'max_happiness',
    'max_gold',
    'max_food',
    'max_culture',
    'independent_dispersed',
    'attacks_made',
    'units_lost',
    'gamedifficulty',
    'enemies_defeated',
    'became_suzerain',
    'trade_routes',
    'settlements_founded',
    'saves',
    'maptype',
    'mapsize',
    'gamespeed',
    'display_service',
    'modded_games',
    'used_advanced_general_options',
    'buildversion'
)

#fact_player_ftue = fact_player_ftue.withColumn('days_since_last_interact', datediff(current_date() ,col(------last_seen-----)))


display(df)

# COMMAND ----------

from pyspark.sql.types import FloatType,IntegerType
from pyspark.sql.functions import col


#find all decimal columns in your SparkDF
decimals_cols = [c for c in df.columns if 'Decimal' in str(df.schema[c].dataType)] 

#convert all decimals columns to ints
for column in decimals_cols:
    df = df.withColumn(column, df[column].cast(IntegerType()))

display(df)    

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, MinMaxScaler,StringIndexer
from pyspark.sql.functions import col

numeric_features =[c for c in df.columns if str(df.schema[c].dataType) in ('IntegerType()', 'DoubleType()', 'LongType()', 'FloatType()', 'BooleanType()')]

categorical_features = ['maptype','gamedifficulty','display_service','buildversion','mapsize','gamespeed']


df = df.fillna('blank', subset=categorical_features)
df = df[['CAMPAIGN_ID']+numeric_features+categorical_features]

for numeric_col in numeric_features:
    df = df.withColumn(numeric_col, when((col(numeric_col) < 0) , 0)
                                         .otherwise(col(numeric_col)))
    

display(df)

# COMMAND ----------

from pyspark.sql.functions import col, exp
def iqr_outlier_treatment(dataframe, columns, factor=2.5):
    """
    Detects and treats outliers using IQR for multiple variables in a PySpark DataFrame.

    :param dataframe: The input PySpark DataFrame
    :param columns: A list of columns to apply IQR outlier treatment
    :param factor: The IQR factor to use for detecting outliers (default is 1.5)
    :return: The processed DataFrame with outliers treated
    """
    for column in columns:
        # Calculate Q1, Q3, and IQR
        quantiles = dataframe.approxQuantile(column, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        # Define the upper and lower bounds for outliers
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Filter outliers and update the DataFrame
        dataframe = dataframe.withColumn(column, when((col(column) < lower_bound) , lower_bound)
                                         .otherwise(col(column)))
        dataframe = dataframe.withColumn(column, when((col(column) > upper_bound), upper_bound)
                                         .otherwise(col(column)))

    return dataframe

# COMMAND ----------

df = iqr_outlier_treatment(df,numeric_features,factor=2.5)
df = df.fillna(0, subset=numeric_features)
display(df)

# COMMAND ----------

version = '2_civ7'
model_name = 'campaign_clustering_turn_100'

target_table = f'sandbox.cg_inverness.{model_name}_data_v{version}'

print(f'target_table: {target_table}')

# COMMAND ----------

(df
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(target_table))

# COMMAND ----------

dbutils.data.summarize(df[numeric_features])