# Databricks notebook source
import random
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import mlflow
import mlflow.spark
import pyspark
import time

# COMMAND ----------

version = '2_civ7'
model_name = 'campaign_clustering_turn_100'

source_table = f'sandbox.cg_inverness.{model_name}_data_v2_features_v{version}'

#--

k_max = 7
random_seed_count = 3
version=1
experiment_name = f'/Users/jak.marshall@2k.com/Kmeans_turn_100_PCA_v2_{version}'

print(f'source_table:      {source_table}')
print(f'experiment_name:   {experiment_name}')


# COMMAND ----------

df = spark.table(source_table)
df.limit(5).toPandas()

# COMMAND ----------

from pyspark.sql.functions import size

num_rows = df.count()
num_columns = df[['features']].schema["features"].metadata["ml_attr"]["num_attrs"]
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

# COMMAND ----------

from pyspark.ml.feature import PCA

pca = PCA(k=21, inputCol="features")
pca.setOutputCol("pca_features")

model = pca.fit(df[['features']])

sum(model.explainedVariance)

# COMMAND ----------

model.setOutputCol("output")

transformed_features=model.transform(df)

# COMMAND ----------

display(transformed_features)

# COMMAND ----------

model_name = 'campaign_clustering_turn_100'
versionpca = '2_civ7'

target_table = f'sandbox.cg_inverness.{model_name}_data_v2_features_v2_PCA_v{versionpca}'

print(f'target_table: {target_table}')

(transformed_features[['output','CAMPAIGN_ID']]
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(target_table))