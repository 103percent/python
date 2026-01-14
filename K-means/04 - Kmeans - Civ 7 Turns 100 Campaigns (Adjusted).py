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
import pandas as pd

# COMMAND ----------

version = '2_civ7'
model_name = 'campaign_clustering_turn_100'

source_table =  f'sandbox.cg_inverness.{model_name}_data_v2_features_v2_PCA_v{version}'


run_title = 'turn_100_clustering_v2'
k_max = 10
random_seed_count = 3

experiment_name = f'/Users/jak.marshall@2k.com/Kmeans_turn_100_PCA_v2_2'

print(f'source_table:      {source_table}')
print(f'experiment_name:   {experiment_name}')

# COMMAND ----------

df = spark.table(source_table)
df.limit(5).toPandas()

# COMMAND ----------

df.count()

# COMMAND ----------

def get_dtype(df,colname):
    return [dtype for name, dtype in df.dtypes if name == colname][0]

# COMMAND ----------

mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
print(experiment.experiment_id)

# COMMAND ----------

'''
%sql
CREATE VOLUME sandbox.cg_inverness.my_volume2;
'''

# COMMAND ----------

dfs_tmpdir = "/Volumes/sandbox/cg_inverness/my_volume2/mlflow_tmp"

# COMMAND ----------

# Settings
k_max=6
k_range = range(2, k_max+1)

random_seed_count=2
seeds = [random.randint(1000, 9999) for _ in range(random_seed_count)]

subset = df
for random_seed in seeds:

    # Run
    run_name = f'{run_title}_{random_seed}'

    # Vars
    wssse_scores = []
    silhouette_scores = []
    child_run_ids = []
    run_ids = []

    evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='output', metricName='silhouette')
    tags = {"mlflow.runName": run_name}

    # Training Runs
    with mlflow.start_run(experiment_id=experiment.experiment_id, tags=tags) as run:
        run_id = run.info.run_id
        run_ids.append(run_id)

        # Child Runs
        for k in k_range:
            with mlflow.start_run(nested=True) as child_run:
                child_run_id = child_run.info.run_id
                child_run_ids.append(child_run_id)

                # Train
                kmeans = KMeans().setK(k).setSeed(random_seed).setFeaturesCol("output")
                model = kmeans.fit(subset[['output','CAMPAIGN_ID']])

                # Predict
                predictions = model.transform(subset[['output','CAMPAIGN_ID']])

                # Eval
                silhouette = evaluator.evaluate(predictions)
                silhouette_scores.append(silhouette)

                wssse = model.summary.trainingCost
                wssse_scores.append(wssse)

                
                #Log
                mlflow.spark.log_model(
                    model,
                    "kmeans_model",
                    pip_requirements=['pyspark=='+pyspark.__version__],
                    input_example = None,
                    signature = None,
                    dfs_tmpdir=dfs_tmpdir
                )
                mlflow.log_param("k", k)
                mlflow.log_param("seed", random_seed)
                mlflow.set_tag("model_type", "kmeans")
                mlflow.set_tag("run_name", run_name) # Experiment vizualition workaround (group by tag run_name)
                mlflow.log_metric("k", k) # Experiment vizualition workaround (x-axis metric k)
                mlflow.log_metric("silhouette_score", silhouette)
                mlflow.log_metric("wsse_score", wssse)
                
                # Log
                print(f'random_seed: {random_seed}')
                print(f"k:           {k}")
                print(f"id:          {child_run_id}")
                print(f"wssse:       {wssse}")
                print(f"silhouette:  {silhouette}")


# COMMAND ----------

original_data = spark.table(f'sandbox.cg_inverness.campaign_clustering_turn_100_data_v2_civ7')
original_data.limit(5).toPandas()

# COMMAND ----------

import os
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/sandbox/cg_inverness/my_volume/mlflow_tmp"

# COMMAND ----------

import mlflow
logged_model = 'runs:/8150c6ad4f8646d5922e8229521005d2/kmeans_model' 
# Load model
loaded_model = mlflow.spark.load_model(logged_model)


first_split_predictions = loaded_model.transform(df[['output','CAMPAIGN_ID']])

# COMMAND ----------

first_split_predictions.groupby('prediction').count().show()

# COMMAND ----------

from pyspark.sql.functions import median,mode,mean,min,max,percentile_approx

kmeans2 = original_data.join(first_split_predictions,original_data.CAMPAIGN_ID == first_split_predictions.CAMPAIGN_ID)

# COMMAND ----------

for col in original_data.columns:
    print(col)
    if kmeans2.schema[col].dataType.simpleString() in ['int', 'double']:
        kmeans2.groupby('prediction').agg(median(col),mean(col),percentile_approx(col,.25),percentile_approx(col,.75)).show()

# COMMAND ----------

for col in original_data.columns:
    if kmeans2.schema[col].dataType.simpleString() not in ['int', 'double'] and col not in ('CAMPAIGN_ID'):
        print(col)
        counts=kmeans2.groupby(col).pivot('prediction').count().toPandas()
        print(counts)
        print(counts.iloc[:,1:].div(counts.iloc[:,1:].sum(axis=0),axis='columns'))

# COMMAND ----------

