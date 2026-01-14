# Databricks notebook source
from pyspark.sql.functions import when, col,try_divide
import pandas as pd

# COMMAND ----------

version = '2_civ7'
model_name = 'campaign_clustering_turn_100'

target_table = f'sandbox.cg_inverness.{model_name}_data_v{version}'

print(f'target_table: {target_table}')

# COMMAND ----------

df = spark.table(target_table)
df.limit(5).toPandas()

# COMMAND ----------

# - Space for developing numeric features

from pyspark.sql import functions as F
#df = df.withColumn('TURNS_PER_ACTIVE_DAY', F.expr('TRY_DIVIDE(NUM_TURNS_PLAYED, NUM_DAYS_PLAYED)'))
#df = df.withColumn('AGES_PER_ACTIVE_DAY', F.expr('TRY_DIVIDE(NUM_AGES_PLAYED, NUM_DAYS_PLAYED)'))
#df = df.withColumn('TURNS_PER_ACTIVE_DAY', F.expr('TRY_DIVIDE(NUM_TURNS_PLAYED, NUM_SESSIONS_PLAYED)'))
#df = df.withColumn('AGES_PER_SESSION', F.expr('TRY_DIVIDE(NUM_AGES_PLAYED, NUM_SESSIONS_PLAYED)'))

# COMMAND ----------

display(df)

# COMMAND ----------

[(c,df.schema[c].dataType) for c in df.columns ]

# COMMAND ----------

numeric_features =[c for c in df.columns if  'Integer' in str(df.schema[c].dataType) or 'Double' in str(df.schema[c].dataType)]

categorical_features = ['maptype','display_service','gamedifficulty','buildversion','mapsize','gamespeed']

# COMMAND ----------

import numpy as np
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

indexers = [StringIndexer(inputCol=cat, outputCol='{}_indexed'.format(cat),handleInvalid='skip') for cat in categorical_features]

encoder = OneHotEncoder(inputCols=[indexer.getOutputCol() for indexer in indexers], outputCols=[f"{indexer.getOutputCol()}_onehot" for indexer in indexers])

assembler_cat = VectorAssembler(inputCols=[f"{indexer.getOutputCol()}_onehot" for indexer in indexers], outputCol='categorical_indexed_vec')

vector_assembler_num = VectorAssembler(inputCols=numeric_features, outputCol='numeric_assembled')
scaler = StandardScaler(inputCol="numeric_assembled", outputCol="scaled_features") 

assembler = VectorAssembler(inputCols=['scaled_features', 'categorical_indexed_vec'], outputCol='features')

pipeline = Pipeline(stages=[*indexers,encoder, assembler_cat,vector_assembler_num, scaler, assembler])

fit_model = pipeline.fit(df.replace('', None).dropna(subset=categorical_features).fillna(0,subset=numeric_features))

features = fit_model.transform(df.replace('', None).dropna(subset=categorical_features).fillna(0,subset=numeric_features)) 

# COMMAND ----------

display(features)

# COMMAND ----------

features[['CAMPAIGN_ID','features']].show()

# COMMAND ----------

features.count()

# COMMAND ----------

#from https://stackoverflow.com/questions/72728208/retrieve-categories-from-spark-ml-onehotencoder
# 

def clean_varname(name):
    if '_' not in name.split('onehot_')[1]:
        return name.replace('categorical_indexed_vec_','').replace('_indexed_onehot','')
    else:
        return name.split('onehot_')[1]
    
def get_input_cols(fitted_pipeline, test_df):    
    schema_attrs = (
        fitted_pipeline
        .transform(df)
        .select("features")
        .schema[0]
        .metadata
        .get('ml_attr')
        .get('attrs')
    )

    numeric_attr = schema_attrs.get('numeric') or []
    nominal_attr = schema_attrs.get('nominal') or []
    binary_attr = schema_attrs.get('binary') or []

    numeric = [(item["idx"], item["name"]) for item in numeric_attr]
    binary = [(item["idx"], clean_varname(item["name"])) for item in binary_attr]
    nominal = [(item["idx"], item["name"]) for item in nominal_attr]

    all_attributes = numeric + binary + nominal
    all_attributes = sorted(all_attributes, key=lambda x: x[0])
    
    return all_attributes

feature_names = get_input_cols(fit_model,df)

num_features =fit_model.stages[-3].getInputCols()

feature_names=[(num,num_features[num]) for num in range(len(num_features))] + feature_names[len(num_features):]

# COMMAND ----------

display(feature_names)

# COMMAND ----------


from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.ml.stat import Correlation
#dataset = spark.createDataFrame(dataset, ['features'])
pearsonCorr = Correlation.corr(features, 'features', 'pearson').collect()[0][0]


# COMMAND ----------


import seaborn as sns
import matplotlib.pyplot as plt
x_axis_labels =[name[1] for name in feature_names][:len(num_features)]
y_axis_labels = [name[1] for name in feature_names][:len(num_features)]
sns.heatmap(pd.DataFrame(pearsonCorr.toArray()[:len(num_features),:len(num_features)]),xticklabels=x_axis_labels, yticklabels=y_axis_labels,center=0,cmap='vlag')
plt.show()


# COMMAND ----------

version = '2_civ7'
model_name = 'campaign_clustering_turn_100'

target_table = f'sandbox.cg_inverness.{model_name}_data_v2_features_v{version}'

print(f'target_table: {target_table}')

(features[['features','CAMPAIGN_ID']]
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(target_table))