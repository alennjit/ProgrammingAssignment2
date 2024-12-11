import os
import sys
import json
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Wine Quality Prediction") \
    .getOrCreate()

# Input and output paths (these should be configured to your environment)
data_input_path = "s3://alensfilebucket/input-data/"
model_input_path = "s3://alensfilebucket/models/wine-quality-model/"
output_path = "s3://alensfilebucket/output-data/"

# Load the dataset
try:
    data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(data_input_path)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Feature columns (these should match the features used during training)
feature_columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                   'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                   'pH', 'sulphates', 'alcohol']

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Load the pre-trained model
try:
    model = RandomForestRegressionModel.load(model_input_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Make predictions
predictions = model.transform(data)

# Select necessary columns and write to output
output = predictions.select("quality", "prediction")
try:
    output.write.format("csv").mode("overwrite").save(output_path)
    print(f"Predictions successfully saved to {output_path}")
except Exception as e:
    print(f"Error saving output: {e}")
    sys.exit(1)

# Stop Spark session
spark.stop()
