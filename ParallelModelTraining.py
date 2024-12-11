import sys
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

TRAINING_DATASET = "s3://alensfilebucket/TrainingDataset.csv"
VALIDATION_DATASET = "s3://alensfilebucket/ValidationDataset.csv"
PIPELINE_PATH = "task-assignment-222-njit-spark-ml-model"

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("WineQualityModelTraining") \
        .getOrCreate()

    # Load training dataset
    train_data = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("delimiter", ";") \
        .load(TRAINING_DATASET)

    # Clean column names by removing excessive double quotes
    for col_name in train_data.columns:
        train_data = train_data.withColumnRenamed(col_name, col_name.replace('"', ''))

    # Split training and test datasets
    train_data, test_data = train_data.randomSplit([0.85, 0.15], seed=42)

    # Feature assembly
    feature_columns = [col for col in train_data.columns if col != "quality"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Define RandomForestClassifier with further optimized parameters
    rf_classifier = RandomForestClassifier(
        labelCol="quality", 
        featuresCol="features", 
        maxDepth=15, 
        maxBins=48, 
        numTrees=200, 
        seed=42
    )

    # Build the pipeline
    pipeline = Pipeline(stages=[assembler, rf_classifier])

    # Fit the model
    model = pipeline.fit(train_data)

    # Make predictions
    predictions = model.transform(test_data)

    # Evaluate predictions
    evaluator = MulticlassClassificationEvaluator(
        labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)

    # Save the model
    model.write().overwrite().save(PIPELINE_PATH)

    print(f"Further Optimized F1 Score: {f1_score * 100:.2f}%")

    # Stop SparkSession
    spark.stop()
