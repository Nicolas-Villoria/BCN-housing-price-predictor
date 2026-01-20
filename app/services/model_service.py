import streamlit as st
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
import os
import sys
import pandas as pd
from delta import configure_spark_with_delta_pip

# Constants
MODEL_NAME = "BarcelonaRentalPriceModel"
MLFLOW_TRACKING_URI = "file:./mlruns"

@st.cache_resource(show_spinner="Initializing Spark Engine (this happens once)...")
def init_spark():
    """
    Initializes a persistent Spark Session with Delta Lake support.
    Cached resource: Only runs once per application lifetime.
    """
    try:
        # Force Spark to bind to localhost to avoid networking issues
        os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

        builder = SparkSession.builder \
            .appName("Streamlit_Inference_Engine") \
            .master("local[*]") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.ui.showConsoleProgress", "false") \
            .config("spark.logConf", "true")

        # Use configure_spark_with_delta_pip to handle dependencies
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        
        return spark
    except Exception as e:
        st.error(f"Failed to initialize Spark: {e}")
        return None

@st.cache_resource(show_spinner="Loading ML Model...")
def load_model():
    """
    Loads the MLflow model. Tries 'Staging' then 'Production' stages.
    Returns: The loaded Spark PipelineModel.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    stages_to_try = ["Staging", "Production", "None"]
    
    for stage in stages_to_try:
        model_uri = f"models:/{MODEL_NAME}/{stage}"
        try:
            model = mlflow.spark.load_model(model_uri)
            print(f"Successfully loaded model from {model_uri}")
            return model
        except Exception:
            continue
            
    # Fallback: Try to find the best run from experiment
    try:
        print("Model Registry lookup failed. Searching for best run...")
        experiment = mlflow.get_experiment_by_name("bda_project_price_prediction")
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="metrics.rmse < 200000",
                order_by=["metrics.rmse ASC"],
                max_results=1
            )
            if not runs.empty:
                run_id = runs.iloc[0].run_id
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.spark.load_model(model_uri)
                print(f"Loaded best model from run {run_id}")
                return model
    except Exception as e:
        print(f"Fallback lookup failed: {e}")

    return None

def predict_price(spark, model, features_dict):
    """
    Performs inference using the loaded Spark model.
    
    Args:
        spark: Active SparkSession
        model: Loaded PipelineModel
        features_dict: Dictionary containing:
            - size (double)
            - rooms (int)
            - bathrooms (int)
            - neighborhood (string)
            - propertyType (string)
            - district (string)
            - avg_income_index (double)
            - density_val (double)
            
    Returns:
        float: Predicted price
    """
    if not spark or not model:
        return None

    # Define schema explicitly to match model expectations
    # Note: The model pipeline expects 'features_raw' to be assembled from these columns
    # We must provide a DataFrame with these columns.
    
    schema = StructType([
        StructField("size", DoubleType(), True),
        StructField("rooms", IntegerType(), True),
        StructField("bathrooms", IntegerType(), True),
        StructField("neighborhood", StringType(), True),
        StructField("propertyType", StringType(), True),
        StructField("district", StringType(), True),
        StructField("avg_income_index", DoubleType(), True),
        StructField("density_val", DoubleType(), True)
    ])
    
    # Create DataFrame
    data = [(
        float(features_dict['size']),
        int(features_dict['rooms']),
        int(features_dict['bathrooms']),
        str(features_dict['neighborhood']),
        str(features_dict['propertyType']),
        str(features_dict['district']),
        float(features_dict['avg_income_index']),
        float(features_dict['density_val'])
    )]
    
    df = spark.createDataFrame(data, schema)
    
    # Run Inference
    prediction_df = model.transform(df)
    
    # Extract prediction
    try:
        result = prediction_df.select("prediction").collect()[0][0]
        return max(0, float(result)) # Ensure no negative prices
    except Exception as e:
        print(f"Inference error: {e}")
        return 0.0
