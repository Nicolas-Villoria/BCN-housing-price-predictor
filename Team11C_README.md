# Barcelona Rental Price Estimation System

## Overview

We developed an end-to-end machine learning system to estimate property rental prices in Barcelona. The project integrates real estate listings with public socioeconomic data to provide granular valuations based on property characteristics and neighborhood context. A Streamlit-based web application serves these predictions to end-users, transforming raw model outputs into actionable business intelligence.

## System Architecture

The system follows a medallion data lake architecture to ensure data quality and traceability. We use Apache Spark for distributed processing and orchestrate the pipeline with Apache Airflow.

Raw data from Idealista and Open Data BCN lands in the Bronze layer as immutable Parquet files. The Silver layer cleans and standardizes this data, storing it in MongoDB to handle schema variability in property listings. The Gold layer performs feature engineering and joins, persisting the analytical dataset in Delta Lake to support ACID transactions and time travel.

Models are trained using PySpark MLlib, with experiments tracked and registered in MLflow. The best-performing model is deployed to a Model Registry, which the user-facing application queries for inference.

## Data & Feature Engineering

We sourced property listings from Idealista to capture market supply, and enriched this with population density and income indices from Barcelona Open Data. This enrichment is critical as location is a primary driver of real estate value.

Data quality is enforced at the Silver layer. We implement deduplication strategies to handle recurring listings and normalize diverse neighborhood naming conventions using a custom lookup table.

In the Gold layer, we engineer features such as price per square meter and outlier detection thresholds. We use broadcast joins to efficiently merge the high-volume listing data with smaller socioeconomic lookup tables, minimizing shuffle overhead.

## Modeling Approach

We trained three distinct model architectures—Linear Regression, Random Forest, and Gradient Boosted Trees—to balance interpretability with predictive power. Linear Regression provides a strong baseline, while tree-based ensembles capture non-linear interactions between location and property features.

Hyperparameter tuning is automated via grid search with 3-fold cross-validation. We evaluate models primarily on RMSE to penalize large errors, which are detrimental in financial contexts. The final champion model is automatically selected and promoted to the Staging phase in the MLflow Model Registry.

## Explainability & Trust

Trust is essential for adoption by non-technical users. Beyond raw price predictions, we expose the drivers of valuation. The application highlights how neighborhood income levels and density impact the specific estimate for a property.

Global feature importance analysis confirms that location and size are the dominant factors, aligning the model's behavior with real estate domain knowledge. This transparency helps users distinguish between algorithmic errors and genuine market anomalies.

## User-Facing Application

The system includes a Streamlit web application designed for real estate agents and property owners. Users input property details through a structured form and receive an estimated market value with a likely price range.

The app abstracts the complexity of the underlying Spark pipeline. It uses a snapshot deployment strategy, reading from a frozen state of the data lake and model registry to ensure high availability and low latency in a cloud environment.

## Engineering Trade-offs

We chose a local deployment of Apache Spark for the inference engine to maintain full compatibility with the trained PySpark models. While this introduces a brief cold-start latency, it guarantees that the inference logic exactly matches the training logic without requiring complex model transpilation.

MongoDB was selected for the Silver layer to accommodate the semi-structured nature of web-scraped property data. This flexibility allows us to ingest new property attributes without immediate schema migrations, decoupling data collection from downstream analysis.

## How to Run

1. Initialize the environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. Run the full data pipeline (ETL + Training):
    ```bash
    # Ensure MongoDB is running locally
    python3 data_collection.py --collectors all
    python3 data_formatting.py
    python3 exploitation_zone.py
    python3 ml_training.py
    ```

3. Launch the valuation application:
    ```bash
    streamlit run app/main.py
    ```

## Future Improvements

We plan to decouple the inference service from the Streamlit frontend by exposing the model via a REST API (FastAPI). This would allow independent scaling of the user interface and the compute-heavy prediction engine. Additionally, integrating a CI/CD pipeline would automate the testing of data transformations and model integrity upon new commits.
