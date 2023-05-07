import streamlit as st
from datetime import datetime, timedelta
import time
import boto3
from boto3.dynamodb.conditions import Key
import os
from decimal import Decimal
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import numpy as np
from plotly import graph_objects as go
from sklearn import metrics as mt
from dotenv import load_dotenv
import awswrangler as wr
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.model import Model


expected_error = lambda y_true, y_predicted: np.mean(y_predicted-y_true)
boto3_session = boto3.Session(region_name="eu-west-1")
bucket = 'pp-strava-data'
TARGET = "suffer_score"
PREDICTORS = [
    "moving_time",
    "average_heartrate",
]  # Check with ML model implementation or create a dependency
PREDICTED = "predicted_suffer_score"

DB = 'strava'
TABLE = 'predicted_suffer_score'

sm_client = boto3.client("sagemaker")
predictor = Predictor("strava", serializer=CSVSerializer())
model_name = predictor._get_model_names()[0]
train_date = sm_client.describe_model(ModelName=model_name)["CreationTime"].strftime("%Y-%m-%d")
train_date = "2023-04-01"


def run():
    st.title('Strava suffer score prediction')
    st.markdown("""
    Part of Strava suffer score pipeline project [repo](https://github.com/prteek/strava-project). 
""")

    st.sidebar.caption("Strava pipeline")
    st.sidebar.image("https://raw.githubusercontent.com/prteek/strava-project/main/resources/strava.png")
    st.caption("The dashboard monitors performance of model that the pipeline re-trains week (on AWS Sagemaker) and uses to make predictions for each workout")

    # Get model and training data
    df_train = wr.s3.read_csv(os.path.join("s3://", bucket, "prepare-training-data", "output", "train.csv"),
                              boto3_session=boto3_session)

    X = df_train[PREDICTORS].values
    y = df_train[TARGET].values
    y_pred = np.array(eval(predictor.predict(X).decode()), dtype=float)

    # Recent predictions
    data = wr.athena.read_sql_query(f"""
                    SELECT * FROM strava.predicted_suffer_score
                    JOIN strava.activities
                    ON strava.predicted_suffer_score.activity_id = strava.activities.id
                    WHERE start_timestamp >= date('{train_date}')
                    """,
            database=DB,
            boto3_session=boto3_session)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Predicted vs actual suffer score",
                                        "Distribution of errors (unseen data)"))

    fig.add_scatter(x=y, y=y_pred, mode='markers', name='Training data',
                    marker=dict(size=10, opacity=0.5))

    fig.add_scatter(x=data[TARGET], y=data[PREDICTED],
                    mode='markers', name='Unseen data',
                    marker=dict(color='yellow', size=10, opacity=0.8))

    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines',
                             name='Perfect prediction', line=dict(color='green', width=2, dash='dash')))
    fig.update_layout({'xaxis': {'title': 'Actual suffer score', 'range': [0, 100]},
                          'yaxis': {'title': 'Predicted suffer score', 'range': [0, 100]}})

    fig.add_trace(go.Histogram(x=data[PREDICTED] - data[TARGET], name='Error',
                               xbins=dict(start=-20, end=20, size=5),
                               marker=dict(color='azure', opacity=0.5)),
                  row=1, col=2)
    fig.update_layout({'xaxis2': {'title': 'Error', 'range': [-20, 20]},})

    st.plotly_chart(fig)

    col1, col2, col3 = st.columns(3)

    mae_train = mt.mean_absolute_error(y, y_pred)
    mae_test = mt.mean_absolute_error(data[TARGET], data[PREDICTED])
    col1.metric("MAE of prediction", f"{round(mae_test,1)}", delta=f"{round(mae_test - mae_train,1)}: Delta from train",
                delta_color="inverse")

    rmse_train = np.sqrt(mt.mean_squared_error(y, y_pred))
    rmse_test = np.sqrt(mt.mean_squared_error(data[TARGET], data[PREDICTED]))
    col2.metric("RMS Error of prediction", f"{round(rmse_test,1)}",
                delta=f"{round(rmse_test - rmse_train,1)}: Delta from train",
                delta_color="inverse")

    r2_train = mt.r2_score(y, y_pred)
    r2_test = mt.r2_score(data[TARGET], data[PREDICTED])
    col3.metric("R2 score of prediction", f"{round(r2_test,2)}", delta=f"{round(r2_test - r2_train,2)}: Delta from train")

    st.markdown("---")

    st.subheader("Suffer score relationship")
    fig = go.Figure()
    fig.add_scatter(x=np.round(X[:,0]/60,1), y=y, mode='markers', name='Training data')
    fig.update_layout({'xaxis': {'title': 'Moving time (minutes)'},
                       'yaxis': {'title': 'Suffer score'}},
                       title="Suffer score vs moving time")

    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_scatter(x=X[:,1], y=y, mode='markers', name='Training data')
    fig.update_layout({'xaxis': {'title': 'Average heart rate'},
                          'yaxis': {'title': 'Suffer score'}},
                            title="Suffer score vs average heart rate")

    st.plotly_chart(fig)







