from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
import pandas as pd
import numpy as np
import pyodbc
import requests
import urllib.request
import json
import os
import ssl

def from_dataframe_to_json(df):
    df_json = json.loads(df.to_json(orient='records'))
    return df_json

def is_numeric_string(s):
    if isinstance(s, str):
        return all(c.isdigit() or c == "," for c in s)
    else:
        return False

app = Flask(__name__)

### swagger specific ###
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Seans-Python-Flask-REST-Boilerplate"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
### end swagger specific ###

@app.route('/get-prediction', methods=['POST'])
def get_prediction():
    # Get the input data
    input_data = request.json

    # Connect to the Azure SQL database and retrieve data
    connection_string = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:becyb-localhost.database.windows.net,1433;Database=becyb-db;Uid=becyb-admin;Pwd=Frw~1na,SEzgzj2;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
    connection = pyodbc.connect(connection_string)
    sql_BECYB_learning_data = 'SELECT * FROM [dbo].[Post]'
    sql_BECYB_post_tag = 'SELECT * FROM [dbo].[Post_Tag]'
    sql_BECYB_post_topic_training = 'SELECT * FROM [dbo].[Post_Topic_Training]'
    sql_BECYB_post_topic_user = 'SELECT * FROM [dbo].[Post_Topic_User]'


    # First SQL transformation
    df_BECYB_learning_data = pd.read_sql(sql_BECYB_learning_data, connection)
    df_BECYB_post_tag = pd.read_sql(sql_BECYB_post_tag, connection)
    merged_data = df_BECYB_learning_data.merge(df_BECYB_post_tag, left_on='post_id', right_on='FK_post_id')

    # Group by post_id, liczba_like, liczba_wyswietlen columns and apply group_concat function to FK_tag_id column
    sql_transform_data_1 = merged_data.groupby(['post_id', 'liczba_like', 'liczba_wyswietlen']).agg({'FK_tag_id': lambda x: ','.join(map(str, x))}).reset_index()

    # Rename the columns
    sql_transform_data_1 = sql_transform_data_1.rename(columns={'FK_tag_id': 'tags_array'})


    # Second SQL transformation
    df_BECYB_post_topic_training = pd.read_sql(sql_BECYB_post_topic_training, connection)
    df_BECYB_post_topic_training = df_BECYB_post_topic_training.rename(columns={'FK_post_id': 'post_id'})
    df_BECYB_post_topic_user = pd.read_sql(sql_BECYB_post_topic_user, connection)
    df_BECYB_post_topic_user = df_BECYB_post_topic_user.rename(columns={'FK_post_id': 'post_id'})

    df_transformed = pd.merge(df_BECYB_post_topic_training, sql_transform_data_1, on='post_id', how='left')
    df_transformed = pd.merge(df_BECYB_post_topic_user, df_transformed, on='post_id', how='left')

    # Group by and apply transformations
    df_transformed = df_transformed.groupby(['post_id', 'liczba_like', 'liczba_wyswietlen', 'tags_array']).agg(
        topics_training=('FK_topic_id_x', lambda x: ','.join(map(str, x))),
        topics_from_views=('FK_topic_id_y', lambda x: ','.join(map(str, x)))
    ).reset_index()

    # Create new columns for each value in sorted array
    topics_tag = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

    df_transformed["is_topics_numeric"] = df_transformed["topics_from_views"].apply(is_numeric_string)

    for val in topics_tag:
        df_transformed['topic '+str(val)] = df_transformed['topics_training'].apply(lambda x: int(str(val) in x.split(','))*5)
        df_transformed['topic '+str(val)] = df_transformed.apply(
            lambda row: int(str(val) in row['topics_from_views'].split(','))*2 + int(row['topic '+str(val)]) if bool(row["is_topics_numeric"]) else row['topic '+str(val)],
            axis=1
        )

    topic_ids = input_data["topic_ids"]

    # add 5 to the columns where topic id is equal to x_n and its value is bigger than 2
    for i, topic_id in enumerate(topic_ids):
        col_name = 'topic {}'.format(i+1)
        if df_transformed[col_name].iloc[topic_id-1] > 2:
            df_transformed[col_name] = df_transformed[col_name] + 5

    # Drop the columns
    df_transformed = df_transformed.drop('tags_array', axis=1)
    df_transformed = df_transformed.drop('topics_training', axis=1)
    df_transformed = df_transformed.drop('topics_from_views', axis=1)

    # # Make the request to Azure ML endpoint
    url = 'http://65516273-ffd0-4c78-bda2-1d1d0e2a69d6.francecentral.azurecontainer.io/score'
    api_key = 'WW5weSoF5aKsBn9J5hTMFJi8ZdwPZL0k'
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    json_str = df_transformed.to_json(orient='records')

    # parse JSON string into list of dictionaries (array of JSON objects)
    json_array = json.loads(json_str)

    prediction = []
    for transformed_data in json_array:
        if len(prediction) < 10:
            try:
                data =  {
                    "Inputs": {
                        "WebServiceInput0": [
                            transformed_data
                        ]
                    },
                    "GlobalParameters": {}
                    }
                body = str.encode(json.dumps(data))
                req = urllib.request.Request(url, body, headers)
                response = urllib.request.urlopen(req)
                result_bytes = response.read()
                result_str = result_bytes.decode('utf-8')
                result_dict = json.loads(result_str)
                post_id = result_dict['Results']['WebServiceOutput0'][0]['post_id']
                label = result_dict['Results']['WebServiceOutput0'][0]['Scored Labels']
                if (label == 1):
                    prediction.append(post_id)
            except urllib.error.HTTPError as error:
                print("The request failed with status code: " + str(error.code))

                # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
                print(error.info())
                print(error.read().decode("utf8", 'ignore'))
        else:
            break

    print(prediction)
    posts_df = df_BECYB_learning_data[df_BECYB_learning_data['post_id'].isin(prediction)]
    filtered_df_json = posts_df.to_dict(orient='records')

    # Return the predicted data
    return jsonify({'data': filtered_df_json})

if __name__ == '__main__':
    app.run()
