import os, uuid
import pandas as pd
import requests
import json
import time
import re
import string
from io import StringIO
from azureml.core import Webservice
from azureml.core import Workspace
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication
from azure.servicebus import ServiceBusClient, ServiceBusMessage
import resources.azure_messaging_config as conf
import ast

class Predict:

	def __init__(self, CONTAINER_NAME, BLOB_NAME, REQUEST_ID):

		self.CONTAINER_NAME = CONTAINER_NAME
		self.BLOB_NAME = BLOB_NAME
		self.REQUEST_ID = REQUEST_ID
		# "container042021"

		self.blob_service_client = BlobServiceClient.from_connection_string(conn_str=conf.AZURE_STORAGE_CONNECTION_STRING)
		self.container_client = self.blob_service_client.get_container_client(self.CONTAINER_NAME)

		interactive_auth = InteractiveLoginAuthentication(tenant_id=conf.TENANT_ID)
		self.ws = Workspace.from_config('resources', auth=interactive_auth)

		self.service = Webservice(workspace=self.ws, name='mynewservice')
		# self.service = Webservice(workspace=self.ws, name='my-new-aci-service')
		# print(self.service.scoring_uri)

	def get_preds(self, service, data):
	    
		scoring_uri = self.service.scoring_uri
		# If the service is authenticated, set the key or token
		# primary_key, _ = service.get_keys()
		
		# api_key = self.service.get_keys()[0]
		# headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
		
		headers = {'Content-Type': 'application/json'}
		data = {'text': data.cleaned_tweet.to_list()}
		data = json.dumps(data)
		# print(data)
		resp = requests.post(scoring_uri, data=data, headers=headers)
		# print(resp.text)

		return resp.text

	def clean(self, df_curr):
	    
	    df_curr['cleaned_tweet'] = df_curr['tweet'].apply(str)
	    df_curr['cleaned_tweet'] = df_curr['cleaned_tweet'].apply(lambda x:re.sub(r'http\S+', '', x))
	    df_curr['cleaned_tweet'] = df_curr['cleaned_tweet'].apply(lambda x:re.sub(r'@\S+ ', '', x))
	    df_curr['cleaned_tweet'] = df_curr['cleaned_tweet'].apply(lambda x:''.join(i for i in x if not i.isdigit()))
	    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
	    df_curr['cleaned_tweet'] = df_curr['cleaned_tweet'].str.translate(table)
	    df_curr['cleaned_tweet'] = df_curr['cleaned_tweet'].str.replace(' +', ' ', regex=True)
	    df_curr['cleaned_tweet'] = df_curr['cleaned_tweet'].str.lower()
	    df_curr['cleaned_tweet'] = df_curr['cleaned_tweet'].str.strip()
	    
	    return df_curr

	def reupload(self, df, blob):
		    
		local_path = os.path.join("tmp", self.BLOB_NAME, self.REQUEST_ID)
		if not os.path.exists(local_path):
			os.makedirs(local_path, 0o777)
		df.to_csv(os.path.join(local_path, 'tmp.csv'), index=False, encoding="utf-8")

		# Blob upload logic should stay here not another func:
		connect_str = conf.AZURE_STORAGE_CONNECTION_STRING
		blob_service_client = BlobServiceClient.from_connection_string(connect_str)
		blob_client = blob_service_client.get_blob_client(container=self.CONTAINER_NAME, blob=blob)

		with open(os.path.join(local_path, 'tmp.csv'), "rb") as data:
		    blob_client.upload_blob(data, overwrite=True)

		os.remove(os.path.join(local_path, 'tmp.csv'))

	def notify_viz_pipeline(self, notif_message):
	    
	    servicebus_client = ServiceBusClient.from_connection_string(conn_str=conf.ML_QUEUE_CONNECTION_STR,
	                                                                logging_enable=True)
	    with servicebus_client:
	        # get a Queue Sender object to send messages to the queue
	        sender = servicebus_client.get_queue_sender(queue_name=conf.VIZ_QUEUE_NAME)
	        with sender:
	        # send one message
	            serialized_msg = ServiceBusMessage(json.dumps(notif_message))
	            sender.send_messages(serialized_msg)
	            print("Notified visualization pipeline for request_id : {}".format((notif_message["request_id"])))

	def main(self, reupload_flag=False):

		blob_list = self.container_client.list_blobs()
		prediction_list = []
		predictions = []

		start = time.time()

		for blob in blob_list:
			if self.REQUEST_ID+'/fileblock' in blob.name:
				print("Working on blob:", blob.name)
				blob_client = self.blob_service_client.get_blob_client(container=self.CONTAINER_NAME, blob=blob.name)
				data = blob_client.download_blob().readall()
				s=str(data,'utf-8')
				data = StringIO(s) 
				df_curr = pd.read_csv(data, lineterminator='\n')
				df_curr = self.clean(df_curr)
				predictions = self.get_preds(self.service, df_curr)
				predictions = ast.literal_eval(predictions)
				print("Predictions generated for blob:", blob.name)

				# df_curr['predictions'] = predictions

				if reupload_flag:
					self.reupload(df_curr, blob.name)

			# prediction_list.extend(predictions)

		print("Time taken for predictions:", time.time() - start)
		
		notif_message = {"container_name" : self.CONTAINER_NAME, "blob_name" : self.BLOB_NAME, "request_id" : self.REQUEST_ID}
		self.notify_viz_pipeline(notif_message)

		return prediction_list

if __name__ == "__main__":

	CONTAINER_NAME, BLOB_NAME, REQUEST_ID = "container052021", 'Blob_24_05_2021', 'request_7533'

	p = Predict(CONTAINER_NAME, BLOB_NAME, REQUEST_ID)
	# 'Blob_21_04_2021/request_5655/fileblock_5.csv'
	predictions = p.main(reupload_flag=True)
	# print(predictions)

# docker build -t dockerpython .
# docker run dockerpython

# SPARK_HOME - C:\Users\pmahankal.HIREZCORP\Desktop\Spark\spark-3.0.0-bin-hadoop2.7
# PYTHONPATH - %SPARK_HOME%\python;%SPARK_HOME%\python\lib\py4j-0.10.9-src.zip:%PYTHONPATH%