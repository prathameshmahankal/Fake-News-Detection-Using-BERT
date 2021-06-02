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

		"""
		First, we will initialize different variables that we will use in the code below
		"""

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

	def get_preds(self, data):
	    
		"""
		This function receives the dataframe for which we are making predictions
		"""

		api_key = ""
		scoring_uri = self.service.scoring_uri
		
		# If the service is authenticated, set the key or token by uncommenting below code
		# api_key = self.service.get_keys()[0]

		if api_key:
			headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
		else:
			headers = {'Content-Type': 'application/json'}
		
		data = {'text': data.cleaned_tweet.to_list()}
		data = json.dumps(data)
		resp = requests.post(scoring_uri, data=data, headers=headers)
		
		return resp.text

	def clean(self, df_curr):

		"""
		This function cleans the tweet column in our dataframe
		"""

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
		    
		"""
		This function reuploads the dataframe with predictions back tot he blob storage
		"""

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

		"""
		This function informs the viz pipeline that the dataset with predictions is successfully reuploaded
		can be used for making the necessary visualizations
		"""

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

	def update_local_list(self, request_id, action):
		with open("messages.json", "r") as jsonFile:
		    messages = json.load(jsonFile)

		if action=="APPEND":
			messages["request_ids"].append(request_id)
			print("Request {} appended".format(request_id))
		elif action=="REMOVE":
			try:
			    messages["request_ids"].remove(request_id)
			    print("Request {} removed".format(request_id))
			except:
				print("Request not present. Might be already removed")

		with open("messages.json", "w") as jsonFile:
		    json.dump(messages, jsonFile)

	def main(self, reupload_flag=False):

		"""
		This function fetches the files to make predictions for and passes it to the get_preds function
		"""

		blob_list = self.container_client.list_blobs() # making a list of all the blobs
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
				df_curr = df_curr[[ 'id','conversation_id','created_at','tweet','source','language',
									'in_reply_to_user_id','hashtags','urls','media','retweet_count',
									'reply_count','like_count','quote_count','user_id','user_screen_name',
									'user_name','user_description','user_location','user_created_at',
									'user_followers_count','user_friends_count','user_statuses_count',
									'user_verified','references']]
				df_curr = self.clean(df_curr)
				
				# generate predictions
				predictions = self.get_preds(df_curr)
				# print(type(predictions))
				

				# Retry mechanism to generate predictions for each fileblock belonging to one request id
				trial = 10
				while trial>0:
					try:
						trial-=1
						predictions = ast.literal_eval(predictions)
					except:
						print("Prediction generation failed. Retrying..")
						continue
					break
				
				print("Predictions generated for blob:", blob.name)
				# print("Length of predictions:", len(predictions))
				if trial>0:
					df_curr['predictions'] = predictions
				else:
					df_curr['predictions'] = [1]*len(predictions)

				if reupload_flag:
					self.reupload(df_curr, blob.name)

			# collect predictions for each file
			prediction_list.extend(predictions)

		print("Time taken for predictions:", time.time() - start)

		notif_message = {"container_name" : self.CONTAINER_NAME, "blob_name" : self.BLOB_NAME, "request_id" : self.REQUEST_ID}
		self.notify_viz_pipeline(notif_message)

		return prediction_list

if __name__ == "__main__":

	CONTAINER_NAME, BLOB_NAME, REQUEST_ID = "container062021", 'Blob_01_06_2021', 'request_3889'

	p = Predict(CONTAINER_NAME, BLOB_NAME, REQUEST_ID)
	predictions = p.main(reupload_flag=True)

# docker build -t dockerpython .
# docker run dockerpython

# docker stop c3ff78ff4eeb

# SPARK_HOME - C:\Users\pmahankal.HIREZCORP\Desktop\Spark\spark-3.0.0-bin-hadoop2.7
# PYTHONPATH - %SPARK_HOME%\python;%SPARK_HOME%\python\lib\py4j-0.10.9-src.zip:%PYTHONPATH%