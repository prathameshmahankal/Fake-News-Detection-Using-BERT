import os
from random import randint
from azure.storage.blob import BlobServiceClient
from azure.servicebus import ServiceBusClient, ServiceBusMessage
import resources.azure_messaging_config as conf
import json
from datetime import datetime
import pandas as pd
import time
import resources.azure_messaging_config as conf
from predict import Predict

def read_queue():
    try:
        servicebus_client = ServiceBusClient.from_connection_string(conn_str=conf.ML_QUEUE_CONNECTION_STR, logging_enable=True)
        with servicebus_client:
            # get the Queue Receiver object for the queue
            receiver = servicebus_client.get_queue_receiver(queue_name=conf.ML_QUEUE_NAME, max_wait_time=5)
            queue_empty = True
            with receiver:
                for msg in receiver:
                    queue_empty = False
                    msg_dict = json.loads(str(msg))
                    print(msg_dict)
                    
                    p = Predict(msg_dict['container_name'],
                                msg_dict['blob_name'], 
                                msg_dict['request_id'])
                    p.update_local_list(msg_dict['request_id'], "APPEND")
                    
                    receiver.complete_message(msg)
                    p.main(reupload_flag=True)
                    p.update_local_list(msg_dict['request_id'], "REMOVE")
                    # receiver.complete_message(msg)
        return queue_empty

    except Exception as ex:
        print('Exception:')
        print(ex)


if __name__ == '__main__':
    wait_time = 1
    # iterations = 0
    while True:
        queue_empty = read_queue()
        wait_time = 5 if queue_empty else 1
        print("Waiting backoff: " + str(wait_time) + " seconds...")
        time.sleep(wait_time)
        # iterations+=1