
# ============
# IMPORT KafkaConnector
# ===========
import threading
import json
import requests
import numpy as np
import cv2
import requests
import jsonpickle
import traceback
import warnings

from kafka_connector import KafkaConnector
from helper import config as kafka_cfg
from utils import *

# ==================================
warnings.filterwarnings("ignore")
thread=None
group_id = kafka_cfg.group_id
kf_host = kafka_cfg.kf_host
kf_port = kafka_cfg.kf_port
kafka_server = kf_host + ":" + kf_port
bootstrap_servers = [kafka_server]
topic_consumer = kafka_cfg.topic_consumer
topic_producer = kafka_cfg.topic_producer
enable_auto_commit = False
timeout = 3000
auto_offset_reset = "earliest"
cvm_version = kafka_cfg.cvm_version
api_server_url = f"{kafka_cfg.broker_host}:{kafka_cfg.broker_port}"
# ====================================

# ===============================================================================

def volume_estimation(img):

    response = predict_food(img)
    response_pickled = jsonpickle.encode(response)

    return response_pickled


def volume_estimation_total(img):

    response = predict_ingredients(img)
    response_vol = predict_food_total(img)

    response["volumes"] = response_vol
    response_pickled = jsonpickle.encode(response)

    return response_pickled


def send_to_cvm(document):
    """ My Processing function"""

    url = f'{api_server_url}/user/login'
    headers = {'content-type': 'application/json'}
    data=json.dumps({"user": "admin", "password": "gatvgatv"})
    response = requests.post(url, data=data, headers=headers)
    resp_data = json.loads(response.text)

    url =  f'{api_server_url}/key/all'
    headers = {'content-type': 'application/json', "Authorization": "Bearer "+resp_data["token"]}
    response = requests.get(url, headers=headers)
    resp_data = json.loads(response.text)
    print(resp_data)

    #document = [{"uuid":"e34d077b45c0ec35ec74d4af8e86c47850a2a52d256cdcbe5c32","image":"http://138.4.47.33:8163/food/images/5e987fc4c20d305f03d9a53e"}]
 
    url = document[0]["image"]#'http://138.4.47.33:8163/food/images/'+'5e98d7be05337c6d3c9c68c8?'
    #url = document["image"]
    headers = {'content-type': 'application/json', 'x-api-key': resp_data["keys"][0]["key"]}
    response = requests.get(url, headers=headers)
    #print(response.text)
    #resp_data = json.loads(response.text)

    filestr = response.content
    #convert string data to numpy array
    npimg = np.frombuffer(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    #cv2.imshow("Food", img)
    #cv2.waitKey(0)

    if cvm_version == "1":
        response = volume_estimation(img)
    elif cvm_version == "2":
        response = volume_estimation_total(img)

    volume_estimation_response_dict = json.loads(response)
    volume_estimation_response_dict.update(document[0])
    return volume_estimation_response_dict

def init_kafka_manager():
    try:
        kafka_manager = KafkaConnector(topic_consumer=topic_consumer,
                                            topic_producer=topic_producer,
                                            group_id=group_id,
                                            bootstrap_servers=[kafka_server],
                                            enable_auto_commit=enable_auto_commit,
                                            consumer_timeout_ms=timeout,
                                            auto_offset_reset=auto_offset_reset)
        # Init Consumer
        kafka_manager.init_kafka_consumer()
        # Init Producer
        kafka_manager.init_kafka_producer()
    except Exception as e:
        print(e)
        print(traceback.print_exc())
    return kafka_manager


def start_analysis(kafka_manager):
        done = True
        print("Start Analysis Poll!")
        while done:
            try:
            	# Start Consuming
                kafka_manager.consumer.poll()
                for msg in kafka_manager.consumer:
                    try:
                        print('Loading Kafka Message')
                        document = json.loads(msg.value)
                        published_document = send_to_cvm(document)
                        if published_document is not None:
                            print('Putting document into Kafka:', published_document)
                            kafka_manager.put_data_into_topic(data=published_document)
                            # Commit the document to avoid processing again the same document
                            kafka_manager.consumer.commit()
                            print('Done!')
                        else:
                            print("Document not ingested into Kafka")
                    except Exception as e:
                        print(traceback.print_exc())
                        kafka_manager.consumer.commit()
                        continue
            except Exception as e:
                print(traceback.print_exc())
                continue


# ===============================================================================

if thread is None:
    kafka_manager = init_kafka_manager()

    if kafka_manager.consumer is not None:
        thread = threading.Thread(target=start_analysis, args=(kafka_manager,))
        thread.start()
