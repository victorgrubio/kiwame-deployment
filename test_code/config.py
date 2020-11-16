import os

#  Ports and hosts
port = "5000"
host = "0.0.0.0"
kf_host = "192.168.0.159"
kf_port = "9092"
broker_host = "http://192.168.0.159"
broker_port = "8090"
topic_consumer = "newFoodPicture"
topic_producer = "ingredients"
cvm_version = "2"
kafka_server = kf_host + ":" + kf_port
group_id = "group1"

# KAFKA Variables
if os.getenv("KAFKA_SERVER") is not None:
    kf_host = os.getenv("KAFKA_SERVER")

if os.getenv("KAFKA_PORT") is not None:
    kf_port = os.getenv("KAFKA_PORT")

if os.getenv("KAFKA_TOPIC_CONSUMER") is not None:
    topic_consumer = os.getenv("KAFKA_TOPIC_CONSUMER")

if os.getenv("KAFKA_TOPIC_PRODUCER") is not None:
    topic_producer = os.getenv("KAFKA_TOPIC_PRODUCER")

if os.getenv("BROKER_HOST") is not None:
    broker_host =  os.getenv("BROKER_HOST")

if os.getenv("BROKER_PORT") is not None:
    broker_port =  os.getenv("BROKER_PORT")

if os.getenv("CVM_VERSION") is not None:
    cvm_version = os.getenv("CVM_VERSION")
