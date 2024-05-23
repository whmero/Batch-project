import tornado
import tornado.httpclient as httpclient
import pickle
from tornado import ioloop
import os
import sys
import cv2
import base64
import numpy as np
import csv
import datetime
from tensorflow.keras.applications import DenseNet121, ResNet50, ResNet101
from tensorflow.keras.models import Model
from vit_keras import vit   
import argparse
import logging

# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


imgs_path = os.listdir("Images")
parser = argparse.ArgumentParser(description="Im2Latex Training Program")
'''client = input("Enter id of the client: ")
parser.add_argument("--clientid", type=int, default = client)
args = parser.parse_args()  
client_c = "client" + str(args.clientid)'''
client_c = sys.argv[1]
sending_start_time = None
receiving_end_time = None
metrics_headers = ['client_id', 'model_name', 'batch', 'split_index', 'time_batch','time between send and receive']
model_name = 'VIT'
split_index = 3
input_shape = (32, 32, 3)

if split_index != 0:
    models = dict()
    models['DensetNet121'] = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    models['Resnet50'] = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    models['Resnet101'] = ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)
    models['VIT'] = vit.vit_b32(image_size=32,activation='softmax',pretrained=True,include_top=True,pretrained_top=False,classes=2)

'''
if model_name == 'VIT':
        split_indices = [3, 17]
    elif model_name == 'Resnet50'or model_name == 'Resnet101':
        split_indices = [5, 91]
    else:
        split_indices = [5, 94]
    split_index = split_indices[split_idx - 1]
'''
def get_model(model_name, split_index):
    if split_index != 0:
        layer = models[model_name].layers[split_index]
        #save splitted part of model belonging to the server in models dictionary
        return Model(inputs=models[model_name].input, outputs=layer.output)
'''
    # Split the model into two parts
    model = models[model_name]
    selected_outputs =model.layers[split_index].output 
    # Now, create the model with the selected intermediate outputs
    model_client = Model(inputs=model.input, outputs=selected_outputs)
    split_layer = model.layers[split_index]
    print(split_layer.name)  '''

def write_to_csv(filename, field_names, data):
    # Check if the file exists
    file_exists = False
    try:
        with open(filename, 'r') as file:   
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file in the appropriate mode
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)

        # Write a new line if the file is empty
        if not file_exists:
            writer.writerow(field_names)  # Example column headers

        # Write the data to the file
        writer.writerow(data)

def handle_response(result):
    data = pickle.loads(result.body)
    json_data = data
    model_name = json_data['model_name']
    batch_size = json_data['batch']
    req_id = json_data['req_id']
    time_between_send_and_receive = (receiving_end_time - sending_start_time).total_seconds()
    logging.info("Received result of processing request no. "+ str(req_id) + " in batching time = "+ str(json_data['time_batch'])+ "seconds from server with model: " + str(model_name)+ " and total processing time =  " + str(time_between_send_and_receive))
    write_to_csv('client.csv', metrics_headers, [json_data['client_id'], model_name,json_data['batch'], str(split_index), json_data['time_batch'], str(time_between_send_and_receive)])



async def main():
    global sending_start_time
    global receiving_end_time
    http_client = httpclient.AsyncHTTPClient()
    for req_id in range(int(len(imgs_path))):
        try:
            img = cv2.imread("Images/" + imgs_path[req_id])
            if split_index != 0:
                '''
                if model_name != "VIT":
                    resized_image = cv2.resize(img,(224,224))
                    img = resized_image
                else:'''
                img = cv2.resize(img,(32,32))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img, axis=0)  # Ensure that input_data has batch dimension
                model = get_model(model_name, split_index)
                img = model.predict(img)
            post_data = {'client_id': client_c, 'request_id': req_id + 1, 'image': img, 'model_name': model_name, 'split_index':split_index}
            serialized_outputs = pickle.dumps(post_data)
            body = base64.b64encode(serialized_outputs)
            sending_start_time = datetime.datetime.now()
            response = await http_client.fetch("http://localhost:8080", method  ='POST', headers = None, body = body)
            receiving_end_time = datetime.datetime.now()
            #response.add_done_callback(lambda f: handle_response(f.result))
            handle_response(response)
            await tornado.gen.sleep(1)
        except httpclient.HTTPError as e:
            # HTTPError is raised for non-200 responses; the response
            # can be found in e.response.
            print("Waiting for server to start execution...........")
        except Exception as e:
            # Other errors are possible, such as IOError.
            print("Waiting of results...............")
    http_client.close()

if __name__ == '__main__':
    io_loop = ioloop.IOLoop.current()
    io_loop.add_callback(main)
    io_loop.start()