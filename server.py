import pickle
import datetime
import tensorflow as tf
import tornado.ioloop

import tornado.gen
from tornado.concurrent import Future

from tensorflow import keras
from keras.applications.densenet import DenseNet121
from keras import layers, Model, Input
from modelUtils import Config, Logger, get_model, dry_run_right_model, input_size
import pandas as pd
import numpy as np

# Read the configurations from the config file.
config = Config.get_config()

device = config['server_device']
model_name = config['model']
split_point = None
model_config = df = pd.read_json('modelConfig.json')
batches = int(model_config["frames_to_process"][0])

BATCH_SIZE = 2
client_data_queue = []
client_response_futures = []
unique_clients = set()


client_data_dict = {} # {client_id: [data1, data2, ...], ...}
client_response_dict = {} # {client_id: [response1, response2, ...], ...}
client_response_futures = [] # [(future1, client_id1), (future2, client_id2), ...]


# For SplitPointHandler
split_point_client_data_dict = {} 
split_point_client_response_futures = []

# For TimeHandler
time_client_data_dict = {} 
time_client_response_futures = []


# Define batch size and initialize a batch list
batch_input = []

with tf.device(device):
    model = get_model(model_name)
    print('*************************************************')
    print(tf.config.list_physical_devices(device_type=None))
    print('**************************************************')

    left_model = None
    right_model = None
    total_right_model_time = 0


class ModelHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def post(self):
        received = pickle.loads(self.request.body)
        data = received
        client_id = received['client_id']
        
        future = Future()
        # Store data and its associated future
        if client_id not in client_data_dict:
            client_data_dict[client_id] = []
        client_data_dict[client_id].append(data)
        
        client_response_futures.append((future, client_id))
        if len(client_data_dict) >= BATCH_SIZE:
            # For simplicity, let's assume process_batch returns a dictionary: {client_id: [processed_data1, processed_data2, ...], ...}
            batch_response_dict = process_batch(client_data_dict)
            
            
            # Resolve all futures
            for response_future, client_id in client_response_futures:
                client_responses = batch_response_dict.get(client_id, [])
                if client_responses:
                    response_data = client_responses#.pop(0)
                    response_future.set_result(response_data)
                
            # Clear for next batch
            client_data_dict.clear()
            client_response_futures.clear()

        response_data = yield future
        self.write(pickle.dumps(response_data))
        
        


class SplitPointHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def post(self):
        received = pickle.loads(self.request.body)
        data = received
        client_id = received['client_id']

        future = Future()

        if client_id not in split_point_client_data_dict:
            split_point_client_data_dict[client_id] = {}
        split_point_client_data_dict[client_id] = data

        split_point_client_response_futures.append((future, client_id))

        if len(split_point_client_data_dict) >= BATCH_SIZE:
            batch_response_dict =  process_split_point_batch(split_point_client_data_dict)
            for response_future, client_id in split_point_client_response_futures:
                client_responses = batch_response_dict.get(client_id, [])
                if client_responses:
                    response_data = client_responses#.pop(0)
                    response_future.set_result(response_data)
            split_point_client_data_dict.clear()
            split_point_client_response_futures.clear()

        response_data = yield future
        self.write(pickle.dumps(response_data))



class TimeHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def post(self):
        received = pickle.loads(self.request.body)
        data = received
        client_id = received['client_id']

        future = Future()

        if client_id not in time_client_data_dict:
            time_client_data_dict[client_id] = {}
        time_client_data_dict[client_id] = data

        time_client_response_futures.append((future, client_id))

        if len(time_client_data_dict) >= BATCH_SIZE:
            batch_response_dict = process_time_batch(time_client_data_dict)
            for response_future, client_id in time_client_response_futures:
                client_responses = batch_response_dict.get(client_id, [])
                if client_responses:
                    response_data = client_responses #.pop(0)
                    response_future.set_result(response_data)
            time_client_data_dict.clear()
            time_client_response_futures.clear()

        response_data = yield future
        self.write(pickle.dumps(response_data))


def model_right(data_batch):
    right_model_output = []
    frame_seq_nos = []
    with tf.device(device):



        one_lst = sum(data_batch, [])
        data = pd.DataFrame.from_dict(one_lst) #,columns = ['client_id', 'data','frame_seq_no'])
        left_model_output = data['data'].tolist()
        frame_seq_nos = data['frame_seq_no'].tolist() 
        #inputs = tf.keras.Input(tf.stack(left_model_output).shape)
        ip_shape1 = tf.keras.Input(tf.stack(left_model_output).shape)
        dense_1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        dense_2 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        op1 = dense_1(ip_shape1)
        op1 = dense_2(op1)
        model_custom = tf.keras.models.Model(inputs=[ip_shape1], outputs=[op1])
        Logger.log(f'Executing right model for frame #s: {len(frame_seq_nos)}')
        right_model_start_time = datetime.datetime.now()
        stacked_tensors = tf.stack(left_model_output)
        reshaped_tensor = tf.reshape(stacked_tensors, (-1, stacked_tensors.shape[0], stacked_tensors.shape[1], stacked_tensors.shape[2], 
        stacked_tensors.shape[3], 3)) 
        right_model_output.append(model_custom(reshaped_tensor))
        #right_model_output.append(right_model(left_model_output))
        right_model_end_time = datetime.datetime.now()
        add_to_total_right_model_time((right_model_end_time - right_model_start_time).total_seconds())
        return_data_batch = [{'result': right_model_output, 'frame_seq_no': len(frame_seq_nos)}]
        print('executed    ')
        return return_data_batch
        
        
def process_batch(client_data_dict):
    """
    This function will process the batch data collected from clients.
    Implement your batch processing logic here.
    """
    batch_response_dict = {}
    with tf.device(device):
        data_val = list(client_data_dict.values())
        data_keys = list(client_data_dict.keys())
        output_data_val = model_right(data_val)
        batch_response_dict = dict(zip(data_keys, output_data_val))
        return batch_response_dict
    
    
'''def process_split_point(x):
    global right_model
    global left_model
    global split_point
    global ips
    global ops
    batch_response_dict = {}
    split_layer = model.layers[x] 
    left_model = tf.keras.Model(inputs=model.input, outputs=split_layer.output)
    next_layer = model.layers[x + 1]
    print(f'Starting from layer # {x + 1} in server. Layer name: {next_layer.name}')
    ops = model.output
    right_model = keras.Model(inputs=next_layer.input, outputs=model.output)
    #ToDo not necessary
    dry_run_right_model(left_model, right_model, input_size.get(model_name))
    return [pickle.dumps('Right model is ready.')]'''
    
def process_split_point(x):
    global right_model
    global left_model
    global split_point
    global ips
    global ops
    batch_response_dict = {}
    split_layer = model.layers[x] 
    left_model = tf.keras.Model(inputs=model.input, outputs=split_layer.output)
    next_layer = model.layers[x + 1]
    print(f'Starting from layer # {x + 1} in server. Layer name: {next_layer.name}')
    ops = model.output
    right_model = keras.Model(inputs=next_layer.input, outputs=model.output)
    #ToDo not necessary
    dry_run_right_model(left_model, right_model, input_size.get(model_name))
    return [pickle.dumps('Right model is ready.')]
    
def process_split_point_batch(split_point_client_data_dict):
    """
    Your batch processing logic for split point data here...
    For the example, just returning the data as is.
    """
    
    batch_response_dict = {}
    with tf.device(device):
        # Your model's batch processing logic here...
        data = pd.DataFrame(list(split_point_client_data_dict.values()))
        keys = data['client_id']
        split_point = data['split_point']
        output_data_val = map(process_split_point, split_point)
        # Sample logic: return the received data
        batch_response_dict = dict(zip(keys, output_data_val))
        return batch_response_dict
    

def get_total_right_model_time(data):
    global total_right_model_time
    client_split_point = data['split_point']
    return_data = None
    if client_split_point == split_point:
        return_data = {'total_right_model_time': total_right_model_time}
        # Reset time for the next split point.
        total_right_model_time = 0
    else:
        return_data = {'total_right_model_time': f'split point mismatch. Expected {split_point} but received {client_split_point}.'}
    return pickle.dumps(return_data)


def add_to_total_right_model_time(current_frame_exec_time):
    global total_right_model_time
    total_right_model_time += current_frame_exec_time


def make_app():
    return tornado.web.Application([
        (r"/model", ModelHandler),
        (r"/split_point", SplitPointHandler),
        (r"/right_model_time", TimeHandler)
    ])


with tf.device(device):
    if __name__ == "__main__":
        app = make_app()
        app.listen(8881)
        print("Server started")
        tornado.ioloop.IOLoop.current().start()

