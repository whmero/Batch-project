import pickle
import datetime
import tensorflow as tf
import tornado.ioloop

import tornado.gen
from tornado.concurrent import Future

from tensorflow import keras
from modelUtils import Config, Logger, get_model, dry_run_right_model, input_size

# Read the configurations from the config file.
config = Config.get_config()

device = config['server_device']
model_name = config['model']
split_point = None

BATCH_SIZE = 1
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
                    response_data = client_responses.pop(0)
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
            batch_response_dict = process_split_point_batch(split_point_client_data_dict)
            for response_future, client_id in split_point_client_response_futures:
                client_responses = batch_response_dict.get(client_id, [])
                if client_responses:
                    response_data = client_responses.pop(0)
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
                    response_data = client_responses.pop(0)
                    response_future.set_result(response_data)
            time_client_data_dict.clear()
            time_client_response_futures.clear()

        response_data = yield future
        self.write(pickle.dumps(response_data))


def set_model_right(data):
    global right_model
    global left_model
    global split_point
    split_point = data['split_point']
    split_layer = model.layers[split_point]
    left_model = tf.keras.Model(inputs=model.input, outputs=split_layer.output)
    next_layer = model.layers[split_point + 1]
    print(f'Starting from layer # {split_point + 1} in server. Layer name: {next_layer.name}')
    right_model = keras.Model(inputs=next_layer.input, outputs=model.output)
    dry_run_right_model(left_model, right_model, input_size.get(model_name))
    return pickle.dumps('Right model is ready.')


def model_right(data_batch):
    right_model_output = []
    frame_seq_nos = []
    with tf.device(device):
        #print(data_batch)
        #left_model_output = [data['data'] for data in data_batch]
	#frame_seq_nos = [data['frame_seq_no'] for data in data_batch]
        for data in data_batch:
            left_model_output = data['data']
            frame_seq_nos.append(data['frame_seq_no'])

            Logger.log(f'Executing right model for frame #s: {frame_seq_nos}')
            right_model_start_time = datetime.datetime.now()
            right_model_output.append(right_model(left_model_output))
            right_model_end_time = datetime.datetime.now()
            add_to_total_right_model_time((right_model_end_time - right_model_start_time).total_seconds())

        return_data_batch = [{'result': right_model_output[i], 'frame_seq_no': frame_seq_nos[i]} for i in range(len(data_batch))]

        return return_data_batch
        
        
def process_batch(client_data_dict):
    """
    This function will process the batch data collected from clients.
    Implement your batch processing logic here.
    """
    batch_response_dict = {}
    with tf.device(device):
        # Your model's batch processing logic here...
        for c_id, data in client_data_dict.items():
            #return_data = []
            #for d in data:
            return_data = model_right(data)
            if c_id not in batch_response_dict:
                batch_response_dict[c_id] = []
            batch_response_dict[c_id].append(return_data)

        # Sample logic: return the received data
        return batch_response_dict
        

def process_split_point_batch(split_point_client_data_dict):
    """
    Your batch processing logic for split point data here...
    For the example, just returning the data as is.
    """
    batch_response_dict = {}
    global right_model
    global left_model
    global split_point

    with tf.device(device):
        # Your model's batch processing logic here...
        for c_id, data in split_point_client_data_dict.items():
            #return_data = []
            #for d in data:

            split_point = data['split_point']
            split_layer = model.layers[split_point]
            left_model = tf.keras.Model(inputs=model.input, outputs=split_layer.output)
            next_layer = model.layers[split_point + 1]
            print(f'Starting from layer # {split_point + 1} in server. Layer name: {next_layer.name}')
            right_model = keras.Model(inputs=next_layer.input, outputs=model.output)
            dry_run_right_model(left_model, right_model, input_size.get(model_name))
            if c_id not in batch_response_dict:
                batch_response_dict[c_id] = []
            batch_response_dict[c_id].append(pickle.dumps('Right model is ready.'))

        # Sample logic: return the received data
        return batch_response_dict
    

def process_time_batch(time_client_data_dict):
    """
    Your batch processing logic for time data here...
    For the example, just returning the data as is.
    """
    
    batch_response_dict = {}
    global total_right_model_time
    #print(time_client_data_dict)
    with tf.device(device):
        # Your model's batch processing logic here...
        for c_id, data in time_client_data_dict.items():
            #print(data, split_point, data['split_point']==split_point)
            client_split_point = data['split_point']
            return_data = None
            if client_split_point == split_point:
            	return_data = {'total_right_model_time': total_right_model_time}
            	total_right_model_time = 0        
            else:
            	return_data = {'total_right_model_time': f'split point mismatch. Expected {split_point} but received {client_split_point}.'}
	    
            if c_id not in batch_response_dict:
                batch_response_dict[c_id] = []
            batch_response_dict[c_id].append(pickle.dumps(return_data))
            
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

