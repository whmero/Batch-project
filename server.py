import socket
import threading
import pickle
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121, ResNet50, ResNet101
from tensorflow.keras import models
from keras import Input
from modelUtils import Config, Logger, get_model, dry_run_right_model, input_size,my_split_points
import pandas as pd
import numpy as np
import math
import json
import base64
import time
import concurrent.futures

IP = "localhost"
PORT = 5566
ADDR = (IP, PORT)
FORMAT = "utf-8"
# Read the configurations from the config file.
config = Config.get_config()
device = config['server_device']
model_name = config['model']
split_point = None
model_config = df = pd.read_json('modelConfig.json')
lst_load_batches = []



BATCH_SIZE = 3

client_data_dict = {} 
split_point_client_data_dict = {} 


# Define batch size and initialize a batch list
batch_input = []
with tf.device(device):
    model = get_model(model_name)
    split_points = my_split_points.get(model_name)
    frames_to_process = config['frames_to_process']
    input_shape = (frames_to_process*224,224,3)
    SIZE = 45000
    LIMIT = 802917
    LIMITS_no = int(math.floor(LIMIT /SIZE) + 1)
    print('*************************************************')
    print(tf.config.list_physical_devices(device_type=None))
    print('**************************************************')

    left_model = None
    right_model = None
    total_right_model_time = 0

def post_process_batch(batch):
    df = pd.DataFrame.from_dict(batch)
    json_data = {}
    encoded_data_list = df['data'].tolist()
    # Decode the data strings into numpy arrays
    decoded_data_list = [np.frombuffer(tf.io.decode_raw(base64.b64decode(encoded_data), tf.float32), dtype=np.float32) for encoded_data in encoded_data_list]
    concatenated_data = np.concatenate(decoded_data_list, axis=0)
    # Convert the concatenated data to a tensor
    tensor_data = tf.convert_to_tensor(concatenated_data)
    # Convert the byte array to a tensor
    lst = [len(batch)] + df['shape'].tolist()[0]
    tensor_shape = tuple(lst)
    tensor = tf.reshape(tensor_data, tensor_shape)
    # Updating Dictionary Values
    json_data["data"] = tensor
    client_id = df['client_id'].tolist()[0]
    json_data["client_id"] = client_id
    
    batch_response_dict = process_batch(json_data)
    
    return batch_response_dict
        
def model_right(data_batch):
    right_model_output = None
    frame_seq_nos = []
    with tf.device(device):
        try:
            print(data_batch.shape,"data_batch")
            Logger.log(f'Executing right model for number of frames: {frames_to_process}')
            right_model_start_time = datetime.datetime.now()
            left_model_output = tf.stack(data_batch)
            print(left_model_output.shape)
            reshaped_tensor = tf.expand_dims(tf.reshape(left_model_output, input_shape), axis=0)
            model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
            model_custom = models.Model(inputs=model.input, outputs=model.output)
            right_model_output = model_custom(reshaped_tensor)  # Pass the stacked_tensors as a list
        except BaseException as e:
            print('Failed to do something: ' + str(e))
        right_model_end_time = datetime.datetime.now()
        add_to_total_right_model_time((right_model_end_time - right_model_start_time).total_seconds())
        return_data_batch = {'result': right_model_output, 'frame_seq_no': frames_to_process, 'right_model_time':total_right_model_time}
        print('executed    ')
        return return_data_batch
        

def process_batch(client_data_dict):
    with tf.device(device):
        data_val = client_data_dict['data']
        output_data_val = model_right(data_val)
        #batch_response_dict = dict(zip(data_keys, output_data_val))
        return output_data_val

def add_to_total_right_model_time(current_frame_exec_time):
    global total_right_model_time
    total_right_model_time += current_frame_exec_time


def handle_client(conn, addr):
    global lst_load_batches
    try:
        buffer2 = None
        buffer = bytearray()
        for i in range(len(split_points)*frames_to_process):
            for i in range(LIMITS_no):
                # Receive a packet
                packet_data = conn.recv(SIZE)
                if buffer2 != None:
                    packet_data = buffer2 + packet_data
                    buffer2 = None
                # Append the received packet data to the buffer
                buffer.extend(packet_data)
                try:
                    index = buffer.index(b"}")
                    # Create a new byte array up to and including the "}" symbol
                    temp1 = buffer[index + 1:]
                    buffer = buffer[:index + 1]
                    temp2 = buffer2
                    if temp1 is not None and temp2 is not None:
                        temp1 += temp2
                        buffer2 = temp1
                except:
                    pass
            print("buffer",len(buffer))
            print(f"Received data from {addr}: {buffer}")
            lst_load_batches.append(json.loads(buffer.decode()))
            print("Appended!!!!!!!!", len(lst_load_batches))
    except BaseException as e:
        print('handle_client: ' + str(e))

def handle_client_batch(conn, addr, batch):
    for j in range(len(split_points)):  
        result = post_process_batch(batch)
        post_client(conn, addr, result)
        print("Client handled successfully",j)
    

def post_client(conn, addr, result):
    post_data = {}
    data = tf.stack(result['result'])
    try:
        encoded_data = base64.b64encode(data.numpy().tobytes()).decode()
        post_data = {'result':encoded_data, 'frame_seq_no':result['frame_seq_no'],'right_model_time':result['right_model_time']}
        byte_array = json.dumps(post_data).encode()
        print('byte_array',len(byte_array))
        conn.sendto(byte_array, ADDR)
        offset = 0
        while offset < len(byte_array):
            # Copy a portion of the byte array into the packet
            packet_data = byte_array[offset:offset + SIZE]

            # Send the packet over UDP
            conn.sendto(packet_data, ADDR)

            # Update the offset
            offset += SIZE
    except BaseException as e:
        print("post_client: " + str(e))

def run_clients(lst_conn, lst_addr):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = []
            for conn, addr in zip(lst_conn, lst_addr):
                print("Got connection over : ",addr)
                future = pool.submit(handle_client, conn, addr)
                futures.append(future)
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
        print(len(lst_load_batches),"lst_load_batches")
    except BaseException as e:
        print("post_client: " + str(e))
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = []
        client_id1 = []
        client_id2 = []
        client_id3 = []
        client_id4 = []
        for i in lst_load_batches:
            if i['client_id'] == "client_1":
                client_id1.append(i)
            elif i['client_id'] == "client_2":
                client_id2.append(i)
            elif i['client_id'] == "client_3":
                client_id3.append(i)
            elif i['client_id'] == "client_4":
                client_id4.append(i)
        print(len(client_id1),len(client_id2),len(client_id3),len(client_id4),"len(client_id)")
        lst = [client_id1, client_id2, client_id3, client_id4]
        for conn, addr, batch in zip(lst_conn, lst_addr, lst):
            future = pool.submit(handle_client_batch, conn, addr, batch)
            futures.append(future)
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    time.sleep(5)
    
def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print(f"[LISTENING] Server is listening on {IP}:{PORT}")
    i = 0
    lst_conn = []
    lst_addr = []
    while i < BATCH_SIZE:
        conn, addr = server.accept()
        lst_conn.append(conn)
        lst_addr.append(addr)
        i += 1
    run_clients(lst_conn, lst_addr)


if __name__ == "__main__":
    main()
