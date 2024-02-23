import socket
import cv2
import math
import datetime
import pickle
import tensorflow as tf
import json
import numpy as np
import pandas as pd
import warnings
import io
from tensorflow import keras
from PIL import Image
from modelUtils import Config, Logger, write_to_csv, get_flops, dry_run_left_model, get_model, convert_image_to_tensor, input_size, my_split_points
from model_profiler import model_profiler
import sys
import base64
import threading
import time as t

io.StringIO
warnings.filterwarnings('ignore')

# Read the configurations from the config file.
config = Config.get_config()
metrics_headers = ['frames_to_process', 'split_no', 'total_processing_time', 'single_frame_time', 'left_output_size', 'avg_consec_inference_gap', 'total_left_model_time','total_right_model_time']
# Assign the configurations to the global variables.
device = config['client_device']
url = config['url']
model_name = config['model']
frames_to_process = config['frames_to_process']

# Initialize the start time to None. This value will be set in main_runner when it is initialized.
start_time = None
client = None

# Track total responses handled.
total_handled_responses = 0

# Total inference gap. Sum of the time gaps between two consecutive inferences.
total_inference_gap = 0
total_left_model_time = 0
total_right_model_time  = 0
prev_frame_end_time = None
left_output_size = 0
flops = 0

IP = "localhost"
PORT = 5566
ADDR = (IP, PORT)
FORMAT = "utf-8"
SIZE = 45000
DENSE = 535285
RESNET = 1070497
LIMIT = int(((RESNET * frames_to_process)/2))
LIMITS_no = int(math.floor(LIMIT /SIZE) + 1)


def get_left_model(split_point):  
    split_layer = model.layers[split_point]
    left_model = keras.Model(inputs=model.input, outputs=model.output)
    return left_model
# Returns JSON body for sending to server.
def get_request_body(left_output, frame_seq_no, split_point):
    # Request JSON.
    request_body = {'client_id': "client_1", 'total_frames_no': frame_seq_no, 'split_point': split_point}
    json_data = json.dumps(request_body).encode()
    return json_data

def producer_video_left(img, left_model):
    size = input_size.get(model_name)
    tensor = convert_image_to_tensor(img, size)
    out_left = left_model(tensor)    
    return tensor

def main_runner():
    global split_point
    global flops
    sent = 0
    for split_point in split_points:
        print('SPLIT: ' + str(split_point))
        left_model = get_left_model(split_point)
        dry_run_left_model(left_model,224)
        profile = model_profiler(left_model, frames_to_process)
        flops = get_flops(profile)
        # Request JSON.
        #request_json = get_request_body(left_model, frames_to_process, split_point)
        #client.send(request_json)
        Logger.log(f'SPLIT POINT IS SET TO {split_point} IN SERVER')
        frame_seq_no = 1
        global start_time
        global left_output_size
        global total_handled_responses
        global total_left_model_time    
        # Read the input from the file.
        cam = cv2.VideoCapture('hdvideo.mp4')
        # This is the start of the video processing. Initialize the start time.
        while frame_seq_no < frames_to_process + 1:
            # Reading next frame from the input.       
            ret, img_rbg = cam.read()   
            # If the frame exists
            if ret:                 
                Logger.log(f'[Inside main_runner] total_frames_no # {frame_seq_no}: Send for left processing.')    
                # Send the frame for left processing.
                out_left = producer_video_left(img_rbg, left_model)  
                print(out_left.shape) 
                if out_left.shape != (1, 224, 224, 3):
                    # Resize the tensor
                    resized_tensor = tf.image.resize(out_left, size=(224, 224))
                else:
                    resized_tensor = out_left
                data = resized_tensor
                frame_seq_no = frame_seq_no
                Logger.log(f'[Inside consumer] Frame # {frame_seq_no}: Preparing body to send request to server.')
                stack = tf.stack(list(data))
                encoded_data = base64.b64encode(stack.numpy().tobytes()).decode()
                post_data = {'client_id': "client_1",'data':encoded_data, 'shape':list(stack.shape),'frame_seq_no': frame_seq_no, 'split_point': split_point}
                # Serialize the JSON object to a byte array
                byte_array = json.dumps(post_data).encode()
                print("byte_array",len(byte_array))
                start_time = datetime.datetime.now()
                sent += 1
                offset = 0
                while offset < len(byte_array):
                    # Copy a portion of the byte array into the packet
                    packet_data = byte_array[offset:offset + SIZE]

                    # Send the packet over UDP
                    client.send(packet_data)

                    # Update the offset
                    offset += SIZE
                
            # Increment frame count after left processing.    
            frame_seq_no += 1  

        # This is the end of the left processing. Set the end time of left video processing.
        end_time = datetime.datetime.now()       
        left_output_size = tf.size(out_left).numpy()
        total_left_model_time = (end_time - start_time).total_seconds()    
    
        cam.release()
        cv2.destroyAllWindows()

        # Wait until all the responses for this split point are received and processed.
        total_handled_responses = 0
    print("sent "+ str(sent) + " packets!!!!!!")

def handle_response():
    global total_handled_responses
    global total_inference_gap
    global prev_frame_end_time
    buffer2 = None
    buffer = bytearray()
    for k in split_points:
        try:
            for size in range(LIMITS_no):
                # Receive a packet
                packet_data = client.recv(SIZE)
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
                    temp1 += temp2
                    buffer2 = temp1
                except:
                    pass
            print(f"Received processed buffer : {buffer.decode()}")
        except BaseException as e:
            print('handle_response: ' + str(e))
        load_batch_data = json.loads(buffer.decode())
        total_right_model_time = load_batch_data['right_model_time']
        #for load_data in load_batch_data:
        result = load_batch_data['result']
        frame_seq_no = load_batch_data['frame_seq_no']

        Logger.log(f'Processed frame # {frame_seq_no}')
        total_handled_responses += 1
        # First frame that is processed. Record its end time. 
        if total_handled_responses == 1:
            prev_frame_end_time = datetime.datetime.now()
        else:
            curr_frame_end_time = datetime.datetime.now()
            total_inference_gap += (curr_frame_end_time - prev_frame_end_time).total_seconds()
            prev_frame_end_time = curr_frame_end_time

        
        # Calculate total time taken to process all 50 frames.
        end_time = datetime.datetime.now()
        time = (end_time - start_time).total_seconds()
        single_frame_time = time/frames_to_process
        # Calculate average inference gap between two consequtive frames
        avg_consec_inference_gap = total_inference_gap/(frames_to_process - 1)
        # Reset to zero for next loop.
        total_inference_gap = 0
        #request_total_right_model_time_from_server()
        log_metrics(k, time, single_frame_time, left_output_size, avg_consec_inference_gap, total_left_model_time,total_right_model_time)
     

def log_metrics(split_point, time, single_frame_time, left_output_size, avg_consec_inference_gap, total_left_model_time,total_right_model_time):
    write_to_csv(model_name + '_async1' + '.csv', metrics_headers, [frames_to_process, split_point, time, single_frame_time, left_output_size, avg_consec_inference_gap, total_left_model_time,total_right_model_time])
    Logger.log(f'CONSECUTIVE INFERENCE GAP BETWEEN TWO FRAMES:: {avg_consec_inference_gap}')
    Logger.log(f'PROCESSING TIME FOR SINGLE FRAME:: {single_frame_time} sec')
    Logger.log(f'TOTAL LEFT PROCESSING TIME:: {total_left_model_time}')
    Logger.log(f'TOTAL PROCESSING TIME:: {time} sec')

def main():
    global client
    global model
    global num_layers
    global split_points
    model = get_model(model_name)
    num_layers = len(model.layers)  
    split_points = my_split_points.get(model_name)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)
    t.sleep(5)
    main_runner()
    handle_response()
        

if __name__ == "__main__":
    main()