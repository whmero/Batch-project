import asyncio
import asyncio
import tornado
import pickle
import base64
from tornado.queues import Queue
from tornado.concurrent import Future
import numpy as np
from vit_keras import vit
from tensorflow.keras.models import Model
import datetime
from tensorflow.keras.applications import DenseNet121, ResNet50, ResNet101
import cv2
from collections import defaultdict
import logging

# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Create a dictionary to hold requests based on client_id
client_requests = defaultdict(list)

request_queue = Queue()
batch_size = 4
#input_shape = (224, 224, 3)
input_shape = (32, 32, 3)
'''
models = dict()
models_splitted = dict()
models['DensetNet121'] = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
models['Resnet50'] = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
models['Resnet101'] = ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)
models['VIT'] = vit.vit_b32(image_size=32,activation='softmax',pretrained=True,include_top=True,pretrained_top=False,classes=2)
split_lst_idx = 0
for model_name in models:
    if model_name == 'VIT':
        split_indices = [3, 17]
    elif model_name == 'Resnet50' or model_name == 'Resnet101':
        split_indices = [5, 91]
    else:
        split_indices = [5, 94]
    if split_lst_idx !=0:
        split_index = split_indices[split_lst_idx - 1]
        layer = models[model_name].layers[split_index + 1]
        #save splitted part of model belonging to the server in models dictionary
        models_splitted[model_name] = Model(inputs=layer.input, outputs=models[model_name].output)

'''
models = dict()
models_splitted = dict()
models['DensetNet121'] = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
models['Resnet50'] = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
models['Resnet101'] = ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)
models['VIT'] = vit.vit_b32(image_size=32,activation='softmax',pretrained=True,include_top=True,pretrained_top=False,classes=2)

async def process_requst():
    batch_input = []
    async for req in request_queue:
        batch_input.append(req)
        if len(batch_input) == batch_size:
            logging.info('Batch execution starts')
            process_batch(batch_input)
            logging.info('Batch execution ends')
            batch_input.clear()
        

def get_model(model_name, split_index):
    if split_index == 0:
        return models[model_name]
    else:
        layer = models[model_name].layers[split_index + 1]
        #save splitted part of model belonging to the server in models dictionary
        return Model(inputs=layer.input, outputs=models[model_name].output)


def process_batch(batch):
    logging.info(f'Processing {len(batch)} inputs')

    model_name = batch[0][2]
    split_index = batch[0][3]

    # Load the images for this batch
    imgs = [req[4] for req in batch]
    imgs = np.array(imgs)
    model = get_model(model_name, split_index)
    if split_index != 0:
        imgs = np.squeeze(imgs, axis=1)
    '''
    if split_index != 0:
        imgs = np.squeeze(imgs, axis=1)
        model = models_splitted[model_name]
    else:
        model = models[model_name]
    '''
    start_time = datetime.datetime.now()
    results = model.predict(imgs)
    end_time = datetime.datetime.now()
    time_batch = (end_time - start_time).total_seconds()
    for i in range(len((results))):
        client_id = batch[i][0]
        req_id = batch[i][1]
        model_name = batch[i][2]
        reply_future = batch[i][5]

        result = results[i]

        reply_data = {
                    'client_id': batch[i][0],
                    'result': result.tolist(),  # Assuming that result is a numpy array.
                    'model_name': model_name,
                    'time_batch': time_batch,
                    'batch': batch_size,
                    'req_id': req_id
                }
        reply_future.set_result(reply_data)
        request_queue.task_done()
        logging.info(f'Done input {i}: Reply for {client_id} -- {req_id} :: Excution time {time_batch}')
            

class MainHandler(tornado.web.RequestHandler):
    async def post(self):
        received_data = base64.b64decode(self.request.body)
        # Deserialize the numpy array from the received byte stream
        req = pickle.loads(received_data)
        # Extracting data from the client request  
        client_id = req['client_id']
        req_id = req['request_id']
        model_name = req['model_name']
        split_index = int(req['split_index'])
        logging.info("Received request " + str(req_id) + " from client: " + str(client_id) + " to be processed with model: " + str(model_name))
    
        # Pre-processing of image
        img = req['image'].astype(np.uint8)
        if split_index == 0:
            if model_name != 'VIT': 
                img = cv2.resize(img, (224, 224))
            else: 
                img = cv2.resize(img, (32, 32))

        # Put the req in the request queue
        reply_future = Future()
        request_queue.put((client_id, req_id, model_name, split_index, img, reply_future))

        # Await the reply_future to get the result
        result = await reply_future

        # Construct the response object with the result
        response = {
            'client_id': result['client_id'],
            'result': result['result'],
            'model_name': result['model_name'],
            'batch': result['batch'],
            'time_batch': result['time_batch'],
            'req_id': result['req_id'],
        }

        # Send the pickled response back to the client
        self.write(pickle.dumps(response))




def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

async def main():
    tornado.ioloop .IOLoop.current().add_callback(process_requst)
    app = make_app()
    app.listen(8080)
    print("server is ready...........................................................")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())