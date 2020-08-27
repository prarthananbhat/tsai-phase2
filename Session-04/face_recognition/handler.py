try:
    import unzip_requirements
except ImportError:
    pass
from requests_toolbelt.multipart import decoder
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from torch import nn
import traceback
import boto3
import os
import tarfile
import io
import base64
import json
import traceback
import numpy as np
print("import End")

print("Initialse S3 Bucket and Modle Path")
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'tsai-session1-models'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'new_model.pt'

print('Downloading model ...')
print(S3_BUCKET)
print(MODEL_PATH)

s3 = boto3.client('s3')

try:
    # get object from s3
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream")
        print("*********** This is printing in console")
        # read it in memory
        bytestream = io.BytesIO(obj['Body'].read())
        print(bytestream)
        print("Loading Model")
        model = torch.jit.load(bytestream)
        model.eval()
        if model:
            print("Model Loaded")
        else:
            print("Unable to Load model")

except Exception as e:
    traceback.print_exc()
    print(repr(e))
    raise(e)


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)


def get_prediction(image_bytes):
    class_names = ['Andrw_Ng', 'Arnab_Goswami', 'Arvind_Kejriwal', 'Elon_Musk', 'MS_Dhoni', 'Mary_Kom', 'Narayan_Mrthy', 'Rahul_Gandhi', 'Smriti_Irani', 'Sushanth_Singh_Rajput']
    tensor = transform_image(image_bytes=image_bytes)
    cpu_model = model.cpu()
    cpu_model.eval()
    outputs = cpu_model(tensor)
    _, preds = torch.max(outputs, 1)
    return class_names[preds], outputs[0]


def get_response_image(image):
    # pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG') # convert the PIL image to byte array
    encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii') #     encode as base64
    return encoded_img



def classify_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        b64img = event['body']
        print(type(b64img))
        img_bytes = base64.decodebytes(base64.b64decode(b64img))
        print(img_bytes)
        print(type(img_bytes))
       

        prediction, confidence = get_prediction(image_bytes=img_bytes)
        image = Image.open(io.BytesIO(img_bytes))
        res_image = get_response_image(image)
        
        # res_image = "data:image/jpeg;base64,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"
        message = 'Face Recognition was succesfull'
        response = {'Status': 'Success', 'message': message, 'prediction': prediction, 'image' : res_image, 'confidence' : confidence.detach().numpy().tolist()}
        # img_bytes = image.tobytes()
        # bytes_encoded = base64.b64encode(img_bytes)
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            # "body": response, "isBase64Encoded":"true"
            "body" : json.dumps(response)
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'multipart/form-data',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }


