from IPython.display import Image
from matplotlib import pyplot as plt
import pandas as pd, numpy as np
pd.options.display.float_format = '{:,.2f}'.format

from google.cloud import vision
import io

import warnings
warnings.simplefilter("ignore")

import os
import cv2
import re
import ntpath

from google.cloud import vision
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\Users\amrit\Documents\GitHub\Video-search\electron-app\public\keyFile.json"

def set_gcp_key(key_filepath):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_filepath

def CloudVisionTextExtractor(handwritings):
    # convert image from numpy to bytes for submittion to Google Cloud Vision
    _, encoded_image = cv2.imencode('.png', handwritings)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    
    # feed handwriting image segment to the Google Cloud Vision API
    client = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=image)
    
    return response

def getTextFromVisionResponse(response):
    texts = []
    for page in response.full_text_annotation.pages:
        for i, block in enumerate(page.blocks):  
            for paragraph in block.paragraphs:       
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    texts.append(word_text)

    return ' '.join(texts)



def get_vid_text(timestamp_dict):
    vid_caption_dict = {}
    for filepath in timestamp_dict:
        timestamp = timestamp_dict[filepath]

        vidcap = cv2.VideoCapture(str(filepath))
        caption_dict = {}

        for time in timestamp:
            time_int = int(time*1000)
            vidcap.set(cv2.CAP_PROP_POS_MSEC,time_int)
            success,image = vidcap.read()
            response = CloudVisionTextExtractor(image)
            image_text = getTextFromVisionResponse(response)
            caption_dict[time] = image_text

        vid_caption_dict[filepath] = caption_dict

    return vid_caption_dict





