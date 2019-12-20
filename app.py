from flask import Flask,render_template,url_for,request,g, flash, redirect
from werkzeug.utils import secure_filename

import numpy as np
from urllib.request import urlopen
import os 
import cv2
import time
import argparse

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from model_utils.config import LABELS

# Parameter & functions
parser = argparse.ArgumentParser(description='ImageNet')
parser.add_argument('--cpu', default=True, help='Use cpu inference')
args = parser.parse_args()

## Upload parameter
UPLOAD_FOLDER = os.path.join('static', 'uploaded_images')
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']

## Model parameter
device = torch.device("cpu" if args.cpu else "cuda")

## FLASK configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

## Define Model
net = models.mobilenet_v2(pretrained=True)
net = net.to(device)
net.eval()
image_size = (224,224)

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
])

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image

def url_to_filename(url):
    return str(url).split("/")[-1]

def predict_image(img):
    img = transformation(img)
    img = img.unsqueeze(0).float()
    img = img.to(device)

    predicted_index = int(net(img).argmax())
    pred = LABELS[predicted_index] ## Prediction label 
    return pred 

def allowed_file(filename):
    # checks if extension in filename is allowed
    return '.' in filename and \
           str(filename.rsplit('.', 1)[1]).lower() in ALLOWED_EXTENSIONS

# Route
@app.route('/')
def home_page():
    return render_template('page.html')

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    ### UPLOAD FILE
    if 'uploadfile' in request.files.keys():
        submission_file = request.files['uploadfile']
        #throw error if extension is not allowed
        if not allowed_file(submission_file.filename):
            message = "Only Accept jpg, jpeg and png format"
            img_path = '.'

        elif submission_file:
            filename = secure_filename(submission_file.filename)

            target_dir = os.path.join(app.config['UPLOAD_FOLDER'])
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            img_path = os.path.join(".", app.config['UPLOAD_FOLDER'] , filename)
            submission_file.save(img_path)
            ### read sesize and predict
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            prediction = predict_image(img)
            message = f"Model prediction: {prediction}"

    else: 
        message = "No Images Uploaded"
        img_path = '.'

    return render_template('page.html',
                            message = message, 
                            img_path = img_path,
                            )

@app.route('/urlprediction', methods=['GET', 'POST'])
def urlpredict():
    ### Insert Url

    if 'urlimage' in request.form.keys():
        inserted_url = request.form['urlimage']
        filename = url_to_filename(inserted_url)
        if len(inserted_url) == 0:
            message = "No Url Inserted"
            img_path = '.'

        elif not allowed_file(filename):
            message = "Only Accept jpg, jpeg and png format"
            img_path = '.'

        else: 
            img_path = os.path.join(".", app.config['UPLOAD_FOLDER'] , filename)
            ### read sesize and predict
            img = url_to_image(inserted_url)
            img = cv2.resize(img, image_size)
            prediction = predict_image(img)
            message = f"Model prediction: {prediction}"
            # show image using url
            img_path = inserted_url

    else: 
        message = "No Url Inserted"
        img_path = '.'
                
    return render_template('page.html',
                            message = message, 
                            img_path = img_path,
                            )
                        
if __name__ == '__main__':
    app.run(debug=True)