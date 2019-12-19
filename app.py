from flask import Flask,render_template,url_for,request,g, flash, redirect
from werkzeug.utils import secure_filename

import os 
import cv2

## Upload parameter
UPLOAD_FOLDER = os.path.join('static', 'uploaded_images')
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']

## FLASK configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    # checks if extension in filename is allowed
    return '.' in filename and \
           str(filename.rsplit('.', 1)[1]).lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home_page():
    return render_template('page.html')

@app.route('/a')
def heal():
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

            img = cv2.imread(img_path)
            prediction = img.shape[0]

            message = f"Model prediction: {prediction}"

    else: 
        message = "No Images Uploaded"
        img_path = '.'
                
    return render_template('page.html',
                            message = message, 
                            img_path = img_path,
    )
                            


if __name__ == '__main__':
    app.run(debug=True)