import os 

from flask import Flask,render_template,url_for,request,g, flash, redirect
from werkzeug.utils import secure_filename

## Upload parameter
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']

## FLASK configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    # checks if extension in filename is allowed
    return '.' in filename and \
           str(filename.rsplit('.', 1)[1]).lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home_page():
    ### UPLOAD FILE
    if 'uploadfile' in request.files.keys():
        submission_file = request.files['uploadfile']
        #throw error if extension is not allowed
        if not allowed_file(submission_file.filename):
            raise Exception('Invalid file extension')

        elif submission_file:
            filename = secure_filename(submission_file.filename)

            target_dir = os.path.join(app.config['UPLOAD_FOLDER'])
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            fullPath = os.path.join(app.config['UPLOAD_FOLDER'] , filename)
            submission_file.save(fullPath)
                
            return redirect(url_for('home_page'))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)