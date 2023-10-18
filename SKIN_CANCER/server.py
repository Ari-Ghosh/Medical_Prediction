from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename
import prediction
import os


app = Flask("__main__")
Uploaded_images = "Uploaded_images"
app.config['Uploaded_images'] = Uploaded_images

@app.route('/', methods=['POST', 'GET'])
def homepage():
  if request.method == 'GET':
      out = "The server is running on port 8000"
      return out

@app.route('/predict', methods=['POST'])
def getOutput():
  output=""
  if request.method == 'POST':
        myimage = request.files.get('data')
        imgname = secure_filename(myimage.filename)
        imgpath = "Uploaded_images/"+imgname
        myimage.save(os.path.join(app.config["Uploaded_images"], imgname))
        output = prediction.prediction(imgpath)
        print(output)
        return output
      

app.run(port=8080);