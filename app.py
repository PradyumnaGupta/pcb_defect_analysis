from flask import Flask, render_template, url_for, flash, redirect,request,send_from_directory
import os
from werkzeug.utils import secure_filename
from model_inference.inference import load_model,model_inference
import cv2
import time

UPLOAD_FOLDER = './uploads/'
BUFFER_FOLDER = './temp_buffer/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BUFFER_FOLDER'] = BUFFER_FOLDER

model = load_model()

@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():
    if request.method == "POST":
        if 'test_image' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        test_image = request.files['test_image']
        if test_image.filename == '':
            flash('No test image selected','danger')
            return redirect(request.url)
        template_image = request.files['template_image']
        if template_image.filename == '':
            flash('No test image selected','danger')
            return redirect(request.url)
        if test_image and template_image:
            test_image_filename = secure_filename(test_image.filename)
            template_image_filename = secure_filename(template_image.filename)
            path1 = os.path.join(app.config['UPLOAD_FOLDER'], test_image_filename)
            path2 = os.path.join(app.config['UPLOAD_FOLDER'], template_image_filename)
            test_image.save(path1)
            template_image.save(path2)
            result = model_inference(model,path1,path2)
            result_image_name = test_image_filename+template_image_filename
            cv2.imwrite(app.config['BUFFER_FOLDER']+result_image_name,result)
            return render_template('home.htm',test_image=url_for('uploaded_file',filename=test_image_filename),template_image=url_for('uploaded_file',filename=template_image_filename),result_image=url_for('generated_file',filename=result_image_name),time=str(time.time()),name="home")
    else:
        return render_template('home.htm',time=str(time.time()),name="home")

@app.route('/about')
def about():
    return render_template('about.htm',test_image=None,template_image=None,result_image=None,time=str(time.time()),name="about")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/temp_buffer/<filename>')
def generated_file(filename):
    return send_from_directory(app.config['BUFFER_FOLDER'],filename)

if __name__=="__main__":
    # cache_buster.register_cache_buster(app)
    app.run(debug=True)
    