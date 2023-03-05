from flask import Flask,render_template,request,redirect,url_for
import os
from main import process
from cleanup import clean

UPLOAD_FOLDER = './static'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/",methods=['GET','POST'])
def home():
    try:
        if request.method == 'POST':
            if 'file1' not in request.files:
                return 'there is no file1 in form!'
            file1 = request.files['file1']
            path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            file1.save(path)
            process(path)
            clean()
            return redirect(url_for('video'))
        return render_template('frontpage.html')
    except:
        return render_template('frontpage.html')

@app.route("/video/",methods=['GET','POST'])
def video():
    if request.method == 'POST':
        return redirect(url_for('home'))
    return render_template('video.html')


if (__name__=='__main__'):
    app.run(debug=True)