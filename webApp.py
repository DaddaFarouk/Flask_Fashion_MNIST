import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, flash, redirect
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/images/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
class_names = ['T-shirt/top', 'Pantalon', 'Pullover', 'Robe', 'Manteau',
               'Sandale', 'Chemise', 'Basquette', 'Sac', 'Bottine']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(filename):
    img_dir = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model = tf.keras.models.load_model('static/model/faroukModel.h5')
    img = Image.open(img_dir).convert('L')
    resized = img.resize((28, 28))
    numpy_image = np.asarray(resized)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(np.reshape(numpy_image, (1, 28, 28)))
    return np.argmax(predictions[0])


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            predicted_value = predict(filename)
            return '''
                        <!doctype html>
                        <title>Upload new File</title>
                        <h1>La prediction pour cette image est : '''+class_names[predicted_value]+'''</h1>
                    '''
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    app.run(debug=True)
