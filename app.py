"""
Skin cancer detection web app
Designed and developed by Wisdom ML
"""

from __future__ import division, print_function
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from heatmap import save_and_display_gradcam, make_gradcam_heatmap

# Disable GPU to run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define a Flask app
app = Flask(__name__, static_url_path='')

# Configuration for file and heatmap folders
app.config['HEATMAP_FOLDER'] = 'heatmap'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Model path
MODEL_PATH = 'C:/skin_cancer_detection_webapp - Copy/models/model_v1.h5'

# Load the trained model
model = load_model(MODEL_PATH)
print('Model loaded. Ready to start serving...')

# Class labels for prediction
class_dict = {
    0: "Atopic Dermatitis",
    1: "Basal Cell Carcinoma",
    2: "Benign Keratosis-like Lesions",
    4: "Healthy",
    5: "Melanocytic Nevi",
    6: "Melanoma",
    7: "Psoriasis pictures Lichen Planus and related diseases",
    8: "Seborrheic Keratoses and other Benign Tumors",
    9: "Tinea Ringworm Candidiasis and other Fungal Infections"
}

@app.route('/uploads/<filename>')
def upload_img(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

def model_predict(img_path, model):
    img = Image.open(img_path).resize((224, 224))  # Ensure target size matches the trained model's input size

    # Preprocess the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255

    # Make predictions
    preds = model.predict(img)[0]
    prediction = sorted(
        [(class_dict[i], round(j * 100, 2)) for i, j in enumerate(preds)],
        reverse=True,
        key=lambda x: x[1]
    )
    return prediction, img

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']

        # Save the file to the 'uploads' directory
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        print(file_path)

        # Save the file
        f.save(file_path)
        file_name = os.path.basename(file_path)

        # Make prediction
        pred, img = model_predict(file_path, model)

        # Generate GradCAM heatmap
        last_conv_layer_name = "block_16_depthwise"  # Change this to a valid layer name
        heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)
        fname = save_and_display_gradcam(file_path, heatmap)

        return render_template('predict.html', file_name=file_name, heatmap_file=fname, result=pred)

    return render_template('predict.html')

# Run the app on localhost:5000
if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5000)
