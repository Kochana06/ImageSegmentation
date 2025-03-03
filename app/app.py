import os
import torch
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from src.prediction.prediction import predict

app = Flask(__name__)


# Configure upload folders
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SEG_FOLDER'] = 'static/segmentations'

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SEG_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template('index.html')

# Home page: Upload file
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            # Save the file to the uploads folder
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(upload_path)
            # Create a segmentation filename based on the original filename
            seg_filename = "seg_" + file.filename
            # Redirect to the prediction page with both filenames as parameters
            return redirect(url_for('predict_page', filename=file.filename, seg_filename=seg_filename))
    return render_template('home.html')


@app.route('/predict/<filename>/<seg_filename>')
def predict_page(filename, seg_filename):
    upload_path = os.path.join("static/uploads", filename)

    pred_class, uploaded_filename, seg_filename, tumor_size_pixels, tumor_size_mm2 = predict(upload_path, seg_filename)

    return render_template(
        'predict.html', 
        filename=uploaded_filename, 
        seg_filename=seg_filename, 
        prediction=f"Predicted Class: {pred_class}",
        tumor_size_pixels=tumor_size_pixels,
        tumor_size_mm2=round(tumor_size_mm2, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)
