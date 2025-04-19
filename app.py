import os
import zipfile
import numpy as np
import gdown
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# === Download & unzip model if needed ===
MODEL_DIR = 'mien_fabric_classifier_savedmodel'
ZIP_PATH = 'model.zip'
FILE_ID = '1rQDfAqMZqK8D_sYLypN50jgKcV_j2pjS'

if not os.path.exists(MODEL_DIR):
    print("📦 Downloading model...")
    gdown.download(f'https://drive.google.com/uc?export=download&id={FILE_ID}', ZIP_PATH, quiet=False)

    print("📂 Extracting model...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# === Load SavedModel ===
model = load_model(MODEL_DIR)
print("✅ Model loaded successfully.")

# === Class labels ===
CLASS_LABELS = ['Mien_pattern_01', 'Mien_pattern_02', 'Mien_pattern_03', 'Mien_pattern_04']

# === YouTube story links ===
YOUTUBE_LINKS = {
    'Mien_pattern_01': 'https://youtu.be/zkrltLG0r9w',
    'Mien_pattern_02': 'https://youtu.be/example_for_02',
    'Mien_pattern_03': 'https://youtu.be/example_for_03',
    'Mien_pattern_04': 'https://youtu.be/example_for_04'
}

# === Web route ===
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None
    youtube_link = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            result = CLASS_LABELS[predicted_index]
            youtube_link = YOUTUBE_LINKS.get(result)

    return render_template('index1.html', result=result, filename=filename, youtube_link=youtube_link)

# === Run app on dynamic port (Render/Heroku) ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
