import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
MODEL_PATH = 'mien_fabric_classifier_forapp.h5'

# ✅ Download model if not found locally
if not os.path.exists(MODEL_PATH):
    file_id = '1AB3tFMw6K8iL-RnGt0TUwQlvA53-ccvd'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

# Class labels
CLASS_LABELS = ['Mien_pattern_01', 'Mien_pattern_02', 'Mien_pattern_03', 'Mien_pattern_04']

# Mapping predicted labels to YouTube URLs
YOUTUBE_LINKS = {
    'Mien_pattern_01': 'https://youtu.be/zkrltLG0r9w',
    'Mien_pattern_02': 'https://youtu.be/example_for_02',
    'Mien_pattern_03': 'https://youtu.be/example_for_03',
    'Mien_pattern_04': 'https://youtu.be/example_for_04'
}

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

# ✅ Dynamic port support for Railway/Heroku
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
