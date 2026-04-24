import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ---------------- CONFIG ----------------
IMG_SIZE = 128
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/output"
MODEL_PATH = r"C:\FINAL_YEAR_PRO\FINAL_COLLEGE_PROJECT_CODE\FINAL_PROJECT_PHASE _1\epoch_two_five_model.keras"
LAST_CONV_LAYER_NAME = "last_conv"

CLASSES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# ---------------- MODEL ----------------
def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu', name=LAST_CONV_LAYER_NAME)(x)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(CLASSES), activation='softmax')(x)
    return Model(inputs, outputs)

model = build_model()
model.load_weights(MODEL_PATH)

# ---------------- GRADCAM ----------------
def get_img_array(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model([model.input], 
                       [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def generate_gradcam(img_path, pred_class):
    img_array = get_img_array(img_path)
    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME, pred_class)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    gradcam_path = os.path.join(GRADCAM_FOLDER, os.path.basename(img_path))
    cv2.imwrite(gradcam_path, superimposed_img)
    return gradcam_path

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    img_array = get_img_array(img_path)
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = float(np.max(preds[0]) * 100)

    gradcam_path = generate_gradcam(img_path, pred_class)

    return render_template('result.html',
                           original=url_for('static', filename='uploads/' + file.filename),
                           gradcam=url_for('static', filename='output/' + file.filename),
                           prediction=CLASSES[pred_class],
                           confidence=round(confidence, 2))

if __name__ == '__main__':
    app.run(debug=True)