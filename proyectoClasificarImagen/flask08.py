from flask import Flask, request
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from io import BytesIO
from PIL import Image

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Cargar el modelo entrenado
model = keras.models.load_model("C:/Users/logij/Downloads/ImageClasificacion.keras")

# Definir el tamaño de la imagen que espera el modelo
image_size = (180, 180)  # Ajusta esto según lo que tu modelo requiera

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploader', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        # Leer la imagen directamente sin guardarla en el disco
        image_stream = BytesIO(file.read())  # Leer el archivo en memoria
        img = Image.open(image_stream).convert('RGB')  # Abrir la imagen y convertirla a RGB
        img = img.resize(image_size)  # Redimensionar la imagen al tamaño que requiere el modelo

        # Convertir la imagen a un array
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Agregar una dimensión para el batch
        
        # Hacer la predicción con el modelo
        predictions = model.predict(img_array)
        score = tf.nn.sigmoid(predictions[0][0]).numpy()  # Convertir la predicción a un valor entre 0 y 1
        
        # Devolver el resultado de predicción
        return f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog."
    else:
        return 'No allowed extension'

if __name__ == '__main__':
    app.run(debug=True)
