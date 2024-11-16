import cv2
import torch
from flask import Flask, render_template, Response
from torchvision import transforms
from PIL import Image
import numpy as np

app = Flask(__name__)

# Cargar el modelo YOLOv5 preentrenado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' es el modelo pequeño de YOLOv5

# Cargar el modelo de predicción de género
gender_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
gender_model.eval()

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara")
    exit()

# Función para predecir el género
def predict_gender(cropped_face):
    if cropped_face is None or cropped_face.size == 0:
        return "Desconocido"
    
    cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cropped_face = transform(cropped_face_pil).unsqueeze(0)

    with torch.no_grad():
        output = gender_model(cropped_face)
    
    _, predicted = torch.max(output, 1)
    return "Mujer" if predicted.item() == 1 else "Hombre"

# Función para generar frames de video
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar la detección con YOLOv5
        results = model(frame)

        # Extraer los resultados y filtrar solo las personas detectadas (Clase 0 corresponde a personas)
        person_count = 0
        for idx, label in enumerate(results.xywh[0][:, -1].tolist()):
            if int(label) == 0:  # Clase 0 es la de personas
                # Obtener las coordenadas de la persona detectada
                x1, y1, x2, y2 = map(int, results.xywh[0][idx][:4])

                # Recortar la imagen de la persona detectada
                person_image = frame[y1:y2, x1:x2]

                # Predecir el género de la persona
                gender = predict_gender(person_image)

                # Mostrar el género y el cuadro de la persona detectada
                cv2.putText(frame, f"Genero: {gender}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar la caja alrededor de la persona
                person_count += 1

        # Solo mostrar las personas detectadas (sin otros objetos)
        cv2.putText(frame, f"Personas detectadas: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Convertir la imagen a formato JPEG para enviarla al cliente
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Cargar la página principal

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)