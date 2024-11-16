# Detector de Personas y Predicción de Género

Este proyecto utiliza el modelo YOLOv5 para la detección de personas y un modelo ResNet18 para predecir el género de las personas detectadas en tiempo real a través de la cámara web.

## Requisitos

- Python 3.10
- Dependencias:
  - torch
  - torchvision
  - flask
  - opencv-python
  - Pillow
  - numpy

## Instalación

1. Clona este repositorio en tu máquina local:
   ```bash
   git clone https://github.com/tu-usuario/tu-repo.git
   cd tu-repo
   ```

2. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Windows usa venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

1. Ejecuta el servidor Flask:
   ```bash
   python app.py
   ```

2. Abre tu navegador web y ve a:
   ```
   http://127.0.0.1:5000/
   ```

El video de la cámara web se mostrará en tiempo real con la detección de personas y predicción de género.

## Licencia

Este proyecto está bajo la licencia MIT.