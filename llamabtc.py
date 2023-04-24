# Instalar las librerías necesarias
!pip install yfinance llama mplfinance
!pip install git+https://github.com/facebookresearch/llama.git
!pip install transformers
!pip install sentencepiece
!pip install fairscale

# Importar las librerías necesarias
import torch
import transformers
import pandas as pd
import yfinance as yf
from llama import LLaMA
from google.colab import files
import matplotlib.pyplot as plt
import mplfinance as mpf

# Definir los parámetros del programa
symbol = "BTC-USD" # Símbolo de la acción de bitcoin
period = "2y" # Período de los datos
interval = "5m" # Intervalo de los datos
model_path = "/content/drive/MyDrive/llama.pt" # Ruta del archivo con el modelo pre-entrenado de LLAMA (cambiar según tu ruta en Google Drive)
adapted_model_path = "/content/drive/MyDrive/llama_bitcoin.pt" # Ruta del archivo donde se guardará el modelo adaptado (cambiar según tu ruta en Google Drive)
num_iterations = 10 # Número de iteraciones del meta-aprendizaje
batch_size = 32 # Tamaño del lote para el entrenamiento
learning_rate = 0.01 # Tasa de aprendizaje para el optimizador
test_size = 0.2 # Porcentaje de los datos para el conjunto de prueba

# Descargar los datos con yfinance
data = yf.download(symbol, period="60d", interval=interval)

# Extraer los precios y los volúmenes
prices = data["Close"]
volumes = data["Volume"]

# Resamplear los datos con diferentes intervalos
prices_30m = prices.resample("30T").last()
prices_1h = prices.resample("1H").last()
prices_4h = prices.resample("4H").last()
prices_1d = prices.resample("1D").last()

# Sumar el valor de volumen a todo el data set
data_30m = pd.concat([prices_30m, volumes], axis=1)
data_1h = pd.concat([prices_1h, volumes], axis=1)
data_4h = pd.concat([prices_4h, volumes], axis=1)
data_1d = pd.concat([prices_1d, volumes], axis=1)

# Elegir el data set que se va a usar para la predicción (puedes cambiarlo según tu preferencia)
data_set = data

# Dividir los datos en entrenamiento y prueba
train_size = int(len(data_set) * (1 - test_size))
train_data = data_set[:train_size]
test_data = data_set[train_size:]

# Cargar el modelo de LLAMA desde Google Drive o subirlo desde tu computadora

model = LLaMA.from_pretrained(model_path); model.save(adapted_model_path)

# model = LlamaModel.load(files.upload())

# Adaptar el modelo a la tarea de predicción de bitcoin
model.adapt(train_data, num_iterations, batch_size, learning_rate)

# Guardar el modelo adaptado en Google Drive o descargarlo a tu computadora
model.save(adapted_model_path)
# files.download(adapted_model_path)

# Evaluar el modelo en el conjunto de prueba
metrics = model.evaluate(test_data)
print(metrics)

# Obtener los datos de las últimas 30 velas de 5 minutos del conjunto de prueba y usar el modelo para predecir los precios de las siguientes 30 velas de 5 minutos
last_30_candles = test_data[-30:]
next_30_candles = model.predict(last_30_candles)

# Crear un dataframe con los precios de apertura, cierre, máximo y mínimo de las 60 velas
ohlc_data = pd.DataFrame()
