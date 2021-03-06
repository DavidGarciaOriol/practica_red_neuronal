import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Red neuronal sencilla que se encarga de aprender
# a transformar grados Celsius (ºC) a Fahrenheit (ºF).
# Es necesario instalar Tensorflow & Matplotlib para su correcto funcionamiento.


## INPUT & OUTPUT ##

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype = float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype = float)

## VARIABLES DE LA RED ##

capa = tf.keras.layers.Dense(units=1, input_shape=[1])

modelo = tf.keras.Sequential([capa])
modelo.compile(optimizer = tf.keras.optimizers.Adam(0.1), loss="mean_squared_error")

## FUNCIONES DE LA RED ##

def entrenar_red():
    print("Comenzando entrenamiento...")
    historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
    print("¡Modelo entrenado!")

    # MOSTRAR GRÁFICO HISTORIAL #

    plt.xlabel("# Época")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(historial.history["loss"])

def hacer_prediccion(celsius):
    print("Predicción: ")
    resultado = modelo.predict([celsius])
    print(f"Resultado: {str(resultado)} fahrenheit.")

def mostrar_variables_modelo():
    print("Variables internas del modelo...")
    print(capa.get_weights())


## EJECUTAR FUNCIONES ##

entrenar_red()

hacer_prediccion(100)

mostrar_variables_modelo()