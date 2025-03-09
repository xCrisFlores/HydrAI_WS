from datetime import datetime
import json
import numpy as np
from flask import Flask
from flask_sock import Sock
from flask_cors import CORS
import requests
from ML_Modules.Regression import Regression
import threading
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
sock = Sock(app)

API_URL = "http://IP:5050/api/consumption"  #Cambia la palabra IP con la IP de tu maquina
reg = Regression()
X_train, y_train = [], []


total_time_active = 0
total_consumption = 0

user_id = None
sensor_id = None

time_active_api = 0
consumption_api = 0
lock = threading.Lock()

active_connections = []
connection_lock = threading.Lock()
latest_data = None

def save_data(API_URL, formatted_data):
    try:
        response = requests.post(API_URL, json=formatted_data, timeout=5)
        response.raise_for_status()
        print(f"Datos enviados a la API: {formatted_data}")
    except Exception as e:
        print(f"Error al enviar datos a la API: {e}")

def send_data_to_api_periodically():
    global time_active_api, consumption_api, user_id, sensor_id

    while True:
        time.sleep(5)

        with lock:
            if time_active_api > 0 or consumption_api > 0:
                if user_id is not None and sensor_id is not None:

                    date_actual = datetime.now()
                    date_api = date_actual.strftime("%Y-%m-%dT%H:%M:%S")

                    data_to_send = {
                        "user_id": user_id,
                        "sensor_id": sensor_id,
                        "consumption": consumption_api,
                        "time_active": time_active_api,
                        "date": date_api
                    }
                    print(f"Enviando datos a API: {data_to_send}")
                    save_data(API_URL, data_to_send)

                    time_active_api = 0
                    consumption_api = 0
                else:
                    print("Error, no se han recibido todos los datos necesarios")

def broadcast_data(data):
    """Broadcast que envia datos a todos los clientes conectados"""
    global active_connections
    to_remove = []

    with connection_lock:
        for ws in active_connections:
            try:
                ws.send(json.dumps(data))
                print(f"Se enviaron los datos")
            except Exception as e:
                print(f"Error al enviar datos: {str(e)}")
                to_remove.append(ws)

        for ws in to_remove:
            if ws in active_connections:
                active_connections.remove(ws)
                print(f"Cliente desconectado, conexiones activas: {len(active_connections)}")

@sock.route('/ws')
def websocket(ws):
    global X_train, y_train, latest_data, active_connections
    global total_time_active, total_consumption, time_active_api, consumption_api, user_id, sensor_id

    with connection_lock:
        active_connections.append(ws)
        print(f"Nuevo cliente conectado, conexiones activas:{len(active_connections)}")

    if latest_data:
        try:
            ws.send(json.dumps(latest_data))
        except Exception as e:
            print(f"Error al enviar datos iniciales: {str(e)}")

    try:
        while True:
            data = ws.receive()
            if not data:
                break

            try:
                formatted_data = json.loads(data)
                print(f"Datos recibidos: {formatted_data}")

                if "ping" in formatted_data:
                    print("Ping recibido de React", formatted_data)
                    ws.send(json.dumps({"status": "ok", "is_response": True}))
                    continue

                if "source" in formatted_data and formatted_data["source"] == "react_app":
                    if "time_active" in formatted_data:
                        tiempo_react = formatted_data["time_active"]
                        try:
                            predicted_consumption = float(reg.predict([[tiempo_react]])[0])
                            response_data = {
                                "status": "ok",
                                "predicted_consumption": predicted_consumption,
                                "timestamp": int(time.time() * 1000),
                                "is_response": True
                            }
                
                            print(f"Prediccion para {tiempo_react} s: {predicted_consumption}")
                            ws.send(json.dumps(response_data))
                        except Exception as e:
                            print(f"Error en la prediccion: {str(e)}")
                            ws.send(json.dumps({"status": "error", "message": "Error en la prediccion"}))


                    continue

                if "time_active" in formatted_data and "consumption" in formatted_data and "user_id" in formatted_data and "sensor_id" in formatted_data:
                    time_active = formatted_data["time_active"]
                    consumption = formatted_data["consumption"]

                    with lock:
                        time_active_api += time_active
                        consumption_api += consumption
                        total_time_active += time_active
                        total_consumption += consumption
                        user_id = formatted_data["user_id"]
                        sensor_id = formatted_data["sensor_id"]

                    X_train.append([total_time_active])
                    y_train.append(total_consumption)

                    if len(X_train) >= 2:
                        try:
                            reg.train(np.array(X_train), np.array(y_train))
                        except Exception as e:
                            print(f"Error entrenando el modelo: {str(e)}")

                    latest_data = formatted_data
                    broadcast_data(formatted_data)
                else:
                    ws.send(json.dumps({"status": "error", "message": "Formato de datos no reconocido", "is_response": True}))

            except json.JSONDecodeError as e:
                ws.send(json.dumps({"error": f"Error en formato JSON: {str(e)}"}))

    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        with connection_lock:
            if ws in active_connections:
                active_connections.remove(ws)


if __name__ == "__main__":
    threading.Thread(target=send_data_to_api_periodically, daemon=True).start()  
    print("Servidor WebSocket iniciado en ws://0.0.0.0:5000/ws")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)