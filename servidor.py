from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)
datos_recibidos = []

@app.route('/detectar', methods=['POST'])
def detectar_persona():
    data = request.json
    print("📥 Datos recibidos:", data)
    datos_recibidos.append(data)
    return jsonify({"estado": "ok"})

@app.route('/ver', methods=['GET'])
def ver_datos():
    return jsonify(datos_recibidos)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
