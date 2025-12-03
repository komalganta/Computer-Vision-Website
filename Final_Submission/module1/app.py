from flask import Flask, render_template, request, jsonify
import math

app = Flask(__name__)

# Calibration values for 'rubixcube1.jpg'
fx = 1667.53
fy = 1983.20
cx = 1073.56
cy = 703.16
Z = 34.0  # cm, distance from camera to object

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    p1 = data['p1']
    p2 = data['p2']

    x1, y1 = p1['x'], p1['y']
    x2, y2 = p2['x'], p2['y']

    # Convert image pixel coordinates to world coordinates (at depth Z)
    # Formula: X = (u - cx) * Z / fx
    X1 = (x1 - cx) * Z / fx
    Y1 = (y1 - cy) * Z / fy
    X2 = (x2 - cx) * Z / fx
    Y2 = (y2 - cy) * Z / fy

    dX = X2 - X1
    dY = Y2 - Y1
    
    # Euclidean distance in the real world
    distance = math.sqrt(dX**2 + dY**2)

    print(f"Calculated: P1({x1},{y1}) -> P2({x2},{y2}) | Dist: {distance:.4f} cm", flush=True)

    return jsonify({
        "dX": round(abs(dX), 4),
        "dY": round(abs(dY), 4),
        "distance": round(distance, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)