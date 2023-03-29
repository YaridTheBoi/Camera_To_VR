import cv2
import numpy as np
from flask import Flask, Response

fps = 25.0  # Przyjmujemy 30 klatek na sekundę jako standard
frame_size = (1280, 720)  # Przyjmujemy rozmiar klatki 640x480 pikseli

new_height = frame_size[1]


center = (new_height // 2, new_height // 2)


radius = min(center[0], center[1])
strength = 0.8


map_x, map_y = np.meshgrid(np.arange(new_height), np.arange(new_height))

map_x = map_x.astype(np.float32)
map_y = map_y.astype(np.float32)

map_x -= center[0]
map_y -= center[1]

dist = np.sqrt(map_x ** 2 + map_y ** 2)

theta = np.arctan2(map_y, map_x)

new_dist = dist ** strength * radius / (radius - dist * strength)

new_x = new_dist * np.cos(theta) + center[0]
new_y = new_dist * np.sin(theta) + center[1]

map_x = new_x
map_y = new_y


app = Flask(__name__)


def detectQR(frame):
    detect = cv2.QRCodeDetector()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    value, points, straight_qrcode = detect.detectAndDecode(frame)
    if(value != ""):
        print(value)
        #print(points[0])
        #print("\n")
        return(points[0])
    return(None)

def gen():
    cap = cv2.VideoCapture(0)
    counter=0
    points = None

    while cap.isOpened():
        ret, frame = cap.read()
        #points= detectQR(frame)
        if not ret:
            break
            
        counter = (counter +1)%10
        
        points= detectQR(frame)

            
        if points is not None:

            corner1= (points[0].astype(int)) -5
            corner2 = (points[2].astype(int)) +5



            frame = cv2.rectangle(frame,corner1, corner2  ,(255, 0, 255), 2)
            #print(corner1, corner2)
        cropped_frame = cv2.resize(frame, (new_height, new_height), interpolation=cv2.INTER_AREA)
        copy = cropped_frame.copy()

        fisheye1 = cv2.remap(cropped_frame, map_x, map_y, cv2.INTER_LINEAR)
        fisheye2 = cv2.remap(copy, map_x, map_y, cv2.INTER_LINEAR)

        resize_copy = cv2.hconcat([fisheye1, fisheye2])

        ret, jpeg = cv2.imencode('.jpg', resize_copy)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)