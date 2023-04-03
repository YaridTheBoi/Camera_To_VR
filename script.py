import cv2
import numpy as np
from flask import Flask, Response
import validators as vali
import requests 
import os
from concurrent.futures import ThreadPoolExecutor

acceptable_formats=(".JPG", ".jpg", ".PNG", ".png", ".JPEG", ".jpeg")
fps = 25.0  # Przyjmujemy 30 klatek na sekundę jako standard
frame_size = (1280, 720)  # Przyjmujemy rozmiar klatki 640x480 pikseli

sharpen_filter = np.array([[0,-1, 0],[-1,5,-1],[0,-1,0]])
alpha = 2.2
beta = 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))

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
executor = ThreadPoolExecutor(max_workers=2)

#zamiast filtrow mozna brac te smart qr, i sprawdzac czy sa redirectem i na tym sprawdzac
def detectQR(frame):
    detect = cv2.QRCodeDetector()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    frame = cv2.filter2D(frame, -1, sharpen_filter)
    frame = 255-frame
    value, points, straight_qrcode = detect.detectAndDecode(frame)

    if(value != ""):
        #print(value)
        # cv2.imshow('QR', frame)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return value, points[0]
    return None, None



def detectQRPreview():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # top_hat = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel)
        # black_hat = cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel)
        # opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)


        # frame = frame+ top_hat -black_hat + opening

        

        _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        frame = cv2.filter2D(frame, -1, sharpen_filter)

        frame = 255-frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        


def checkIfPhoto(qr_value):
    if(vali.url(qr_value)):
        response = requests.head(qr_value)
        
        if(qr_value[-4:] in acceptable_formats or qr_value[-5:] in acceptable_formats):
            img_data = requests.get(qr_value).content
            with open('qrphoto.jpg', 'wb') as handler:
                handler.write(img_data)
            
            

def checkIfPhotoAsync(qr_value):
    executor.submit(checkIfPhoto, qr_value)

def gen():
    cap = cv2.VideoCapture(0)
    counter=0
    qrdelay= 10
    points = None
    qr_points = None
    qr_value = None
    last_qr_value = None
    text_segment = None
    text_start=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        counter = (counter +1)%10
        value, points= detectQR(frame)
        if(points is None and qrdelay>0) :
            qrdelay = (qrdelay-1)
        elif (qrdelay == 0 ):
            qrdelay = 10
            qr_points = None
            qr_value = None
        else: 
            qrdelay =  10
            qr_points = points
            qr_value = value
            
        
        if qr_points is not None:

            corner1= (qr_points[0].astype(int)) -5
            corner2 = (qr_points[2].astype(int)) +5
            if(qr_value != last_qr_value):
                last_qr_value = qr_value
                text_start=0    

                checkIfPhotoAsync(last_qr_value)
                print(last_qr_value)

            if(not last_qr_value[-4:] in acceptable_formats or last_qr_value[-5:] in acceptable_formats):
                text_end = lambda start : min(start+10, len(last_qr_value))
                text_segment = last_qr_value[text_start:text_end(text_start)]
                
                if(counter%2 ==0 and len(last_qr_value) > 10):
                    text_start = (text_start + 1)%len(last_qr_value)
                    print(text_segment)
                text_position = [corner1[0], corner1[1]-10]
                frame = cv2.putText(frame, text_segment, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1 ,cv2.LINE_AA)
            else:
                try:
                    
                    qrphoto = cv2.imread("qrphoto.jpg")
                    qr_region = frame[corner1[1]:corner2[1], corner1[0]:corner2[0]]
                    overlay_resized = cv2.resize(qrphoto, (qr_region.shape[1], qr_region.shape[0]))
                    alpha = 1  # waga grafiki
                    beta = 1- alpha  # waga fragmentu klatki z kamery
                    overlay = cv2.addWeighted(qr_region, beta, overlay_resized, alpha, 0)
                    frame[corner1[1]:corner2[1], corner1[0]:corner2[0]] = overlay
                except:
                    pass
            frame = cv2.rectangle(frame,corner1, corner2  ,(255, 0, 255), 2)



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

@app.route('/qr-vision')
def qrVision():
    return Response(detectQRPreview(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)