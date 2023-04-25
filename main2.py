import cv2
import numpy as np
import validators as vali
import requests 
import os
import kivy
#kivy.require('1.11.1')  # wersja Kivy
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from concurrent.futures import ThreadPoolExecutor
from kivy.uix.image import Image, CoreImage

class MainApp(App):

    def detectQR(self,frame):
        detect = cv2.QRCodeDetector()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        frame = cv2.filter2D(frame, -1, self.sharpen_filter)
        frame = 255-frame
        value, points, straight_qrcode = detect.detectAndDecode(frame)

        if(value != ""):
            #print(value)
            # cv2.imshow('QR', frame)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            return value, points[0]
        return None, None

    def checkIfPhoto(self, qr_value):
        if(vali.url(qr_value)):
            response = requests.head(qr_value)
            
            if(qr_value[-4:] in self.acceptable_formats or qr_value[-5:] in self.acceptable_formats):
                img_data = requests.get(qr_value).content
                with open('qrphoto.jpg', 'wb') as handler:
                    handler.write(img_data)
                
    def checkIfPhotoAsync(self, qr_value):
        self.executor.submit(self.checkIfPhoto, qr_value)

    def applyQROverlay(self, frame, last_qr_value,  corner1, corner2, counter):
        if(not last_qr_value[-4:] in self.acceptable_formats or last_qr_value[-5:] in self.acceptable_formats):
            self.text_end = lambda start : min(start+10, len(last_qr_value))
            self.text_segment = last_qr_value[self.text_start:self.text_end(self.text_start)]
            
            if(counter%2 ==0 and len(last_qr_value) > 10):
                self.text_start = (self.text_start + 1)%len(last_qr_value)
                print(self.text_segment)
            self.text_position = [corner1[0], corner1[1]-10]
            frame = cv2.putText(frame, self.text_segment, self.text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1 ,cv2.LINE_AA)
        else:
            try:
                self.qrphoto = cv2.imread("qrphoto.jpg")
                qr_region = frame[corner1[1]:corner2[1], corner1[0]:corner2[0]]
                overlay_resized = cv2.resize(self.qrphoto, (qr_region.shape[1], qr_region.shape[0]))
                alpha = 1  # waga grafiki
                beta = 1- alpha  # waga fragmentu klatki z kamery
                overlay = cv2.addWeighted(qr_region, beta, overlay_resized, alpha, 0)
                frame[corner1[1]:corner2[1], corner1[0]:corner2[0]] = overlay
            except:
                pass
        frame = cv2.rectangle(frame,corner1, corner2  ,(255, 0, 255), 2)

        return frame





    def gen(self, frame):
        self.counter = (self.counter +1)%10
        value, points= self.detectQR(frame)
        if(points is None and self.qrdelay>0) :
            self.qrdelay = (self.qrdelay-1)
        elif (self.qrdelay == 0 ):
            self.qrdelay = 10
            self.qr_points = None
            self.qr_value = None
        else: 
            self.qrdelay =  10
            self.qr_points = points
            self.qr_value = value
            
        
        if self.qr_points is not None:

            corner1= (self.qr_points[0].astype(int)) -5
            corner2 = (self.qr_points[2].astype(int)) +5
            if(self.qr_value != self.last_qr_value):
                self.last_qr_value = self.qr_value
                self.text_start=0    

                self.checkIfPhotoAsync(self.last_qr_value)
                print(self.last_qr_value)

            frame = self.applyQROverlay(frame, self.last_qr_value, corner1, corner2, self.counter)


        cropped_frame = cv2.resize(frame, (self.new_height-self.frame_offset, self.new_height-self.frame_offset), interpolation=cv2.INTER_AREA)
        
        vr_feed = np.full((self.new_height, self.new_height,4), 0)

        vr_feed[self.frame_offset//2: self.frame_offset//2+cropped_frame.shape[0], self.frame_offset//2:self.frame_offset//2+cropped_frame.shape[1]] = cropped_frame
        
        vr_feed = np.hstack((vr_feed, vr_feed))


        return vr_feed


    def build(self):


        self.acceptable_formats=(".JPG", ".jpg", ".PNG", ".png", ".JPEG", ".jpeg")
        self.fps = 30.0  # Przyjmujemy 30 klatek na sekundę jako standard
        self.frame_size = (1280, 720)  # Przyjmujemy rozmiar klatki 640x480 pikseli

        self.new_height = self.frame_size[1]

        #wyostrzanie do qr
        self.sharpen_filter = np.array([[0,-1, 0],[-1,5,-1],[0,-1,0]])
        self.alpha = 2.2
        self.beta = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
        self.counter=0
        self.qrdelay= 10
        self.points = None
        self.qr_points = None
        self.qr_value = None
        self.last_qr_value = None
        self.text_segment = None
        self.text_start=0
        self.frame_offset = 50
        self.capture = Camera(resolution=((640, 480)), play=True)
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.dispalyMe=Image()

        Clock.schedule_interval(self.update, 1 / self.fps)
        return self.dispalyMe

    def update(self, dt):

        texture = self.capture.texture
        frame = np.frombuffer(texture.pixels, dtype=np.uint8).reshape(texture.height, texture.width, 4)


        frame = self.gen(frame)


        frame = frame.astype(np.uint8)
        # Wyświetl przetworzony obraz
        
        texture = Texture.create(size=(self.new_height*2, self.new_height), colorfmt='rgba')
        texture.blit_buffer(frame.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.dispalyMe.texture = texture

if __name__ == '__main__':
    app = MainApp()
    app.run()