import cv2
import numpy as np
import os.path as path

input_file=input("Nazwa wejscia: ")
output_file=input("Nazwa wyjscia: ")

if(not path.isfile(input_file)):
    print("Nie ma takiego pliku wejsciowego")
    quit()

cap=cv2.VideoCapture(input_file)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(fps, frame_size)


new_height = frame_size[1]
new_widht = new_height*2

new_framesize = (new_widht, new_height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, new_framesize)


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




# Pętla po ramkach wideo
while cap.isOpened():
    # Odczytaj kolejną ramkę wideo
    ret, frame = cap.read()

    if ret:
        # Skopiuj ramkę
        cropped_frame = cv2.resize(frame, (new_height, new_height), interpolation=cv2.INTER_AREA)
        copy = cropped_frame.copy()

        fisheye1 = cv2.remap(cropped_frame, map_x, map_y, cv2.INTER_LINEAR)
        fisheye2 = cv2.remap(copy, map_x, map_y, cv2.INTER_LINEAR)

        resize_copy = cv2.hconcat([fisheye1, fisheye2])
        # Zapisz skopiowaną ramkę do pliku wyjściowego
        out.write(resize_copy)

    else:
        break

# Zakończ odtwarzanie pliku i zwolnij zasoby
cap.release()
out.release()
