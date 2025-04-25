from ultralytics import YOLO
import cv2

import pandas as pd

import datetime

import math

import time



# model_path = 'C:/Users/tinao/OneDrive/Desktop/Keypoint/runs_Rhinos_Left/pose/train/weights/last.pt'
# model_path = 'C:/Users/tinao/OneDrive/Desktop/Keypoint/runs/pose/train12/weights/last.pt'
model_path = 'C:/Users/tinao/OneDrive/Desktop/Keypoint/runs/pose/train8/weights/last.pt'


image_path = './samples/ssat43.jpg'
img = cv2.imread(image_path)



model = YOLO(model_path)


results = model.predict(image_path, conf=0.01)[0]

#results = model(image_path)

plotted_results = results[0].plot()

# save a image with the plotted results
cv2.imwrite("ssat43.jpg", plotted_results)

#for result in results:
    #for keypoint_indx, keypoint in enumerate(result.keypoints.tolist()):
        #cv2.putText(img, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
                    #cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        #boxes = result[0].boxes  # Boxes object for bbox outputs
        #keypoints = result[0].keypoints # Keypoints object for pose outputs
        #probs = result[0].probs  # Class probabilities for classification outputs



# Extracting keypoints

#keypoints_in_pixel = result[0].keypoints


#print(result[0].keypoints)

# Extracting one keypoint location

#xy_0 = keypoints_in_pixel[0]

#print(xy_0)





# Saving in csv file

list = []
for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        cls = int(box.cls[0])
        path = image_path
        class_name = model.names[cls]
        conf = int(box.conf[0]*100)
        bx = box.xywh.tolist()
        today = pd.Timestamp(datetime.date.today())
        time = pd.Timestamp('now')
        #df = pd.DataFrame({path: path, 'class_name': class_name, 'class_id': cls, 'confidence': conf, 'box_coord': bx})
        df = pd.DataFrame({path: image_path, 'Date': today, 'Date and Time': time, 'class_name': class_name, 'class_id': cls, 'confidence': conf, 'box_coord': bx})
        list.append(df)
    df = pd.concat(list)

    #df.to_csv('predict_labels.csv', index=False) # Run for the first time to generate file
    df.to_csv('predict_labels.csv', mode='a', index=False) # The following runs



#filename = 'eight.jpg'
#cv2.imwrite(filename, img)
#cv2.imshow('img', img)
#cv2.waitKey(0)
