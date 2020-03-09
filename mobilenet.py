import cv2
import numpy as np
import pafy
"""
url = 'https://youtu.be/u68EWmtKZw0?list=TLPQMDkwMzIwMjCcOgKmuF00yg'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)
"""
cap = cv2.VideoCapture(0)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
while True:
    _,image = cap.read()
    image = cv2.resize(image,(640,640))
    (h, w) = image.shape[:2]
    rows = open('models/MobileNetSSD_deploy.prototxt').read().strip().split('\n')
    
    net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt", "models/MobileNetSSD_deploy.caffemodel")
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007,(300, 300), 127.5)
    
    net.setInput(blob)
    preds = net.forward()
    for i in np.arange(0, preds.shape[2]):
        confidence = preds[0, 0, i, 2]
        if confidence > 0.3:
            idx = int(preds[0, 0, i, 1])
            
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            box = preds[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 1)
    cv2.imshow('MobileSSD',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
