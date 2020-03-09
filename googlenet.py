import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:
    _,image = cap.read()
    rows = open('models/synset_words.txt').read().strip().split('\n')
    classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]
    net = cv2.dnn.readNetFromCaffe("models/bvlc_googlenet.prototxt", "models/bvlc_googlenet.caffemodel")
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
    
    net.setInput(blob)
    preds = net.forward()
    """
    # Get inference time:
    t, _ = net.getPerfProfile()
    print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))
    """
    indexes = np.argsort(preds[0])[::-1][:10]

    # We draw on the image the class and probability associated with the top prediction:
    text = "label: {}\nprobability: {:.2f}%".format(classes[indexes[0]], preds[0][indexes[0]] * 100)
    y0, dy = 30, 30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    #print(indexes)
    # Print top 10 prediction:
    """
    for (index, idx) in enumerate(indexes):
        print("{}. label: {}, probability: {:.10}".format(index + 1, classes[idx], preds[0][idx]))
    """
    cv2.imshow('googlenet',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break
cap.release()
cv2.destroyAllWindows()

