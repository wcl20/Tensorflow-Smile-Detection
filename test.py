import argparse
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", default="weights.hdf5", help="Path to model")
    parser.add_argument("-v", "--video", default=None, help="Path to (optional) video")
    args = parser.parse_args()

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = load_model(args.weights)

    if not args.video:
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(args.video)

    while True:
        success, frame = capture.read()
        if args.video and not success:
            break

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxs = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
        for bbox in bboxs:
            x, y, w, h = bbox
            # Extract region of interest
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.astype("float") / 255.
            roi = np.expand_dims(img_to_array(roi), axis=0)
            # Smile prediction
            not_similing, smiling = model.predict(roi)[0]
            label = "Smiling" if smiling > not_similing else "Not Smiling"
            color = (0, 255, 0) if smiling > not_similing else (0, 0, 255)
            # Draw bbox
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
