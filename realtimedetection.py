import cv2
from keras.models import model_from_json, Sequential
import numpy as np

# Load the model architecture
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

# Fix for Sequential class issue in newer Keras versions
model = model_from_json(model_json, custom_objects={'Sequential': Sequential})

# Load the trained weights
model.load_weights("facialemotionmodel.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Use DroidCam (USB) as webcam
# Change index to 0, 1, or 2 depending on your device
cap = cv2.VideoCapture(0)

# Optional: lower resolution for smoother performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        img = extract_features(face_img)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, prediction_label, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

    # Display output
    cv2.imshow("Emotion Detection (DroidCam USB)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
