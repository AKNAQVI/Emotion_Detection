import cv2
import os
from deepface import DeepFace
import datetime

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create folder to save photos
if not os.path.exists("captured_faces"):
    os.makedirs("captured_faces")

webcam = cv2.VideoCapture(0)

saved_faces = set()  # To avoid saving same face repeatedly

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Emotion detection using DeepFace
        try:
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # Draw rectangle and show emotion
        color = (0, 255, 0) if emotion == "happy" else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Auto snapshot if happy
        if emotion == "happy":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_faces/smile_{timestamp}.jpg"
            if filename not in saved_faces:
                cv2.imwrite(filename, face_img)
                print(f"📸 Saved: {filename}")
                saved_faces.add(filename)

    cv2.imshow("Face & Emotion Detection", frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
