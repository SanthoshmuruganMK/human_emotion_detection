import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Validate OpenCV installation
if not hasattr(cv2, "CascadeClassifier"):
    module_file = getattr(cv2, "__file__", "unknown")
    raise ImportError(
        f"Imported cv2 from {module_file} but it has no CascadeClassifier. "
        "Install the standard opencv-python package with:\n"
        "    pip install --upgrade opencv-python\n"
        "and make sure there is no local file named cv2.py shadowing the package."
    )

# Load Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load trained emotion detection model
project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, "emotion_model.h5")
classifier = load_model(model_path, compile=False)

# Emotion labels — must match the number of output classes the model was trained on.
# Your train/ folder has 7 classes (alphabetical order as Keras loads them):
#   angry, disgust, fear, happy, neutral, sad, surprise
mapping = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}
# Dynamically match label count to the model's actual output size
_num_classes = classifier.output_shape[-1]
if _num_classes != len(mapping):
    raise ValueError(
        f"Expected a 7-class emotion model, but {model_path} has "
        f"{_num_classes} output classes. Run train_model.py first."
    )
emotion_labels = mapping
model_input_shape = classifier.input_shape
input_height = model_input_shape[1]
input_width = model_input_shape[2]

#input datase
# Create folder
folder_path = "input_dataset/"
os.makedirs(folder_path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    saved_image = False




    for (x, y, w, h) in faces:
        #flipped_frame = cv2.flip(frame, 1)
        #cv2.imshow('Mirrored Camera', flipped_frame)
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48),
                              interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:

            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # Predict emotion
            prediction = classifier.predict(roi, verbose=0)[0]
            pred_idx = int(prediction.argmax())
            label = emotion_labels[pred_idx] if pred_idx < len(emotion_labels) else f"Class_{pred_idx}"
            confidence = round(float(np.max(prediction)) * 100, 2)

            # Display emotion label
            text = f"{label} ({confidence}%)"
            cv2.putText(frame, text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

            # Save only one image for the detected emotion label
            if not saved_image:
                face_img = frame[y:y + h, x:x + w]
                face_filename = os.path.join(folder_path, f"{label}_{confidence}.jpg")
                if not os.path.exists(face_filename):
                    cv2.imwrite(face_filename, face_img)
                    print(f"Saved face image: {face_filename}")
                saved_image = True

    if len(faces) == 0:
        cv2.putText(frame, "No face detected",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)


    # Show output window
    cv2.imshow("AI Emotion Detection System", frame)
    
    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
print(classifier.summary())
print("Input width", input_width)
print("Input height", input_height)
print("Number of classes", _num_classes)
print("Emotion labels:", emotion_labels)
print("Model input shape:", model_input_shape)
print("Model output shape:", classifier.output_shape)
print("accuracy:", classifier.metrics_names)
