import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def detect_and_draw_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    rgb_face = None
    for (x, y, w, h) in faces:
        x -= 5
        y -= 35
        w += 5
        h += 55
        
        x = max(x, 0)
        y = max(y, 0)
        w = max(w, 1)
        h = max(h, 1)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (37, 50))
        rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb_face = rgb_face.astype('float32') / 255.0
    if rgb_face is None:
        return None, frame
    return rgb_face, frame

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    face, frame_with_face = detect_and_draw_face(frame)
    
    cv2.imshow('Face Detection', frame_with_face)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def extract_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        print("No face found")
        return None
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    resized = cv2.resize(face, (37, 50))
    rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb_face = rgb_face.astype('float32') / 255.0
    return rgb_face