import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def extract_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("No face found")
        return None
    
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]

    return face

# face_img = extract_face(r"C:\Users\Great.Abhieyighan\Projects\FaceSiamese\backend\IMG_4770 (1).jpg")
# if face_img is not None:
#     cv2.imshow("Face", face_img)
#     cv2.waitKey(0)