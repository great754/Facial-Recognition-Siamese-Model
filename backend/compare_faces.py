import extract_face
import numpy as np
import cv2
import compare
import torch
# import resnet_compare
import matplotlib.pyplot as plt
import time

def compare_faces(face1_path, face2_path):
    face1 = extract_face.extract_face(face1_path)
    face2 = extract_face.extract_face(face2_path)

    if face1 is None or face2 is None:
        return "One or both images do not contain a detectable face."

    # Flatten the images to compare them
    face1_flat = face1.flatten()
    face2_flat = face2.flatten()

    n_features = face1_flat.shape[0]
    
    face1_tensor = torch.tensor(face1_flat, dtype=torch.float32).unsqueeze(0)
    face2_tensor = torch.tensor(face2_flat, dtype=torch.float32).unsqueeze(0)
    distance = compare.same_person(face1_tensor, face2_tensor)
    
    distance = distance.item()

    # _, axes = plt.subplots(1, 2, figsize=(8, 4))
    # axes[0].imshow(face1)
    # axes[0].axis('off')
    # axes[1].imshow(face2)
    # axes[1].axis('off')
    # plt.tight_layout()
    # plt.show()
    return f"The distance between the two faces is: {distance}"


# def compare_resnet_faces(face1_path, face2_path):
#     face1 = extract_face.extract_face(face1_path)
#     face2 = extract_face.extract_face(face2_path)

#     if face1 is None or face2 is None:
#         return "One or both images do not contain a detectable face."

#     distance = resnet_compare.same_person(face1, face2)
    
#     distance = distance.item()

#     # _, axes = plt.subplots(1, 2, figsize=(8, 4))
#     # axes[0].imshow(face1)
#     # axes[0].axis('off')
#     # axes[1].imshow(face2)
#     # axes[1].axis('off')
#     # plt.tight_layout()
#     # plt.show()
#     return f"The resnet distance between the two faces is: {distance}"

# print(compare_faces(r"C:\Users\Great.Abhieyighan\Projects\FaceSiamese\backend\DAH_7409.jpg",
#                     r"C:\Users\Great.Abhieyighan\Projects\FaceSiamese\backend\IMG_4770 (1).jpg"))
# print(compare_resnet_faces(r"C:\Users\Great.Abhieyighan\Projects\FaceSiamese\backend\DAH_7409.jpg",
                    # r"C:\Users\Great.Abhieyighan\Projects\FaceSiamese\backend\IMG_4770 (1).jpg"))



def compare_faces_live(face2_path):
    face2 = extract_face.extract_face(face2_path)
    if face2 is None:
        print("The second image does not contain a detectable face.")
        return
    face2_flat = face2.flatten()
    face2_tensor = torch.tensor(face2_flat, dtype=torch.float32).unsqueeze(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    last_inference_time = 0
    inference_interval = 0
    model_busy = False
    latest_distance = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        face1, frame_with_face1 = extract_face.detect_and_draw_face(frame)
        if face1 is not None and not model_busy:
            model_busy = True

            face1_flat = face1.flatten()
            face1_tensor = torch.tensor(face1_flat, dtype=torch.float32).unsqueeze(0)

            distance = compare.same_person(face1_tensor, face2_tensor)
            latest_distance = distance.item()
            last_inference_time = time.time()
            model_busy = False

        if latest_distance is not None:
            confidence = (1 - latest_distance) * 100
            cv2.putText(frame_with_face1, f"Confidence: {confidence:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Confidence', frame_with_face1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


compare_faces_live(r"C:\Users\Great.Abhieyighan\Projects\FaceSiamese\backend\DAH_7409.jpg")