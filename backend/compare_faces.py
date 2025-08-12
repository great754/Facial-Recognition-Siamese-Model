import extract_face
import numpy as np
import cv2
import compare
import torch
import resnet_compare
import matplotlib.pyplot as plt

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


def compare_resnet_faces(face1_path, face2_path):
    face1 = extract_face.extract_face(face1_path)
    face2 = extract_face.extract_face(face2_path)

    if face1 is None or face2 is None:
        return "One or both images do not contain a detectable face."

    # Flatten the images to compare them
    # face1_flat = face1.flatten()
    # face2_flat = face2.flatten()

    # n_features = face1_flat.shape[0]
    
    # face1_tensor = torch.tensor(face1, dtype=torch.float32).unsqueeze(0)
    # face2_tensor = torch.tensor(face2, dtype=torch.float32).unsqueeze(0)
    distance = resnet_compare.same_person(face1, face2)
    
    distance = distance.item()

    # _, axes = plt.subplots(1, 2, figsize=(8, 4))
    # axes[0].imshow(face1)
    # axes[0].axis('off')
    # axes[1].imshow(face2)
    # axes[1].axis('off')
    # plt.tight_layout()
    # plt.show()
    return f"The resnet distance between the two faces is: {distance}"

print(compare_faces(r"C:\Users\Great.Abhieyighan\Projects\FaceSiamese\backend\DAH_7409.jpg",
                    r"C:\Users\Great.Abhieyighan\Projects\FaceSiamese\backend\IMG_4770 (1).jpg"))

# print(compare_resnet_faces(r"C:\Users\Great.Abhieyighan\Projects\FaceSiamese\backend\DAH_7409.jpg",
                    # r"C:\Users\Great.Abhieyighan\Projects\FaceSiamese\backend\IMG_4770 (1).jpg"))