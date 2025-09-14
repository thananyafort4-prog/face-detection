import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # ปิด Warning

import cv2
import face_recognition
import os

# --- โหลดฐานข้อมูลบุคคลจากโฟลเดอร์ ---
known_encodings = []
known_names = []
known_images = {}   # เก็บรูปโปรไฟล์

folder_path = r"d:\Work\data\known_faces"  # path ไปยังโฟลเดอร์รูปบุคคล

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)

            # เก็บรูปโปรไฟล์เป็น BGR (ใช้แสดงผล)
            profile_img = cv2.imread(path)
            profile_img = cv2.resize(profile_img, (150, 150))  # ย่อให้เล็กลง
            known_images[name] = profile_img

            print(f"[INFO] Loaded {name}")

# --- เปิดกล้อง ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1].astype("uint8")
    face_locations = face_recognition.face_locations(rgb_frame)

    face_encodings = []
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

            # ถ้ามีรูป profile → แสดง
            if name in known_images:
                cv2.imshow("Profile", known_images[name])

        # วาดกรอบ + ชื่อบนกล้อง
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # แสดงภาพจากกล้อง
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
