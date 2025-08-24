from ultralytics import YOLO
import cv2

# โหลดโมเดล
face_model = YOLO("yolov8n-face.pt")   # จับใบหน้า

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับใบหน้า
    face_results = face_model(frame, stream=True)
    faces = []
    for r in face_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            faces.append((x1, y1, x2, y2))
            # วาดกรอบใบหน้า
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # นับจำนวนใบหน้า
    num_faces = len(faces)

    # วาดแท็บนับจำนวนบนมุมภาพ
    tab_text = f"Faces: {num_faces}"
    cv2.rectangle(frame, (10, 10), (200, 50), (0, 0, 0), -1)  # แท็บพื้นหลังดำ
    cv2.putText(frame, tab_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Face Detection with Count", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
