import cv2
import os

path = os.path.dirname(os.path.abspath(__file__))
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(path + r'\classifier\face.xml')
user_id = input("Enter your name for the dataset: ")

frame_count = 0
offset = 50
breaktru = False

while True:
    ret, frame = cam.read()
    faces = detector.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(128, 128), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        filename = f"dataset/{user_id}_{frame_count:02d}.jpg"
        face_roi = frame[max(y-offset, 0):min(y+h+offset, frame.shape[0]), max(x-offset, 0):min(x+w+offset, frame.shape[1])]
        resized_face = cv2.resize(face_roi, (128, 128), interpolation=cv2.INTER_AREA)
        cv2.imwrite(filename, resized_face)
        cv2.rectangle(frame, (x-offset, y-offset), (x+w+offset, y+h+offset), (225, 0, 0), 2)
        cv2.imshow('im', frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        if frame_count > 18:
            cam.release()
            cv2.destroyAllWindows()

            print(f"Captured {frame_count} images for {user_id}")
            breaktru = True
            break

        frame_count += 1

    if breaktru:
        break
