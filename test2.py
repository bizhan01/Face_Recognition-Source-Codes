import threading
import cv2
from deepface import DeepFace


def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


cap = cv2.VideoCapture('face.mp4')
# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the delay between frames
delay = int(1000 / (fps / 4))  # Delay for half the original speed

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

counter = 0
face_match = False
reference_img = cv2.imread('reference.jpg')

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()

    counter += 1

    if face_match:
        cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("video", frame)
    key = cv2.waitKey(1)

    # Introduce a delay between frames
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()