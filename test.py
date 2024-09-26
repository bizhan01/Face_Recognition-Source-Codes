import cv2

# Load the pre-trained face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the reference image
reference_image = cv2.imread('reference.jpg')

# Convert the reference image to grayscale
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the reference image
reference_faces = face_cascade.detectMultiScale(reference_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Initialize the video capture
cap = cv2.VideoCapture('face.mp4')  # Use 0 for the default camera

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces in the current frame
    for (x, y, w, h) in faces:
        # Crop the face region from the current frame
        face_roi = gray_frame[y:y + h, x:x + w]

        # Match the face region with the reference image
        result = cv2.matchTemplate(reference_gray, face_roi, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Define a threshold for matching
        match_threshold = 0.8

        if max_val >= match_threshold:
            # Draw a rectangle around the matched face in the current frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with face detection
    cv2.imshow('Face Detection', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()