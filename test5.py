import cv2
import numpy as np

# Load the reference image
reference_image = cv2.imread('lady.jpg', cv2.IMREAD_COLOR)

# Convert the reference image to grayscale
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Initialize the feature detector (ORB)
orb = cv2.ORB_create()

# Find the keypoints and descriptors of the reference image
ref_keypoints, ref_descriptors = orb.detectAndCompute(reference_gray, None)

# Initialize the video capture
cap = cv2.VideoCapture('face.mp4')  # Use 0 for the default camera
# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the delay between frames
delay = int(1000 / (fps / 4))  # Delay for half the original speed

# Initialize the FLANN matcher for faster matching
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors of the current frame
    frame_keypoints, frame_descriptors = orb.detectAndCompute(frame_gray, None)

    # Match the descriptors of the reference image and the current frame
    matches = flann.knnMatch(ref_descriptors, frame_descriptors, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    if len(good_matches) > 10:
        # Draw a bounding box around the reference image in the current frame
        ref_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        frame_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = reference_gray.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2)

        # Display the message
        cv2.putText(frame, 'Reference Image Found', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Image Tracking', frame)

    # Introduce a delay between frames
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()