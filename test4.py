import cv2
import numpy as np

# Load the reference image
reference_image = cv2.imread('reference.jpg', cv2.IMREAD_COLOR)

# Convert the reference image to grayscale
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Initialize the feature detector (ORB)
orb = cv2.ORB_create()

# Find the keypoints and descriptors of the reference image
ref_keypoints, ref_descriptors = orb.detectAndCompute(reference_gray, None)

# Initialize the video capture
cap = cv2.VideoCapture('face.mp4')  # Use 0 for the default camera

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors of the current frame
    frame_keypoints, frame_descriptors = orb.detectAndCompute(frame_gray, None)

    # Initialize the feature matcher (Brute-Force)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors of the reference image and the current frame
    matches = bf.match(ref_descriptors, frame_descriptors)

    # Sort the matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the top match
    if len(matches) > 0:
        best_match = matches[0]
        img_match = cv2.drawMatches(reference_gray, ref_keypoints, frame_gray, frame_keypoints, [best_match], None, flags=2)
        cv2.imshow('Matched Image', img_match)
    else:
        cv2.imshow('Matched Image', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()