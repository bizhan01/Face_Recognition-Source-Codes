import cv2
import numpy as np

# Load the reference image
reference_image = cv2.imread('reference.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the feature detector (ORB)
orb = cv2.ORB_create()

# Find the keypoints and descriptors of the reference image
ref_keypoints, ref_descriptors = orb.detectAndCompute(reference_image, None)

# Initialize the video capture
cap = cv2.VideoCapture('face.mp4')  # Use 0 for the default camera
# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the delay between frames
delay = int(1000 / (fps / 4))  # Delay for half the original speed

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors of the current frame
    frame_keypoints, frame_descriptors = orb.detectAndCompute(gray_frame, None)

    # Initialize the feature matcher (Brute-Force)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors of the reference image and the current frame
    matches = bf.match(ref_descriptors, frame_descriptors)

    # Sort the matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the top 10 matches
    matched_frame = cv2.drawMatches(reference_image, ref_keypoints, gray_frame, frame_keypoints, matches[:10], None, flags=2)

    # Display the matched frame
    cv2.imshow('Matched Frame', matched_frame)

    # Introduce a delay between frames
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()