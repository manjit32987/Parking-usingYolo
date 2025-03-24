import cv2
import numpy as np

# Load the video
video_path = 'parking2.mp4'  # Replace with your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Resize the frame for better visibility
frame = cv2.resize(frame, (1200, 800))
clone = frame.copy()
points = []

def select_parking_slot(event, x, y, flags, param):
    """Callback function to capture mouse clicks and store coordinates."""
    global points, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Mark the selected point
        
        if len(points) == 4:
            cv2.polylines(frame, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            print(f"Parking Slot Coordinates: {points}")
            points = []  # Reset for the next parking slot

    cv2.imshow("Select Parking Slots", frame)

# Create a resizable window
cv2.namedWindow("Select Parking Slots", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select Parking Slots", select_parking_slot)

while True:
    cv2.imshow("Select Parking Slots", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):  # Reset the image
        frame = clone.copy()
    elif key == ord('q'):  # Quit
        break

cv2.destroyAllWindows()
cap.release()
