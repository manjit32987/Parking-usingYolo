import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8s.pt')

# Read class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().strip().split("\n")

# Define parking areas as a list of polygons
parking_areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],
    [(105, 353), (86, 428), (137, 427), (146, 358)],
    [(159, 354), (150, 427), (204, 425), (203, 353)],
    [(217, 352), (219, 422), (273, 418), (261, 347)],
    [(274, 345), (286, 417), (338, 415), (321, 345)],
    [(336, 343), (357, 410), (409, 408), (382, 340)],
    [(396, 338), (426, 404), (479, 399), (439, 334)],
    [(458, 333), (494, 397), (543, 390), (495, 330)],
    [(511, 327), (557, 388), (603, 383), (549, 324)],
    [(564, 323), (615, 381), (654, 372), (596, 315)],
    [(616, 316), (666, 369), (703, 363), (642, 312)],
    [(674, 311), (730, 360), (764, 355), (707, 308)]
]

# Initialize video capture
cap = cv2.VideoCapture('parking1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLO object detection
    results = model.predict(frame)
    detections = results[0].boxes.data
    detections_df = pd.DataFrame(detections).astype("float")

    # Track cars in each parking spot
    occupied_spots = [0] * len(parking_areas)

    for _, row in detections_df.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row)
        class_name = class_list[class_id]

        if 'car' in class_name:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            for i, area in enumerate(parking_areas):
                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    occupied_spots[i] = 1  # Mark this spot as occupied

    # Calculate available and occupied spaces
    total_spaces = len(parking_areas)
    occupied_spaces = sum(occupied_spots)
    free_spaces = total_spaces - occupied_spaces

    # Display available and occupied spaces text
    cv2.putText(frame, f"Free Spaces: {free_spaces}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Occupied Spaces: {occupied_spaces}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw parking area labels and status
    for i, (area, occupied) in enumerate(zip(parking_areas, occupied_spots)):
        color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        cv2.putText(frame, str(i + 1), area[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the frame
    cv2.imshow("Parking Lot", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
