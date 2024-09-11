import cv2  # Used for image and video processing.
import pandas as pd  # For handling data in a tabular format.
from ultralytics import YOLO  # Loads and runs the YOLOv8 model to detect objects.
import cvzone  # Used for easily displaying text on frames.
import Tracker_module as T  # Import your Tracker module

model = YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    # This function is triggered when the mouse moves in the window. It prints the coordinates (x, y) of the current mouse position on the frame.
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Initialize video capture
cap = cv2.VideoCapture(
    r"C:\Users\USER\Desktop\uni internship\OBJECT DETECTION AND TRACKING\yolov8-multiple-vehicle-class-main\tf.mp4")

# Check if video capture was successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# A file named coco.txt contains class names (like 'car', 'bus', 'truck').
# It reads this file and stores the class names in a list called class_list.
with open(
        r"C:\Users\USER\Desktop\uni internship\OBJECT DETECTION AND TRACKING\yolov8-multiple-vehicle-class-main\coco.txt",
        "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

count = 0

# Initialize the Tracker class for tracking cars, buses, and trucks.
tracker = T.Tracker()  # For cars
tracker1 = T.Tracker()  # For buses
tracker2 = T.Tracker()  # For trucks

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or end of video")
            break
        count += 1

        # The loop processes each frame of the video.
        # count % 2 != 0: This ensures that only every 2nd frame is processed (to reduce computational load).
        if count % 2 != 0:
            continue

        # Resize frame to reduce computation
        frame = cv2.resize(frame, (640, 360))  # Adjust as needed

        # This runs YOLOv8 on the current frame to detect objects.
        results = model.predict(frame, conf=0.5, iou=0.4)  # Adjust thresholds
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        cars = []
        buses = []
        trucks = []
        for index, row in px.iterrows():
            # The co-ordinates of the bounding boxes
            # For each object detected, the bounding box coordinates (x1, y1, x2, y2) are extracted.
            # Then, based on the class (like car, bus, truck), the bounding box is added to a corresponding list (cars, buses, trucks).

            x1 = int(row[0])  # Bounding box top-left x coordinate
            y1 = int(row[1])  # Bounding box top-left y coordinate
            x2 = int(row[2])  # Bounding box bottom-right x coordinate
            y2 = int(row[3])  # Bounding box bottom-right y coordinate
            d = int(row[5])  # Class index

            if 'car' in class_list[d]:
                cars.append([x1, y1, x2, y2])
            elif 'bus' in class_list[d]:
                buses.append([x1, y1, x2, y2])
            elif 'truck' in class_list[d]:
                trucks.append([x1, y1, x2, y2])

        # Update the trackers with the detected objects
        bbox_idx_cars = tracker.update(cars)
        bbox_idx_buses = tracker1.update(buses)
        bbox_idx_trucks = tracker2.update(trucks)

        # Draw bounding boxes and IDs for cars
        for bbox in bbox_idx_cars:
            x3, y3, x4, y4, id1 = bbox
            cx3 = int(x3 + x4) // 2
            cy3 = int(y3 + y4) // 2

            cv2.circle(frame, (cx3, cy3), 4, (255, 0, 0), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'Car {id1}', (x3, y3), 1, 1)

        # Draw bounding boxes and IDs for buses
        for bbox in bbox_idx_buses:
            x3, y3, x4, y4, id1 = bbox
            cx3 = int(x3 + x4) // 2
            cy3 = int(y3 + y4) // 2

            cv2.circle(frame, (cx3, cy3), 4, (0, 255, 0), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
            cvzone.putTextRect(frame, f'Bus {id1}', (x3, y3), 1, 1)

        # Draw bounding boxes and IDs for trucks
        for bbox in bbox_idx_trucks:
            x3, y3, x4, y4, id1 = bbox
            cx3 = int(x3 + x4) // 2
            cy3 = int(y3 + y4) // 2

            cv2.circle(frame, (cx3, cy3), 4, (255, 255, 0), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
            cvzone.putTextRect(frame, f'Truck {id1}', (x3, y3), 1, 1)

        cv2.imshow("RGB",
                   frame)  # The frame is displayed in the window titled "RGB", and the loop continues until the video ends or the user presses the 'Esc' key to exit.
        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
