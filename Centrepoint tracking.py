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
    r"C:\Users\USER\Desktop\uni internship\OBJECT DETECTION AND TRACKING\yolov8-multiple-vehicle-class-main\dashcam.mp4")

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

# Store the center points for each object
path_cars = {}
path_buses = {}
path_trucks = {}

# Max number of center points to display in the path
max_path_length = 10  # Adjust this value for a shorter/longer path

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or end of video")
            break
        count += 1

        # Process every second frame to reduce computation
        if count % 2 != 0:
            continue

        frame = cv2.resize(frame, (640, 360))

        results = model.predict(frame, conf=0.5, iou=0.4)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        cars, buses, trucks = [], [], []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])

            if 'car' in class_list[d]:
                cars.append([x1, y1, x2, y2])
            elif 'bus' in class_list[d]:
                buses.append([x1, y1, x2, y2])
            elif 'truck' in class_list[d]:
                trucks.append([x1, y1, x2, y2])

        bbox_idx_cars = tracker.update(cars)
        bbox_idx_buses = tracker1.update(buses)
        bbox_idx_trucks = tracker2.update(trucks)

        # Function to draw a neon line
        def draw_neon_line(frame, path_points, color=(0, 255, 255), thickness=2):
            for i in range(1, len(path_points)):
                # First, draw a thick, transparent line as the glow effect
                cv2.line(frame, path_points[i-1], path_points[i], (color[0], color[1], color[2], 50), thickness+8)
                cv2.line(frame, path_points[i-1], path_points[i], (color[0], color[1], color[2], 100), thickness+4)
                # Then, draw the main solid neon-colored line
                cv2.line(frame, path_points[i-1], path_points[i], color, thickness)

        # Draw paths for cars first, before the bounding boxes
        for bbox in bbox_idx_cars:
            x3, y3, x4, y4, id1 = bbox
            cx3 = int(x3 + x4) // 2
            cy3 = int(y3 + y4) // 2

            if id1 not in path_cars:
                path_cars[id1] = []

            # Append the current center point to the path and limit path length
            path_cars[id1].append((cx3, cy3))
            if len(path_cars[id1]) > max_path_length:
                path_cars[id1].pop(0)

            # Draw the neon path behind the car
            draw_neon_line(frame, path_cars[id1], (0, 255, 255), 2)

        # Draw paths for buses
        for bbox in bbox_idx_buses:
            x3, y3, x4, y4, id1 = bbox
            cx3 = int(x3 + x4) // 2
            cy3 = int(y3 + y4) // 2

            if id1 not in path_buses:
                path_buses[id1] = []

            path_buses[id1].append((cx3, cy3))
            if len(path_buses[id1]) > max_path_length:
                path_buses[id1].pop(0)

            # Draw the neon path behind the bus
            draw_neon_line(frame, path_buses[id1], (0, 255, 255), 2)

        # Draw paths for trucks
        for bbox in bbox_idx_trucks:
            x3, y3, x4, y4, id1 = bbox
            cx3 = int(x3 + x4) // 2
            cy3 = int(y3 + y4) // 2

            if id1 not in path_trucks:
                path_trucks[id1] = []

            path_trucks[id1].append((cx3, cy3))
            if len(path_trucks[id1]) > max_path_length:
                path_trucks[id1].pop(0)

            # Draw the neon path behind the truck
            draw_neon_line(frame, path_trucks[id1], (0, 255, 255), 2)

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

        # Display the frame
        cv2.imshow("RGB", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()

