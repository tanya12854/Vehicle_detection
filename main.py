import cv2
import numpy as np
import os

cap = cv2.VideoCapture('video.mp4')

min_width_react = 80
min_height_react = 80
count_line_position = 550
fps = cap.get(cv2.CAP_PROP_FPS)
speed_limit = 100  # km/h

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Create a directory to save screenshots if it doesn't exist
if not os.path.exists('22speeding_vehicles'):
    os.makedirs('22speeding_vehicles')

def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

def calculate_speed(y1, y2, fps):
    distance = 10  # meters
    time_diff = abs(y2 - y1) / fps  # seconds
    if time_diff > 0:
        speed = (distance / time_diff) * 3.6  # km/h
        return speed
    else:
        return 0  # Return 0 if time difference is too small to measure

def capture_vehicle_image(frame, x, y, w, h, speed, vid):
    # Expand the region to capture
    expand_factor = 1.5
    new_w = int(w * expand_factor)
    new_h = int(h * expand_factor)
    new_x = max(0, x - (new_w - w) // 2)
    new_y = max(0, y - (new_h - h) // 2)
    
    # Ensure the expanded region doesn't go out of frame
    new_x = min(new_x, frame.shape[1] - new_w)
    new_y = min(new_y, frame.shape[0] - new_h)
    
    # Capture the expanded region
    vehicle_image = frame[new_y:new_y+new_h, new_x:new_x+new_w].copy()
    
    # Add speed information to the image
    cv2.putText(vehicle_image, f"Speed: {speed:.2f} km/h", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save the image
    cv2.imwrite(f'22speeding_vehicles/vehicle_{vid}_speed_{speed:.2f}.jpg', vehicle_image)

detect = []
vehicles = {}
offset = 6
counter = 0
vehicle_id = 0

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    
    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)
        if not validate_counter:
            continue

        center = center_handle(x, y, w, h)
        detect.append(center)
        
        for vid, (cx, cy) in vehicles.items():
            dist = np.sqrt((center[0] - cx)**2 + (center[1] - cy)**2)
            if dist < 50:  # If the center is close to a known vehicle
                speed = calculate_speed(cy, center[1], fps)
                vehicles[vid] = center
                if speed > 0:
                    cv2.putText(frame1, f"{speed:.2f} km/h", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Check if vehicle is speeding and crossing the line
                if speed > speed_limit and center[1] < (count_line_position + offset) and center[1] > (count_line_position - offset):
                    capture_vehicle_image(frame1, x, y, w, h, speed, vid)
                    cv2.putText(frame1, "SPEEDING", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                break
        else:
            vehicle_id += 1
            vehicles[vehicle_id] = center

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, f"Vehicle {vehicle_id}", (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 244, 0), 2)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        if center[1] < (count_line_position + offset) and center[1] > (count_line_position - offset):
            counter += 1
            cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
            detect.remove(center)
            print("Vehicle Counter:" + str(counter))

    cv2.putText(frame1, "VEHICLE COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    
    cv2.imshow('Video Original', frame1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
