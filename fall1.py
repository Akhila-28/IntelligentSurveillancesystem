import cv2
import cvzone
import math
from ultralytics import YOLO

def process_frame(frame, model, classnames):
    # Process the frame with YOLO model
    results = model(frame)

    for result in results:
        parameters = result.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

            if threshold < 0:
                cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2, colorR=(0, 0, 255))
    return frame

def main(input_source):
    # Load the YOLO model
    model = YOLO('yolov8s.pt')

    # Load class names
    classnames = []
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()

    if input_source.isdigit():
        cap = cv2.VideoCapture(int(input_source))  # Use webcam
    else:
        cap = cv2.VideoCapture(input_source)  # Use video file

    # Verify if video capture is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        return

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        frame = cv2.resize(frame, (980, 740))
        frame = process_frame(frame, model, classnames)
        
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    # Load the YOLO model
    model = YOLO('yolov8s.pt')

    # Load class names
    classnames = []
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()

    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (980, 740))
    frame = process_frame(frame, model, classnames)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_type = input("Enter 'webcam', 'video', or 'image': ").strip().lower()
    
    if input_type == 'webcam':
        input_source = '0'
        main(input_source)
    elif input_type == 'video':
        video_path = input("Enter the path to the video file: ").strip()
        main(video_path)
    elif input_type == 'image':
        image_path = input("Enter the path to the image file: ").strip()
        process_image(image_path)
    else:
        print("Invalid input type. Please enter 'webcam', 'video', or 'image'.")