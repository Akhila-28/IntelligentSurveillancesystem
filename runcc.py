import sys
import os
import argparse
import imutils
import cv2
import time
import schedule
import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS

# Add mylib to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mylib'))

try:
    from mylib import config, thread
    from mylib.detection import detect_people
    # from mylib.mailer import Mailer
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

#----------------------------Parse req. arguments------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
    help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
    help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
    help="whether or not output frame should be displayed")
args = vars(ap.parse_args())
#------------------------------------------------------------------------------#

# Load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# Load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Check if we are going to use GPU
if config.USE_GPU:
    print("")
    print("[INFO] Looking for GPU")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# If a video path was not supplied, grab a reference to the camera
if not args.get("input", False):
    print("[INFO] Starting the live stream..")
    vs = cv2.VideoCapture(config.url)
    if config.Thread:
        cap = thread.ThreadingClass(config.url)
    time.sleep(2.0)

# Otherwise, grab a reference to the video file
else:
    print("[INFO] Starting the video..")
    vs = cv2.VideoCapture(args["input"])
    if config.Thread:
        cap = thread.ThreadingClass(args["input"])

writer = None
# Start the FPS counter
fps = FPS().start()

# Loop over the frames from the video stream
while True:
    # Read the next frame from the file
    if config.Thread:
        frame = cap.read()
    else:
        (grabbed, frame) = vs.read()
        # If the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

    # Resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    # Initialize the set of indexes that violate the max/min social distance limits
    serious = set()
    abnormal = set()

    # Ensure there are *at least* two people detections (required in order to compute our pairwise distance maps)
    if len(results) >= 2:
        # Extract all centroids from the results and compute the Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # Loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # Check to see if the distance between any two centroid pairs is less than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # Update our violation set with the indexes of the centroid pairs
                    serious.add(i)
                    serious.add(j)
                # Update our abnormal set if the centroid distance is below max distance limit
                if (D[i, j] < config.MAX_DISTANCE) and not serious:
                    abnormal.add(i)
                    abnormal.add(j)

    # Loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # Extract the bounding box and centroid coordinates, then initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # If the index pair exists within the violation/abnormal sets, then update the color
        if i in serious:
            color = (0, 0, 255)
        elif i in abnormal:
            color = (0, 255, 255)  # orange = (0, 165, 255)

        # Draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 2)

    # Draw some of the parameters
    Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
    cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
    Threshold = "Threshold limit: {}".format(config.Threshold)
    cv2.putText(frame, Threshold, (470, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

    # Draw the total number of social distancing violations on the output frame
    text = "Total serious violations: {}".format(len(serious))
    cv2.putText(frame, text, (10, frame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

    text1 = "Total abnormal violations: {}".format(len(abnormal))
    cv2.putText(frame, text1, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

    #------------------------------Alert function----------------------------------#
    if len(serious) >= config.Threshold:
        cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
        if config.ALERT:
            print("")
            print('[INFO] Sending mail...')
            # Mailer().send(config.MAIL)
            print('[INFO] Mail sent')
        # config.ALERT = False
    #------------------------------------------------------------------------------#

    # Check to see if the output frame should be displayed to our screen
    if args["display"] > 0:
        # Show the output frame
        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # If an output video file path has been supplied and the video writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # Initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # If the video writer is not None, write the frame to the output video file
    if writer is not None:
        writer.write(frame)

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("===========================")
print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
print("===========================")

# Release the file pointers
print("[INFO] Cleaning up...")
if not args.get("input", False):
    vs.stop()
else:
    vs.release()

if writer is not None:
    writer.release()

cv2.destroyAllWindows()
