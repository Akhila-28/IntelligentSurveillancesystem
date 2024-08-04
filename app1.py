from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
import time
import numpy as np
import imutils
from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist

app = Flask(__name__)

# Create a directory to save uploaded videos
UPLOAD_FOLDER = 'static/uploaded_videos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configuration for detection parameters (config.py should contain these)
class Config:
    MODEL_PATH = 'path_to_your_model_files'
    USE_GPU = True  # Set to True if using GPU
    MIN_DISTANCE = 50  # Minimum distance between centroids
    MAX_DISTANCE = 150  # Maximum distance to consider abnormal
    Threshold = 10  # Threshold for triggering alerts
    ALERT = True  # Set to True to enable alerting

config = Config()

# Function to process uploaded video
def process_video(video_path):
    labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    if config.USE_GPU:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    unconnected_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_layers[0], np.ndarray):
        ln = [ln[i[0] - 1] for i in unconnected_layers]
    else:
        ln = [ln[i - 1] for i in unconnected_layers]

    vs = cv2.VideoCapture(video_path)
    writer = None
    fps = FPS().start()

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

        serious = set()
        abnormal = set()
        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < config.MIN_DISTANCE:
                        serious.add(i)
                        serious.add(j)
                    if (D[i, j] < config.MAX_DISTANCE) and not serious:
                        abnormal.add(i)
                        abnormal.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            startX, startY, endX, endY = bbox
            cX, cY = centroid
            color = (0, 255, 0)
            if i in serious:
                color = (0, 0, 255)
            elif i in abnormal:
                color = (0, 255, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 2)

        Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
        cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
        Threshold = "Threshold limit: {}".format(config.Threshold)
        cv2.putText(frame, Threshold, (470, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
        text = "Total serious violations: {}".format(len(serious))
        cv2.putText(frame, text, (10, frame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
        text1 = "Total abnormal violations: {}".format(len(abnormal))
        cv2.putText(frame, text1, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
        if len(serious) >= config.Threshold:
            cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
            if config.ALERT:
                print("[INFO] Sending mail...")
                # Implement your Mailer class or function here
                # Example: Mailer().send(config.MAIL)
                print("[INFO] Mail sent")

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(video_path.replace(".mp4", "_output.avi"), fourcc, 25, (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)
        fps.update()

    fps.stop()
    print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
    vs.release()
    writer.release()
    cv2.destroyAllWindows()

# Function to detect people in a frame
def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    centroids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))

    idxs = cv2.dnn.NMSBoxes(boxes, confidence, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centroid = centroids[i]
            results.append((confidence, (x, y, x + w, y + h), centroid))

    return results

# Route to render index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video upload
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        process_video(filename)  # Start processing the uploaded video
        return redirect(url_for('index'))

# Route to start webcam processing
@app.route('/start_video')
def start_video():
    # Implement webcam processing logic here using OpenCV
    return "Webcam processing started"

# Route to stream processed video frames
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to generate video frames for streaming
def gen_frames():
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

        for (i, (prob, bbox, centroid)) in enumerate(results):
            startX, startY, endX, endY = bbox
            cX, cY = centroid
            color = (0, 255, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(debug=True)
