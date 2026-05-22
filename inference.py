import cv2
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("models/best.pt")

# Load video
video_path = "test_videos/traffic.mp4"

cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save output video
out = cv2.VideoWriter(
    "outputs/output_video.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Save frame
    out.write(annotated_frame)

    # Show frame
    cv2.imshow("Traffic Detection", annotated_frame)

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Detection completed!")