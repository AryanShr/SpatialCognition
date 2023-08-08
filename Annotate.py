import cv2
import sys
def select_objects(video_path,start_time=0):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    video.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    bboxes = cv2.selectROIs("Select Objects to Track", frame, fromCenter=False, showCrosshair=True)

    return bboxes
def track_multiple_objects(video_path, initial_bboxes, start_time = 0):
    tracker_type = "MultiTracker"  # Tracker type can be: BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT
    multi_tracker = cv2.legacy.MultiTracker_create()
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    video.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    for bbox in initial_bboxes:
        multi_tracker.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update multiple trackers
        ok, bboxes = multi_tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding boxes for each tracked object
        for i, bbox in enumerate(bboxes):
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Tracking failure
                cv2.putText(frame, f"Object {i+1} tracking failure", (100, 80 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


if __name__ == "__main__":
    # Specify the video path and the initial bounding box coordinates (x, y, width, height) for each object
    video_path = 'Dataset/01_noglove_cam2_720.mp4'
    start_time = 2 * 60 + 2
    initial_bboxes = select_objects(video_path, start_time)

    # Call the function to track multiple objects
    track_multiple_objects(video_path, initial_bboxes, start_time)
