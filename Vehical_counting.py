import cv2
import numpy as np

# traffic video
cam = cv2.VideoCapture("video.mp4")

min_width_rect = 80  # min width of rectangle
min_hight_rect = 80  # min width of rectangle

# Line for counting
count_line_position = 550

# Initialize Subtractor
algo = cv2.createBackgroundSubtractorMOG2()

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1

    return cx, cy

detect = []
offset = 6
counter = 0

while True:
    # read the frames from video
    successfully_read, frame = cam.read()

    # convert frame into grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscaled_frame, (3, 3), 5)

    # apply on each frame
    img_sub = algo.apply(blur)

    # apply dilation
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))

    # apply kernal
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dialteada = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernal)
    dialteada = cv2.morphologyEx(dialteada, cv2.MORPH_CLOSE, kernal)

    counterShape, h = cv2.findContours(dialteada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw line
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0, 0, 255), 4)

    # draw rectangle
    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_hight_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)

        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if y<(count_line_position + offset) and y>(count_line_position - offset):
                counter += 1
            cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0, 255, 255), 4)
            detect.remove((x, y))

            print("Vehical Counter:" + str(counter))

    cv2.putText(frame, "VEHICAL COUNTER:" + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # show frames
    cv2.imshow("Vehicle Counting System!", frame)
    key = cv2.waitKey(1)

    if key == 113 or key == 81:
        break


# Ending
cv2.destroyAllWindows()
cam.release()
print("Code Completed!")