# https://www.youtube.com/watch?v=MkcUgPhOlP8&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K&index=28
import cv2
import numpy as np
from datetime import datetime
import time
import argparse

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()

# Do not show image to user
parser.add_argument("--view", action='store_true',required=False, help="supress image to user")
parser.add_argument("-o", "--output_dir", required=False, help="path to output location")
# Specify the source is a video
parser.add_argument("-src", "--source", required=False, help="path and name to input video")

args = parser.parse_args()

# Set data directory path
DATA_DIR = "./data/"
if args.output_dir:
    DATA_DIR = args.output_dir   
out_filename = f"{DATA_DIR}contour-{datetime.now()}.avi"


# Is source video specifed use instead of writing a video
if args.source:
    cap = cv2.VideoCapture(args.source)
else:   
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # width
    cap.set(4, 720)  # height

# Find the  frame height and width    
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
# Set up the output file
fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
out = cv2.VideoWriter(out_filename, fourcc, 5.0, (1280, 720))

# Print the frame information and output filename
print(f"Souce: wxh: {frame_width} x {frame_height} ")
print(f"writing: {out_filename}")


while cap.isOpened():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # t0 = time.time()
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # t1 = time.time()
    # print(t1-t0)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame1,
            "Status: {}".format("Movement"),
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    cv2.putText(
        frame1,
        "Time: {}".format(str(datetime.now())),
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )
    image = cv2.resize(frame1, (1280, 720))
    out.write(image)
    if args.view:
        cv2.imshow("feed", frame1)

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()
