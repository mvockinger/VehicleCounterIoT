import cv2
import numpy as np
import time
import datetime

from oauth2client.service_account import ServiceAccountCredentials
import gspread

# google api define the scope, import the required credentials and setting up the client
scope = ['https://www.googleapis.com/auth/spreadsheets',
         'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name(
    "vehiclecounter-353114-32fe182c4fc5.json", scope)
client = gspread.authorize(creds)

# opening the connection to the vehiclecounter spreadsheet
vc = client.open("vehiclecounter").worksheet("data")

# Camera
cap = cv2.VideoCapture(1)

# Settings for the min width and height of the detected elements
min_width_rect = 80
min_height_rect = 100

# Countline Position & width in the frame (pos400, width 600)
count_line_position = 530
count_line_start = 1050
count_line_end = 1200

# Directionline Position in the frame
direction_line_position = count_line_position + 10

# Direction Flag (direction is false)
direction = False

# Subtractor
algo = cv2.createBackgroundSubtractorMOG2()
# array of detected elements
detected = []

# Allowable error between pixel
offset = 8
# Counter to visualize the difference between two detected objects
counter = 0

# finding the center points of a rectangle


def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy

# Code, to write into the google spreadsheet


def push_entry_to_cloud():
    # getting the timestamp from now, format it and seperate the date from the time
    ts = datetime.datetime.now()
    timestamp = ts.strftime("%d.%m.%Y (%H:%M:%S)")
    stamp = timestamp.split("(")
    date = stamp[0]
    date = date[:-1]
    time = stamp[1]
    time = time[:-1]

    # building a structure to append into the worksheet
    data = []
    data.append(date)
    data.append(time)

    # checking the global variable direction to find out about the direction the vehicle went
    global direction
    if direction is True:
        print("Vehicle from the garage. " + timestamp)
        data.append("leaving")
    elif direction is False:
        print("Vehicle into the garage. " + timestamp)
        data.append("parking")

    vc.append_row(data)

    # resetting the direction variable
    direction = False
    return


while True:
    # reading the capture to analyze it
    ret, frame1 = cap.read()
    # preparing the greyscale and the blur of the capture
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    # applying on each frame
    img_sub = algo.apply(blur)
    # increases the white region in the detection window to get better results
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # closes the holes inside the foreground objects, or small black points on the object -> leads to better contours
    dilatedata = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilatedata = cv2.morphologyEx(dilatedata, cv2.MORPH_CLOSE, kernel)
    contourShape, h = cv2.findContours(
        dilatedata, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # setting up the countline and the directionline and apply them to the frame
    cv2.line(frame1, (count_line_start, count_line_position),
             (count_line_end, count_line_position), (255, 127, 0), 3)
    cv2.line(frame1, (count_line_start, direction_line_position),
             (count_line_end, direction_line_position), (127, 127, 0), 3)

    # checking the detected objects in the frame
    for(i, c) in enumerate(contourShape):
        (x, y, w, h) = cv2.boundingRect(c)
        # checking the size of the detected elements and validate if they are big enough
        validate_contour = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_contour:
            continue

        # mark detected elements with a bounding rect
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # calling the center_handle function and draw the center point
        center = center_handle(x, y, w, h)
        detected.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        # checking the positions of the valid detected elements to see if they crossed the lines
        for (x, y) in detected:
            # setting the direction-flag if element goes from the garage to the street
            if y < (direction_line_position + offset) and y > (direction_line_position - offset) and x > (count_line_start) and x < (count_line_end):
                direction = True
            # calling the push_entry_to_cloud function and putting the script to sleep for 1 second to avoid multiple counting
            if y < (count_line_position + offset) and y > (count_line_position - offset) and x > (count_line_start) and x < (count_line_end):
                time.sleep(1)
                counter += 1
                push_entry_to_cloud()
            # coloring the detectionline orange if there are any elements detected in the frame
            cv2.line(frame1, (count_line_start, count_line_position),
                     (count_line_end, count_line_position), (0, 127, 255), 3)
            detected.remove((x, y))

            #print("Vehicle Counter:" + str(counter))

    # vehicle counter on the top left of the frame to get an overview of how many vehicles where detected in the session
    cv2.putText(frame1, "VEHICLE COUNTER :" + str(counter),
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # showing the two frames
    cv2.imshow('Detector', dilatedata)
    cv2.imshow('Camera Footage', frame1)

    # ending the script by pressing 'esc'
    if cv2.waitKey(40) == 27:
        break

cv2. destroyAllWindows()
cap.release()
