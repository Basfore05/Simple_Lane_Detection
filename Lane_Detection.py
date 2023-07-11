import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope) # y = mx+c , i.e., x = y-c/m
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    gray= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150) 
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for   x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image



def region_of_interest(image):
    height = image.shape[0]
    # triangle = np.array([(200, height), (1100, height), (550, 250)], dtype=np.int32)
    polygon = np.array([(200, height), (1100, height), (550, 250)], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [polygon], 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#  loading and showing the image
image = cv2.imread('test_image.jpg')


# edge detection - identifying sharp changes in intensity in adjacent pixels (like turnings)
# for finding edges, gradient is used

# ************        Step 1      **************
# convert the image into grayscale, because in grayscale image 1px consists of only one color where intensity ranges from 0-255 but a normal color is a combination of three color channels 

lane_image = np.copy(image)
# gray= cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
  
# *************      Step 2      **************** 

# Gaussian Blur = clarifies the image, image may be blurry which creates barriers in detecting the lanes 

# blur = cv2.GaussianBlur(gray_scale, (5,5), 0)

# 5/5 = at first affect will be applied to the 5/5 area of the the full image after that for next 5/5 and so on

# *************     Step 3        ***************

# Canny Method = traces the outline of the edges that corresponds sharp changes in intensity gradients
# used to show the strongest gradients in our image
#  a function is created where the canny method is used and will return canny which we gonna print
# canny = cv2.Canny(blur, 50, 150) 


# *************     Step 4         ****************

#  plt is used to show our image in x and y directions and here, we just need to pass the result to show
# it creates a triangular path in our image which is our region of interest

# *************     Step 5       upward in the function region of interest        ****************


# *************     Step 6       =  Hough Transform     ***************
#  Hough Transform =    detects the straight lines in our region of interest

# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # used to show our lines in the original color image
# cv2.imshow("Result", combo_image)
# cv2.waitKey(0)


cap = cv2.VideoCapture("test2.mp4")
while cap.isOpened():
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # Use matplotlib.pyplot.imshow() to display the image
    # plt.imshow(cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
    # plt.show()

    # res=cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Lane", combo_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



