'''
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import cv2
import numpy as np
#import paho.mqtt.client as mqtt
import time
from losantmqtt import Device
import signal
import sys
import argparse

import configparser



config = configparser.RawConfigParser()
config.read('./props.cfg')


MIN_SIZE_RATIO = config.get('MAIN', 'MIN_SIZE_RATIO')
MAX_SIZE_RATIO = config.get('MAIN', 'MAX_SIZE_RATIO')

MIN_ANGLE = config.get('MAIN', 'MIN_ANGLE')
MAX_ANGLE = config.get('MAIN', 'MAX_ANGLE')
MIN_VALUE = config.get('MAIN', 'MIN_VALUE')
MAX_VALUE = config.get('MAIN', 'MAX_VALUE')

LOW_ALARM  = config.get('MAIN', 'LOW_ALARM')
HIGH_ALARM = config.get('MAIN', 'HIGH_ALARM')

IS_LIVE = config.get('MAIN', 'IS_LIVE')
CAMERA_ID = config.get('MAIN', 'CAMERA_ID')

device_id     = config.get('MAIN', 'device_id')
access_key    = config.get('MAIN', 'access_key')
access_secret = config.get('MAIN', 'access_secret')

print( access_key )
print( access_secret )

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r
    

def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    

def calibrate_gauge(img):

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.medianBlur(gray, 5)

    
    ## detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(width*0.25), int(width*0.35))

    # average found circles, found it to be more accurate than trying
    # to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b)

    #draw center and circle
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle
    
    separation = 10.0 #in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval,2))  #set empty arrays
    p2 = np.zeros((interval,2))
    p_text = np.zeros((interval,2))
    for i in range(0,interval):
        for j in range(0,2):
            if (j%2==0):
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

    #add the lines and labels to the image
    for i in range(0,interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)

    #cv2.imwrite('gauge-%s-calibration.%s' % (gauge_number, file_type), img)
    
    min_angle = 40 #input('Min angle (lowest possible angle of dial) - in degrees: ') #the lowest possible angle
    max_angle = 320 #input('Max angle (highest possible angle) - in degrees: ') #highest possible angle
    min_value = 0 # input('Min value: ') #usually zero
    max_value = 60 # input('Max value: ') #maximum reading of the gauge
    units = 'psi'#input('Enter units: ')

    return min_angle, max_angle, min_value, max_value, units, x, y, r

def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r):

    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set threshold and maxValue
    thresh = 70
    maxValue = 255


    # apply thresholding which helps for finding lines
    th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV);

    #dst2 = dst2[y-(r+10):y+(r+10), x-(r+10):x+(r+10)]

    # draw filled circle in white on black background as mask
    mask = np.zeros_like(gray2)
    mask = cv2.circle(mask, (x,y), r-70, (255,255,255), -1)

    # apply mask to image
    dst2 = cv2.bitwise_and(dst2, mask)

    # found Hough Lines generally performs better without Canny / blurring,
    # though there were a couple exceptions where it would only work with Canny / blurring
    dst2 = cv2.medianBlur(dst2, 15)
    dst2 = cv2.Canny(dst2, 50, 250)
    dst2 = cv2.GaussianBlur(dst2, (19, 19), 0)

    # for testing, show image after thresholding
    #cv2.imwrite('gauge-%s-tempdst2.%s' % (gauge_number, file_type), dst2)

    # find lines
    minLineLength = 10
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=0)  
    # rho is set to 3 to detect more lines, easier to get more then filter them out later

    # remove all lines outside a given radius
    final_line_list = []
    #print "radius: %s" %r

    diff1LowerBound = 0.15 #diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    diff1UpperBound = 0.25
    diff2LowerBound = 0.5 #diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    diff2UpperBound = 1.0
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
            diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
            #set diff1 to be the smaller (closest to the center) of the two), makes the math easier
            if (diff1 > diff2):
                temp = diff1
                diff1 = diff2
                diff2 = temp
            # check if line is within an acceptable range
            if (((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r) and (diff2<diff2UpperBound*r)) and (diff2>diff2LowerBound*r)):
                line_length = dist_2_pts(x1, y1, x2, y2)
                # add to final list
                final_line_list.append([x1, y1, x2, y2])

    if len(final_line_list) == 0:
        print( 'No lines found.' )
        return -1

    # assumes the first line is the best one
    x1 = final_line_list[0][0]
    y1 = final_line_list[0][1]
    x2 = final_line_list[0][2]
    y2 = final_line_list[0][3]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #for testing purposes, show the line overlayed on the original image
    #cv2.imwrite('gauge-1-test.jpg', img)
    #cv2.imwrite('gauge-%s-lines-2.%s' % (gauge_number, file_type), img)

    #find the farthest point from the center to be what is used to determine the angle
    dist_pt_0 = dist_2_pts(x, y, x1, y1)
    dist_pt_1 = dist_2_pts(x, y, x2, y2)
    if (dist_pt_0 > dist_pt_1):
        x_angle = x1 - x
        y_angle = y - y1
    else:
        x_angle = x2 - x
        y_angle = y - y2
    # take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    #np.rad2deg(res) #coverts to degrees


    #these were determined by trial and error
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  #in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  #in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  #in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  #in quadrant IV
        final_angle = 270 - res

    #print final_angle

    old_min = float(min_angle)
    old_max = float(max_angle)

    new_min = float(min_value)
    new_max = float(max_value)

    old_value = final_angle

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value

def on_command(device, command):
    print("Command received.")
    cmd = command["name"]
    payload = command["payload"]
    

def signal_handler(signum, frame):
    sys.exit(0)
    print("Shutting Down...")



def main():

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Gauge pressure detection.')
    parser.add_argument('-c', '--camera', dest='camera', type=int, default=-1,
                    help="Use the camera for live aquisition.")

    args = parser.parse_args()
    
    if args.camera <= 0:
        IS_LIVE = False
    else:
        IS_LIVE = True
        CAMERA_ID = int(args.camera)           
 
    camera = None
    img = None
 
    # Read image for calibration
    if IS_LIVE:
    	camera = cv2.VideoCapture( CAMERA_ID )
    	ret, img = camera.read()
    else:
        img = cv2.imread('gauge-1.jpg', cv2.COLOR_BGR2GRAY)
    
    # calibration
    min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(img)
     

    # Construct device
    device = Device(device_id, access_key, access_secret)
    # Listen for commands.
    device.add_event_observer("command", on_command)
    # Connect to Losant.
    device.connect(blocking=False)


    # Send temperature once every second.
    while True:
    
        avg = 0
        
        for i in range(0, 10):
        
            img = None
            
            # get gauge
            if IS_LIVE:
            	ret, img = camera.read()
            else:
            	img = cv2.imread('gauge-1.jpg')
            
            val = get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r)
            avg = avg + val
            time.sleep(1)
        
        print("Current reading: %s %s" %(val, units))
    
        device.loop()
        
        if device.is_connected():
            device.send_state({"gas_pressure": val})



if __name__=='__main__':
    main()
