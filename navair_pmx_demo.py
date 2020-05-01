# file: navair_pmx_demo.py
# description: navair_pmx_demo.py provides methods for performing 
#              rivet hole detection, and edge distance measurement from 
#              RGB imagery using 2D Hough Transform with OpenCV.
# usage: 
#              from navair_pmx_demo import *
#              file = '../../rivets.jpg'
#              circles = detect_rivets( file )
#              edges = get_edge_lines( file )
#
# author: Zack Greenberg
# email: zack.greenberg@clarifai.com
# last edit: 04/30/2020

import cv2
import numpy as np

def detect_rivets( fname ):

    """ 
       Detect rivet hole centers in RGB imagery using -
       --Bitwise NOT inversion
       --Gausian Blur
       --Threshold 
       --Mask
       --2D Hough Transform
       
       Params--
       fname: RGB imagery file name. 
              Contains view of rivet holes to be detected.
              
       returns-- 
       circles: numpy array with dims NUM_DETECTIONS x 3
    """

    # Load image from file
    image = cv2.imread(fname)

    # Perform inversion bitwise NOT to make dark circles bright
    imagem = cv2.bitwise_not(image)

    # Convert to grayscale
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
  
    # Apply Gaussian Kernel
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Intensity threshold, range 190-235 may need tuning
    thresh = cv2.threshold(blurred, 190, 235, cv2.THRESH_BINARY)[1]

    # Apply mask threshold
    res = cv2.bitwise_and(imagem, imagem, mask=thresh)

    # Perform 2D Hough Transform Circle Detection
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,0.5,8,
                               param1=50,param2=15,minRadius=1,maxRadius=10)

    # Draw results on input image
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(imagem,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(imagem,(i[0],i[1]),2,(0,0,255),3)

    # Display Detection results    
    cv2.imshow('detected circles', imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return circles

def get_edge_lines( fname ):
    
    """ 
       Detect edge lines of aircraft panel in RGB imagery using -
       --Canny filter
       --2D Hough Lines Transform
       
       Params--
       fname: RGB imagery file name. 
              Contains view of rivet holes to be detected.
              
       returns-- 
       lines: numpy array with dims NUM_DETECTIONS x 4
    """
        
    # Load RGB rivet imagery
    img = cv2.imread(fname)
         
    # Perform Canny edge detection
    edges = cv2.Canny(img, 150, 400, apertureSize=3)
         
    # Perform 2D Hough Line Detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength=1000, maxLineGap=1000)
       
    for line in lines:
        cv2.line(img, (line.transpose()[0], line.transpose()[1]), 
                      (line.transpose()[2], line.transpose()[3]),
                      (0, 0, 255), 1)
            
    # Display results
    cv2.imshow('detected circles', img)                                           
    cv2.waitKey(0)                                                                   
    cv2.destroyAllWindows()
    
    return lines
