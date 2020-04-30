# file: detect_rivet_holes.py
# description: detect_rivet_holes.py provides a method for performing 
#              2D Circle detection on RGB imagery using 2D Hough Transform
#              with OpenCV.
# usage: 
#              from detect_rivet_holes import detect
#              file = '../../rivets.jpg'
#              detect( file )
#
# author: Zack Greenberg
# email: zack.greenberg@clarifai.com
# last edit: 04/30/2020

import cv2
import numpy as np

def detect( fname ):

    """ 
       Detect rivet hole centers in RGB imagery using -
       --Bitwise NOT inversion
       --Gausian Blur
       --Threshold 
       --Mask
       --2D Hough Transform
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
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,0.8,5,
                            param1=50,param2=20,minRadius=1,maxRadius=20)

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
