#!/usr/bin/env python
'''
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''

import sys
import cv2
import numpy as np

"""if __name__ == '__main__':
# If image path and f/q is not passed as command
# line arguments, quit and display help message
if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)"""


# speed-up using multithreads
cv2.setUseOptimized(True);
cv2.setNumThreads(4);

# read image
#im = cv2.imread(sys.argv[1])
im=cv2.imread("/Users/Jpcoseco/Pictures/imagenes/cvl_pcb_dslr_1/pcb1/rec1.jpg")
#im= cv2.medianBlur(im, 9)

#    cv2.imshow("imagen a buscar"im)
# resize image
#newHeight = 200
#newWidth = int(im.shape[1] * 200 / im.shape[0])
#im = cv2.resize(im, (newWidth, newHeight))

# create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# set input image on which we will run segmentation
ss.setBaseImage(im)

# Switch to fast but low recall Selective Search method
#if (sys.argv[2] == 'f'):
#    ss.switchToSelectiveSearchFast()

# Switch to high recall but slow Selective Search method
#elif (sys.argv[2] == 'q'):
ss.switchToSelectiveSearchQuality()
# if argument is neither f nor q print help message
#else:
#    print(__doc__)
#    sys.exit(1)

# run selective search segmentation on input image


rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))

# number of region proposals to show
numShowRects = 100
# increment to increase/decrease total number
# of reason proposals to be shown
increment = 50
bounding_boxes=[]
while True:
    # create a copy of original image
    imOut = im.copy()

    # itereate over all the region proposals
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            #bounding_boxes.append([x,y,w,h])
        else:
            break

    # show output
    cv2.imshow("Output", imOut)

    # record key press
    k = cv2.waitKey(0) & 0xFF

    print k
    # m is pressed
    if k == 109:
        # increase total number of rectangles to show by increment
        numShowRects += increment
    # l is pressed
    elif k == 108 and numShowRects > increment:
        # decrease total number of rectangles to show by increment
        numShowRects -= increment
    elif k == 113:
        break
    """elif k == ord('s'):
        for i, rect in enumerate(rects):
            print "saving bounding boxes"
            x, y, w, h = rect
            #cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            bounding_boxes.append([x,y,w,h])
    np.save("bb",bounding_boxes)
    print "bounding boxes saved"
    elif k ==0:
        print "up"
    elif k ==1:
        print "down"
    elif k ==2:
        print "left"
    elif k ==3:
        print "right"
    elif k==122:
        print "zoom"""

    # q is pressed

    # close image show window