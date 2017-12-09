import cv2
import numpy as np


# todo - Recibir sin mascara
# todo - mejorar el estadistico usado
# todo - dar aviso de error.

def show(tit,im):
    im = cv2.resize(im, (960, 540))
    cv2.imshow("im",im)

def hough_angle(img, minLineLength):

    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, minLineLength)
    ang = []
    for line in lines:
        rho, theta= line[0]
        ang.append(np.mod(int(theta*180/np.pi),90) )
    angprom = np.sum(ang) / len(ang)

    return angprom

def straighten(img, mask, threshold=400):
    #todo meter un threshold para la orientacion
    angle = hough_angle( mask, threshold)
    rows, cols, end = np.shape(img)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    mask=cv2.warpAffine(mask, M, (cols, rows))
    return dst,mask


"""im  = cv2.imread("cvl_pcb_dslr_1/pcb12/rec3-mask.png")
img= cv2.imread("cvl_pcb_dslr_1/pcb12/rec3.jpg")

show("im", im)

dst=straighten(img,im,100)

show("trans",dst)

cv2.waitKey(0)

9. Extreme Points

Extreme Points means topmost, bottommost, rightmost and leftmost points of the object.

leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

Orientation
(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
"""