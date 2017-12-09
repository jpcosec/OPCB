import cv2
import random
import numpy as np

#Se realiza transformacion de espacio de colores
#YCB= cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#preprosecamientos varios
# Y=YCB[:,:,0]
# cv2.imshow("Y",Y)
# YCB[:,:,0] = cv2.medianBlur(Y,3)
# cv2.imshow("3",cv2.cvtColor(YCB, cv2.COLOR_YCrCb2BGR))
# YCB[:,:,0] = cv2.medianBlur(Y,5)
# cv2.imshow("5",cv2.cvtColor(YCB, cv2.COLOR_YCrCb2BGR))
# th2 = cv2.adaptiveThreshold(YCB[:,:,0],255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(YCB[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\

# def get_cont(img):
#     contoursmin=0
#     contoursmax=999999999999999999
#     thresmin=5
#     thresmax=255
#     while (contoursmax-contoursmin)>5:
#         ret, thresh = cv2.threshold(Y, 127, 255, 0)
#         im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     return contours, thresh
#

import os
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
k=0

"""images=load_images_from_folder("imagenes")
for image in images:
    cv2.imshow(str(k),image)
    k+=1
cv2.waitKey(0)"""

def random_color():
    colors =[[128,0,0],
            [139,0,0],
            [165,42,42],
            [178,34,34],
            [220,20,60],
            [255,0,0],
            [255,99,71],
            [255,127,80],
            [205,92,92],
            [240,128,128],
            [233,150,122],
            [250,128,114],
            [255,160,122],
            [255,69,0],
            [255,140,0],
            [255,165,0],
            [255,215,0],
            [184,134,11],
            [218,165,32],
            [238,232,170],
            [189,183,107],
            [240,230,140],
            [128,128,0],
            [255,255,0],
            [154,205,50],
            [85,107,47],
            [107,142,35],
            [124,252,0],
            [127,255,0],
            [173,255,47],
            [0,100,0],
            [0,128,0],
            [34,139,34],
            [0,255,0],
            [50,205,50],
            [144,238,144],
            [152,251,152],
            [143,188,143],
            [250,154],
            [255,127],
            [46,139,87],
            [102,205,170],
            [60,179,113],
            [32,178,170],
            [47,79,79],
            [128,128],
            [139,139],
            [255,255],
            [255,255],
            [224,255,255],
            [206,209],
            [64,224,208],
            [72,209,204],
            [175,238,238],
            [127,255,212],
            [176,224,230],
            [95,158,160],
            [70,130,180],
            [100,149,237],
            [191,255],
            [30,144,255],
            [173,216,230],
            [135,206,235],
            [135,206,250],
            [25,25,112],
            [0,0,128],
            [0,0,139],
            [0,0,205],
            [0,0,255],
            [65,105,225],
            [138,43,226],
            [75,0,130],
            [72,61,139],
            [106,90,205],
            [123,104,238],
            [147,112,219],
            [139,0,139],
            [148,0,211],
            [153,50,204],
            [186,85,211],
            [128,0,128],
            [216,191,216],
            [221,160,221],
            [238,130,238],
            [255,0,255],
            [218,112,214],
            [199,21,133],
            [219,112,147],
            [255,20,147],
            [255,105,180],
            [255,182,193],
            [255,192,203],
            [250,235,215],
            [245,245,220],
            [255,228,196],
            [255,235,205],
            [245,222,179],
            [255,248,220],
            [255,250,205],
            [250,250,210],
            [255,255,224],
            [139,69,19],
            [160,82,45],
            [210,105,30],
            [205,133,63],
            [244,164,96],
            [222,184,135],
            [210,180,140],
            [188,143,143],
            [255,228,181],
            [255,222,173],
            [255,218,185],
            [255,228,225],
            [255,240,245],
            [250,240,230],
            [253,245,230],
            [255,239,213],
            [255,245,238],
            [245,255,250],
            [112,128,144],
            [119,136,153],
            [176,196,222],
            [230,230,250],
            [255,250,240],
            [240,248,255],
            [248,248,255],
            [240,255,240],
            [255,255,240],
            [240,255,255],
            [255,250,250],
            [0,0,0],
            [105,105,105],
            [128,128,128],
            [169,169,169],
            [192,192,192],
            [211,211,211],
            [220,220,220],
            [245,245,245],
            [255,255,255]]
    return colors[random.randrange(len(colors))]


def getWriteContour(im,bin):
    img=im.copy()
    thresh=bin
    #ret, thresh = cv2.threshold(bin, 0, 200, 0)
    cv2.imshow("mask", thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (random_color()), 2)
    return img


print "obtienendo contornos 1"
img = cv2.imread('PCB_Sample.png')
bin0=cv2.imread('Back.png',0)
#bin0=cv2.threshold(bin0, 127, 255, 0)
thresh=cv2.bitwise_not(bin0)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=4)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == 4] = [255,0,0]
print np.max(markers)

cv2.imshow('watershed', opening)
cv2.waitKey(0)


"""bin0=cv2.imread('mask0.png')[:,:,0]
cont1=getWriteContour(img, bin0)
cv2.imshow("mask0", cont1)
cv2.imwrite("cont0.png", cont1)

bin1=cv2.imread('mask1.png')[:,:,0]
cont2=getWriteContour(img, bin1)
cv2.imshow("mask1", cont2)
cv2.imwrite("cont1.png", cont2)

bin2=cv2.imread('mask2.png')[:,:,0]
cont3=getWriteContour(img, bin2)
cv2.imshow("mask2", cont3)
cv2.imwrite("cont2.png", cont3)

bin3=cv2.imread('mask3.png')[:,:,0]
cont4=getWriteContour(img, bin3)
cv2.imshow("mask3", cont4)
cv2.imwrite("cont3.png", cont4)

mix=bin1+bin2
cont5=getWriteContour(img, mix)
cv2.imshow("mix12", cont5)
cv2.imwrite("cont12.png", cont5)

mix2=bin0+bin3
cont6=getWriteContour(img, mix2)
cv2.imshow("mix03", cont6)
cv2.imwrite("cont03.png", cont6)

mix3=bin1+bin3+bin2
cont7=getWriteContour(img, mix3)
cv2.imshow("mix123", cont7)
cv2.imwrite("cont123.png", cont7)"""

#watershed


cv2.waitKey(0)



