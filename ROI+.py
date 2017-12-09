import cv2
import numpy as np
import os.path

# Funcion para encontrar IOU de 2 boxes, obtenida de
#https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle

    ious=[]
    if (np.ndim(boxB)==1):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = abs((xB - xA + 1) * (yB - yA + 1))

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        #print "iou", iou, interArea
        return iou

    else:
        gnum =  np.shape(boxB)[0]
        #print "gnum ",gnum,
        for i in range(gnum):
            #print "boxa", boxA
            #print "boxb",i, boxB[i]
            xA = max(boxA[0], boxB[i][0])
            yA = max(boxA[1], boxB[i][1])
            xB = min(boxA[2], boxB[i][2])
            yB = min(boxA[3], boxB[i][3])



            interArea = abs((xB - xA + 1) * (yB - yA + 1))

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[i][2] - boxB[i][0] + 1) * (boxB[i][3] - boxB[i][1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            div=float(boxAArea + boxBArea - interArea)
            if (div==0):
                iou= 1
            else:
                iou = interArea / div
            #print "iou", iou, interArea
            ious.append(iou)

        # return the intersection over union value
        return ious

def dct_clear(img,n):

    imf = np.float32(img)/255.0 # float conversion/scale
    dst = cv2.dct(imf) # the dct
    for i in range(n):
        for j in range(n):
            if (i+j <n):
                dst[i,j]=0

    dst[0,0]=0
    inv = cv2.idct(dst)
    mi=np.min(inv)
    ma=np.max(inv)
    #print mi,ma
    k=255.0/(ma-mi)
    inv=(inv-mi)*k
    #inv=abs(inv)*255
    #print inv
    return inv.astype(np.uint8)

def get_conected(img):
    #Se obtienen regiones conexas
    a , labels ,  stats, centroids = cv2.connectedComponentsWithStats(img)
    areas=[]
    #se obtiene areas
    for i in np.arange(a):
        areas.append(stats[i][cv2.CC_STAT_AREA])
    #Se obtiene el area de mayor conectividad
    lab_max=np.argmax(areas[1:-1])
    #Se obtiene region de mayor conectividad
    max_connected=np.array(np.equal(labels,lab_max+1)*255,np.uint8)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(max_connected, cv2.MORPH_OPEN, kernel)
    return opening

def purge(res,im,mask,w,h,background_ratio_threshold,prev_uoi_threshold,originals,bounding_boxes,folder,imno,n):
    loc = np.where(res >= match_threshold)
    area = w * h
    #todo - implementar
    p=0
    for pt in zip(*loc[::-1]):
        box_match=[ pt[0],pt[1], pt[0] + w, pt[1] + h]
        #cv2.imshow("imcrop2", im[pt[1]:pt[1] + h, pt[0]:pt[0] + w])
        #cv2.waitKey(0)
        # Se obtiene un ratio de cantidad de fondo en roi
        intersec_back = np.sum(mask[pt[1]:pt[1] + h, pt[0]:pt[0] + w])
        back_ratio = intersec_back / (area)
        # Se calcula el maximo UOI con las cajas traseras
        prev_bbs_uoi=bb_intersection_over_union(box_match, bounding_boxes)
        max_prev_uoi=max(prev_bbs_uoi)

        # Si
        if (back_ratio<background_ratio_threshold and max_prev_uoi<prev_uoi_threshold):
            p += 1
            bounding_boxes.append(box_match)
            cv2.rectangle(im, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
            imCr=originals[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
            cv2.imwrite("match/" + folder + "_" + imno+"_"+str(n) +"_"+str(p) +".png", imCr)
    return im, bounding_boxes


def prepros(im):
    # im = cv2.resize(im, (960, 540))
    im = cv2.bilateralFilter(im, 5, 90, 370)
    # Select ROI
    # Prepros
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    w, h, d = original_shape = tuple(im.shape)
    assert d == 3
    hcol = np.zeros((w, h, d), np.uint8)
    hcol = hcol + 50
    hcol[:, :, 0] = hsv[:, :, 0]
    hcol[:, :, 1] = dct_clear(hsv[:, :, 1], 2)
    hcol[:, :, 2] = dct_clear(hsv[:, :, 2], 2)


    return hcol[:,:,1]

import orientation as orient
folder="17"
imno="1"
original=cv2.imread("cvl_pcb_dslr_1/pcb"+folder+"/rec"+imno+".jpg")
omask=cv2.imread("cvl_pcb_dslr_1/pcb"+folder+"/rec"+imno+"-mask.png",0)

original,omask=orient.straighten(original, omask,400)

x,y,w,h = cv2.boundingRect(omask)
print "porte imagen ",w,h
original=original[y:y+h,x:x+w]
omask=omask[y:y+h,x:x+w]

#Se obtiene la mascara de fondo
import background_segmentation as bs


if (os.path.isfile("cvl_pcb_dslr_1/pcb"+folder+"/rec"+imno+"-nmask.png")):
    nmask = cv2.imread("cvl_pcb_dslr_1/pcb" + folder + "/rec" + imno + "-nmask.png", 0)
    print "cargada mascara nmask"
else:
    print "Guardando mascara nmask"
    nmask=(bs.back_mask(original) + cv2.bitwise_not(omask))
    cv2.imwrite("cvl_pcb_dslr_1/pcb"+folder+"/rec"+imno+"-nmask.png",nmask)
cv2.imshow("mask=",cv2.resize(nmask,(900,600)))
omask=nmask/255
#Se obtienen recortes
originalers=original.copy()

dx=0
dy=0

x=0+dx
x2=900+dx
y=0+dy
y2=550+dy

originals=originalers[y:y2,x:x2]
im=original[y:y2,x:x2]
mask=omask[y:y2,x:x2]
img_search= prepros(im)




match_threshold = 0.8
background_ratio_threshold = 0.05
prev_uoi_threshold=0.2


"""if (os.path.isfile("bbs/bbs_" + folder + "_" + imno+".npy")):
    bounding_boxes=np.load("bbs/bbs_" + folder + "_" + imno+".npy")
    print bounding_boxes
    for bb in bounding_boxes:
        x,y,x2,y2=bb
        cv2.rectangle(im, (x, y), (x2 ,y2), (0, 0, 0), 2)
else:"""
bounding_boxes=[]

count=0
while(1):
    #Se selecciona un area
    x,y,w,h = cv2.selectROI("Roi_selector", im)
    if (w==0):
        np.save("bbs/bbs_" + folder + "_" + imno+"-"+str(x)+"-"+str(y), bounding_boxes)
        print "break", count
        break
    #Se agrega a cosa
    # todo - agregar
    box_roi = [x, y, x + w, y + h]
    bounding_boxes.append(box_roi)
    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #se obtiene imagen original
    imCrop = img_search[y:y + h, x:x + w]
    imCr = originals[y:y + h, x:x + w]
    cv2.imwrite("crops/"+folder+"_"+imno+"_"+str(count)+".png",imCr)
    #Se obtienen imagenes parecidas al corte
    res = cv2.matchTemplate(img_search, imCrop, cv2.TM_CCOEFF_NORMED)
    im,bounding_boxes=purge(res, im, mask, w, h, background_ratio_threshold, prev_uoi_threshold,
          originals,bounding_boxes,folder,imno,count)
    count+=1



