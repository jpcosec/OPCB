import cv2
import numpy as np
import os.path
import background_segmentation as bs
import orientation as orient

# FUNCIONES AUXILIARES

# Funcion para encontrar IOU de 2 boxes
def bb_intersection_over_union(boxA, boxB):
    # Recibe 2 boxes, retorna el IOU de ambas
    # Modificada desde para procesar varias imagenes desde
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

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

# Funcion que limpia las n diagonales superior izquierdas de la DCT de la imagen
def dct_clear(img,n):
    #Recibe imagen y entero, retorna la imagen con la DCT limpia

    #Se convierte la imagen a floats y se obtiene la transformacion de cosenos discretos
    imf = np.float32(img)/255.0
    dst = cv2.dct(imf)
    #Se itera sobre la esquina dejandola en 0
    for i in range(n):
        for j in range(n):
            if (i+j <n):
                dst[i,j]=0
    dst[0,0]=0
    #Se obtiene la inversa de la DCT
    inv = cv2.idct(dst)
    #Y se reescala la imagen se salida
    mi=np.min(inv)
    ma=np.max(inv)
    k=255.0/(ma-mi)
    inv=(inv-mi)*k
    return inv.astype(np.uint8)


# Funcion para obtener una zona conexa en base a una imagen binaria
def get_conected(img):
    # Se obtienen regiones conexas
    a , labels ,  stats, centroids = cv2.connectedComponentsWithStats(img)
    areas=[]
    # se obtiene areas
    for i in np.arange(a):
        areas.append(stats[i][cv2.CC_STAT_AREA])
    # Se obtiene el area de mayor conectividad
    lab_max=np.argmax(areas[1:-1])
    # Se obtiene region de mayor conectividad
    max_connected=np.array(np.equal(labels,lab_max+1)*255,np.uint8)
    # Se hace operacion de apertura
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(max_connected, cv2.MORPH_OPEN, kernel)
    return opening

# Funcion utilizada para descartar zonas con demasiado soldermask
def purge(res,im,mask,w,h,background_ratio_threshold,prev_uoi_threshold,originals,bounding_boxes,folder,imno,n):
    # Recibe una imagen y un numero de matches, junto con varios parametros

    # Se buscan aquellos puntos donde la semejanza del match sea mayor al umbral
    loc = np.where(res >= match_threshold)
    area = w * h

    p = 0
    for pt in zip(*loc[::-1]):
        # Se guarda el valor del box
        box_match=[ pt[0],pt[1], pt[0] + w, pt[1] + h]

        # Se obtiene un ratio de cantidad de fondo en roi
        intersec_back = np.sum(mask[pt[1]:pt[1] + h, pt[0]:pt[0] + w])
        back_ratio = intersec_back / area

        # Se calcula el maximo UOI con boxes guardados anteriormente
        prev_bbs_uoi = bb_intersection_over_union(box_match, bounding_boxes)
        max_prev_uoi = max(prev_bbs_uoi)

        # Si el fondo en ROI y el UOI con boxes anteriores son menores que sus respectivos threshold
        if (back_ratio<background_ratio_threshold and max_prev_uoi<prev_uoi_threshold):
            p += 1
            # Se agrega bounding box a lista de bounding boxes
            bounding_boxes.append(box_match)
            # Se dibuja cuadro
            cv2.rectangle(im, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
            # Se corta y guarda corte de imagen
            imCr=originals[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
            cv2.imwrite("match/" + folder + "_" + imno+"_"+str(n) +"_"+str(p) +".png", imCr)
    return im, bounding_boxes

# Funcion utilizada para preprocesar la imagen con filtrado bilateral y limpieza del DCT
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


# CUERPO DE SCRIPT

# Carga de imagenes y nombres
folder="/Users/Jpcoseco/Pictures/imagenes/cvl_pcb_dslr_1/"
imno="1"
nfolder="1"

img_name = folder + "pcb"+ nfolder +"/rec"+ imno +".jpg"
mask_name = folder + "pcb"+ nfolder +"/rec"+ imno +"-mask.png"
soldermask_name = folder + "pcb"+ nfolder +"/rec"+ imno +"-nmask.png"

original=cv2.imread(img_name)
omask=cv2.imread(mask_name,0)

if(os.path.isfile(img_name)):
    print "imagen cargada"
else:
    print "imagen no encontrada"
    exit()

if (os.path.isfile(mask_name)):
    print "mascara de fondo"
else:
    print "mascara de fondo no encontrada"
    exit()


# Se orientan y cortan las imagenes
original,omask =orient.cut_and_straighten(original, omask)
xo,yo,wo,ho = cv2.boundingRect(omask)
print "porte imagen ",wo,ho

original=original[yo:yo+ho,xo:xo+wo]
omask=omask[yo:yo+ho,xo:xo+wo]

#Se obtiene la mascara de fondo

if (os.path.isfile(soldermask_name)):
    print "cargada mascara de soldemask"
    nmask = cv2.imread(soldermask_name, 0)

else:
    print "Guardando mascara de soldermask"
    nmask=(bs.solder_mask(original) + cv2.bitwise_not(omask))
    cv2.imwrite(soldermask_name,nmask)

# Se muestra mascara del soldermask
cv2.imshow("mask=",cv2.resize(nmask,(900,600)))
omask=nmask/255

# Se obtienen recortes de la imagen para buscar
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

# Se preprocesa imagen de busqueda para
img_search= prepros(im)

# Se definen valores de umbral
match_threshold = 0.8
background_ratio_threshold = 0.05
prev_uoi_threshold=0.2

# Se itera sobre imagen buscando boxes
bounding_boxes=[]
count=0
while(1):
    # Se obtiene ROI desde imagen usando mouse
    x,y,w,h = cv2.selectROI("Roi_selector", im)

    # Se agregan coordenadas del ROI en la lista
    bounding_boxes.append([x,y,w,h])

    # Y se guardan en la imagen
    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Se realiza corte sobre la imagen de busqueda y original
    imCrop = img_search[y:y + h, x:x + w]
    imCr = originals[y:y + h, x:x + w]

    # Se guarda el corte de la imagen original
    cv2.imwrite("crops/"+folder+"_"+imno+"_"+str(count)+".png",imCr)

    #Se obtienen imagenes parecidas al corte
    res = cv2.matchTemplate(img_search, imCrop, cv2.TM_CCOEFF_NORMED)

    # Se purgan imagenes para quitar imagenes que no procedan
    im, bounding_boxes=purge(res, im, mask, w, h, background_ratio_threshold, prev_uoi_threshold,
          originals, bounding_boxes, folder, imno, count)
    count+=1
# Se guardan bounding boxes
bbs_name = folder + "pcb"+ nfolder +"/rec"+ imno +"-bbsGT"
np.save(bbs_name ,bounding_boxes)
print "Bounding boxes guardados"