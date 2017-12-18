"""
    Script implementado para realizar visualizaciones de PCBs, no completo
"""
import cv2
import numpy as np
import os.path
import background_segmentation as bs

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

#Funcion para mostrar imagen reducida
def show(tit,im):
    im = cv2.resize(im, (960, 540))
    cv2.imshow(tit,im)

#Funcion para realizar busqueda selectiva
def selective_search(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(img)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()

class component:
    'Clase que define componentes dentro de imagen'
    empCount = 0

    def __init__(self, x, y, w, h, clase=None, mask=None ,selected=False):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.clase=clase
        self.mask=mask
        self.selected=selected
    # internas- descriptor, segmentacion interna (pads, cuerpo), net (definir si va en pad o en fondo
    # Metodos operaciones sobre bbs, cambiar de grupo,
    # segmentar manualmente, seleccionar manualmente, mover manualmente
    # display bbs segun grupo

    def setclase(self, newclass):
        self.clase=newclass

    def getclase(self):
        return self.clase

    def setmask(self, newmask):
        self.clase=newmask

    def getmask(self):
        return self.mask

    def getrect(self):
        return (self.x, self.y),(self.x + self.w, self.y + self.h)

    def getbbs(self):
        return [self.x,self.y,self.w,self.h]

    def getimcrop(self, img, resize=0):
        x1 = max(0, self.x - int(self.w * resize))
        x2 = self.x  +int(self.w * (1 + resize))
        y1 = max(self.y - int(self.h * resize), 0)
        y2 = self.y + int(self.h * (1 + resize))
        return img[y1:y2,x1:x2]

    def drawbb(self,img,colors):
        pts=self.getrect()
        cv2.rectangle(img, pts[0], pts[1], colors[self.clase*3], 3)



#Funcion para realizar operaciones de vista basicas en imagenes (incompleta)
def ventana(img, soldermask,img_nom):

    components=[]
    print "Leyendo bounding Boxes"
    BBS=np.load(img_nom.replace(".jpg","-BBS.npy"))#leer bbs
    if os.path.exists(img_nom.replace(".jpg", "-labels.npz")):
        kmeansdata=np.load(img_nom.replace(".jpg", "-labels.npz"))#leer clase
        labels=kmeansdata['labels']
    else:
        return False
    #crear objeto bb,clase
    i=0
    print "Dibujando bounding boxes, cada clase en un color distinto"
    for bb in BBS:
        x,y,w,h=bb
        components.append(component(x,y,w,h,clase=labels[i]))
        components[i].drawbb(img,colors)
        i+=1
    cv2.imwrite("BBS.png", img)
    cv2.namedWindow('image')


    while(1):
        show('image',img)
        k = cv2.waitKey(1) & 0xFF
        print k

        # todo, implementar en funcion
        if k == 27:
            break

        elif k == ord('s'):
            print "guardando imagen"
            cv2.imwrite("img",img)
        elif k==ord('c'):
            print "crop"
            x,y,w,h = cv2.selectROI("image", img)
            if (h==0 or w==0):
                print "No se ha seleccionado ROI"
                continue
            com = component(x,y,w,h)
            cv2.imshow("imcrop",com.getimcrop(img))

    cv2.destroyAllWindows()