from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from matplotlib.colors import hsv_to_rgb
from matplotlib import pyplot as plt
import numpy as np
import cv2


def make_mask(labels, labels_idx, w, h):
    # print "shapes", np.shape(labels), np.shape(labels_idx)
    # print "w", w, "h",h
    mask = np.zeros((max(labels) + 1, w, h), dtype=np.uint8)
    # print "shape mask", np.shape(mask)

    i = labels_idx[0] / h
    j = labels_idx[0] % h
    # print "i,j",i,",",j

    for idx in np.arange(len(labels)):
        lab = labels[idx]
        # print lab, i[idx],j[idx]
        mask[lab, i[idx], j[idx]] = 1
    # print "terminada mascara"
    return mask


# Pinta una mascara
def paint_mask(mask, colour):
    w, h = tuple(mask.shape)
    d = len(colour)
    image = np.zeros((w, h, d))
    for i in range(w):
        for j in range(h):
            image[i, j] = colour * mask[i, j]

    return np.uint8(image)


# Separa escala de grises en hsv
def idx_gray_split(image_array, umbral):
    #
    s = image_array[:, 1]
    v = image_array[:, 2]
    umbral = umbral
    color_idx = np.where(np.logical_and(np.greater_equal(s, umbral), np.greater_equal(v, umbral)))
    gray_idx = np.where(np.logical_or(np.less(s, umbral), np.less(v, umbral)))
    # print color_idx,gray_idx
    return color_idx, gray_idx

# Hace el binning
def binning(input_array):
    bin_array = input_array.astype(np.float64)
    bin_array[:, 0] = np.round(bin_array[:, 0] / 6)
    bin_array[:, 1] = np.round(bin_array[:, 1] * 8.0 / 255.0)
    bin_array[:, 2] = np.round(bin_array[:, 2] * 8.0 / 255.0)
    return bin_array

# Deshace el binning
def unbinning(array):
    array[:, 0] = array[:, 0] * 6
    array[:, 1] = array[:, 1] / 8 * 255
    array[:, 2] = array[:, 2] / 8 * 255
    return array.astype(np.uint8)


# Realiza k-means por color, retorna centroides y etiquetas
def color_kmeans(pixel_array, n_colors):
    pixel_array = binning(pixel_array)
    image_array_sample = shuffle(pixel_array, random_state=0)[:1000]
    # kmeans = KMeans(n_clusters=n_colors, init=colors).fit(image_array_sample)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(pixel_array)
    codebook = unbinning(kmeans.cluster_centers_)

    return codebook, labels

# Realiza k-means en los pixeles grises
def gray_kmeans(pixel_array, n_grays):
    clusters = np.array([0, 125, 255]).reshape(-1, 1)
    # n_c=len(clusters)
    image_array_sample = shuffle(pixel_array[:, 2], random_state=0)[:1000]
    # kmeans = KMeans(n_clusters=n_c, init=clusters).fit(image_array_sample.reshape(-1,1))

    kmeans = KMeans(n_clusters=n_grays, random_state=0).fit(image_array_sample.reshape(-1, 1))
    labels = kmeans.predict(pixel_array[:, 2].reshape(-1, 1))
    gray_codebook = kmeans.cluster_centers_
    codebook = np.zeros((n_grays, 3), np.uint8)
    codebook[:, 0] = 0
    codebook[:, 1] = 0
    for i in np.arange(n_grays):
        codebook[i, 2] = int(gray_codebook[i])
    return codebook, labels

# no hace nada
def nothing(x):
    pass

# Limpia las n diagonales superior-izquierdas del dct
def dct_clear(img, n):
    imf = np.float32(img) / 255.0  # float conversion/scale
    dst = cv2.dct(imf)  # the dct
    for i in range(n):
        for j in range(n):
            if (i + j < n):
                dst[i, j] = 0

    dst[0, 0] = 0
    inv = cv2.idct(dst)
    mi = np.min(inv)
    ma = np.max(inv)
    # print mi,ma
    k = 255.0 / (ma - mi)
    inv = (inv - mi) * k
    # inv=abs(inv)*255
    # print inv
    return inv.astype(np.uint8)


# Obtiene zonas conexas para una clasificacion mas limpia
def get_conected(soldermask, serigraphy):
    # Se obtienen regiones conexas
    img = soldermask + serigraphy
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img, kernel, iterations=2)
    # showImGray(img)
    # showImGray(erosion)

    # se obtienen los componentes conexos
    a, labels, stats, centroids = cv2.connectedComponentsWithStats(erosion)
    areas = []
    # se obtiene areas
    for i in np.arange(a):
        areas.append(stats[i][cv2.CC_STAT_AREA])
    # Se obtiene el area de mayor conectividad
    lab_max = np.argmax(areas[1:-1])
    # Se obtiene region de mayor conectividad
    max_connected = np.array(np.equal(labels, lab_max + 1) * 255, np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(max_connected, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((10, 10), np.uint8)
    back = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return back

# muestra imagen en matplotlib
def hsvplot(hsv, cutBox=False):
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # showImBGR(image,cutBox)
    if (cutBox != False):
        x1, y1, x2, y2 = cutBox
        image = image[y1:y2, x1:x2]
    plt.imshow(image)
    plt.show()


def hsv_kmeans(img, n_colors, n_grays):
    img = cv2.bilateralFilter(img, 5, 90, 370)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    w, h, d = original_shape = tuple(img.shape)

    assert d == 3
    hcol = np.zeros((w, h, d), np.uint8)
    hcol = hcol + 50
    hcol[:, :, 0] = hsv[:, :, 0]
    hcol[:, :, 1] = dct_clear(hsv[:, :, 1], 0)
    hcol[:, :, 2] = dct_clear(hsv[:, :, 2], 0)


    image_array = np.reshape(hcol, (w * h, d))

    # Se separa entre imagenes en grises y en no grises
    color_idx, gray_idx = idx_gray_split(image_array, 40)

    pixel_color = image_array[color_idx]
    pixel_gray = image_array[gray_idx]

    # print "pixel color", pixel_color
    # print "pixel gray",pixel_gray


    # Se toma parte de la data para el clustering
    print("Realizando clustering con colores separados")
    color_codebook, color_labels = color_kmeans(pixel_color, n_colors)
    gray_codebook, gray_labels = gray_kmeans(pixel_gray, n_grays)

    print "Color Codegook", color_codebook,
    print "Gray Codebook", gray_codebook

    gray_masks = make_mask(np.array(gray_labels), np.array(gray_idx), w, h)
    color_masks = make_mask(np.array(color_labels), np.array(color_idx), w, h)
    return gray_masks, color_masks, color_codebook, gray_codebook

# muestra imagen en matplotlib
def hsvplot(hsv, cutBox=False):
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # showImBGR(image,cutBox)
    if (cutBox != False):
        x1, y1, x2, y2 = cutBox
        image = image[y1:y2, x1:x2]
    plt.imshow(image)
    plt.show()


def hsv_kmeans(img, n_colors, n_grays):
    img = cv2.bilateralFilter(img, 5, 90, 370)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    w, h, d = original_shape = tuple(img.shape)

    assert d == 3
    hcol = np.zeros((w, h, d), np.uint8)
    hcol = hcol + 50
    hcol[:, :, 0] = hsv[:, :, 0]
    hcol[:, :, 1] = dct_clear(hsv[:, :, 1], 0)
    hcol[:, :, 2] = dct_clear(hsv[:, :, 2], 0)


    image_array = np.reshape(hcol, (w * h, d))

    # Se separa entre imagenes en grises y en no grises
    color_idx, gray_idx = idx_gray_split(image_array, 40)

    pixel_color = image_array[color_idx]
    pixel_gray = image_array[gray_idx]

    # print "pixel color", pixel_color
    # print "pixel gray",pixel_gray


    # Se toma parte de la data para el clustering
    print("Realizando clustering con colores separados")
    color_codebook, color_labels = color_kmeans(pixel_color, n_colors)
    gray_codebook, gray_labels = gray_kmeans(pixel_gray, n_grays)

    print "Color Codegook", color_codebook,
    print "Gray Codebook", gray_codebook

    gray_masks = make_mask(np.array(gray_labels), np.array(gray_idx), w, h)
    color_masks = make_mask(np.array(color_labels), np.array(color_idx), w, h)
    return gray_masks, color_masks, color_codebook, gray_codebook

def imptest():
    print "lasorra"

def back_mask(img,n_colors=2, n_grays=3):
    # Se obtiene la imagen clusterizada en color hsv
    gray_masks, color_masks, color_codebook, gray_codebook = hsv_kmeans(img, n_colors, n_grays)

    # Se obtienen zonas mayoritarias, en este caso la mayor color sera soldermask y la mayor gris serigrafia
    garr = []
    for i in np.arange(np.shape(gray_masks)[0]):
        garr.append(np.sum(gray_masks[i]))

    carr = []
    for i in np.arange(np.shape(color_masks)[0]):
        carr.append(np.sum(color_masks[i]))
    maxc = np.argmax(carr)
    maxg = np.argmax(garr)

    print "propuesta soldermask ", maxc, " propuesta serigrafia ", maxg
    print "con valores de ", carr, garr

    back = get_conected(color_masks[maxc], gray_masks[maxg])
    return back

def color_cluster(img, n_colors = 2, n_grays = 3):
    w, h, d = original_shape = tuple(img.shape)
    # Se obtiene la imagen clusterizada en color hsv

    gray_masks, color_masks, color_codebook, gray_codebook = hsv_kmeans(img, n_colors, n_grays)

    print "generando mascara con color"
    # Se recuperan mascaras
    res_image = np.zeros(original_shape, np.uint8)
    gray_image = np.zeros(original_shape, np.uint8)
    color_image = np.zeros(original_shape, np.uint8)

    for i in np.arange(np.shape(gray_masks)[0]):
        painted_mask = paint_mask(gray_masks[i], gray_codebook[i])
        #hsvplot(painted_mask)
        # hsvwrite(folder +"gray_mask"+str(i)+".png", painted_mask)
        res_image += painted_mask
        gray_image += painted_mask

    for i in np.arange(np.shape(color_masks)[0]):
        painted_mask = paint_mask(color_masks[i], color_codebook[i])
        # hsvwrite(folder + "painted_mask" + str(i)+".png", painted_mask)
        #hsvplot(painted_mask)
        res_image += painted_mask
        color_image += painted_mask
    #cv2.imwrite(img)
    hsvplot(res_image)
    #hsvplot(gray_image)
    #hsvplot(color_image)
    #hsvplot(paint_mask(color_masks[1] + gray_masks[1], color_codebook[0]))

n_colors = 2
n_grays = 3
folder="/Users/Jpcoseco/PycharmProjects/OPGE/test/"


# Carga imagen, transforma a opencv y cambia tamano
img = cv2.imread('perro.jpg')
color_cluster(img,10,10)
cv2.waitKey(0)
# Se obtiene la imagen clusterizada en color hsv
"""gray_masks, color_masks, color_codebook, gray_codebook=hsv_kmeans(img,n_colors,n_grays)

# Se obtienen zonas mayoritarias, en este caso la mayor color sera soldermask y la mayor gris serigrafia
garr=[]
for i in np.arange(np.shape(gray_masks)[0]):
    garr.append(np.sum(gray_masks[i]))

carr=[]
for i in np.arange(np.shape(color_masks)[0]):
    carr.append(np.sum(color_masks[i]))
maxc=np.argmax(carr)
maxg=np.argmax(garr)


print "propuesta soldermask ", maxc, " propuesta serigrafia ", maxg
print "con valores de ", carr,garr

# Cambiar a true para obtener mascaras
mm=False

if (mm==True):
    print "generando mascara con color"
    #Se recuperan mascaras
    res_image = np.zeros((w, h, d),np.uint8)
    gray_image = np.zeros((w, h, d),np.uint8)
    color_image=  np.zeros((w, h, d),np.uint8)

    for i in np.arange(np.shape(gray_masks)[0]):
        painted_mask=paint_mask(gray_masks[i], gray_codebook[i])
        hsvplot( painted_mask)
        #hsvwrite(folder +"gray_mask"+str(i)+".png", painted_mask)
        res_image+=painted_mask
        gray_image+=painted_mask


    for i in np.arange(np.shape(color_masks)[0]):
        painted_mask = paint_mask(color_masks[i], color_codebook[i])
        #hsvwrite(folder + "painted_mask" + str(i)+".png", painted_mask)
        hsvplot( painted_mask)
        res_image += painted_mask
        color_image += painted_mask
    hsvplot(res_image)
    hsvplot(gray_image)
    hsvplot(color_image)
    hsvplot(paint_mask(color_masks[1]+gray_masks[1],color_codebook[0]))

back=get_conected(color_masks[maxc],gray_masks[maxg])
imand=cv2.bitwise_and(img,img,mask=back)
pads=cv2.bitwise_and(img,img,mask=cv2.bitwise_not(back))
cv2.imwrite("comps.png",pads)
cv2.imwrite("Back.png",back)


# todo - generalizar para cualquier placa incluyendo fondo"""