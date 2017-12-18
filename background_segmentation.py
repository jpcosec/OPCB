"""
    Implementacion libre de segmentacion de imagenes en colores basado con modificaciones en
    "Fast Image Segmentation Based on K-Means Clustering with Histograms in HSV Color Space" (2008)
    de Chen, Chen y Chien, DOI:10.1109/MMSP.2008.4665097
"""

from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import numpy as np
import cv2

# METODOS AUXILIARES DE CLUSTERING HSV

# Funcion que limpia las n diagonales superior izquierdas de la DCT de la imagen
def dct_clear(img,n=0):
    #Recibe imagen y entero, retorna la imagen con la DCT limpia

    #Se convierte la imagen a floats y se obtiene la transformacion de cosenos discretos
    imf = np.float32(img)/255.0
    dst = cv2.dct(imf)

    #Se itera sobre la esquina dejandola en 0
    for i in range(n):
        for j in range(n):
            if (i+j <=n):
                dst[i,j]=0
    dst[0,0]=0

    #Se obtiene la inversa de la DCT
    inv = cv2.idct(dst)

    #Y se reescala la imagen para salida
    mi=np.min(inv)
    ma=np.max(inv)
    k=255.0/(ma-mi)
    inv=(inv-mi)*k
    return inv.astype(np.uint8)

# Funcion que crea tensor de imagenes binarias desde lista de pixeles
def make_mask(labels, labels_idx, w, h):
    # Retorna un tensor con imagenes binarias de dimensiones especificadas
    # Recibe lista de pixeles etiquetadas con su respectiva clase

    # Se crea tensor con las dimensiones especificadas
    mask = np.zeros((max(labels) + 1, w, h), dtype=np.uint8)
    i = labels_idx[0] / h
    j = labels_idx[0] % h
    # Se itera sobre vector asignando capa a posicion de pixel
    for idx in np.arange(len(labels)):
        lab = labels[idx]
        mask[lab, i[idx], j[idx]] = 1

    return mask

# funcion para colorear una mascara con un color indicado
def paint_mask(inmask, incodebook):
    w, h = tuple(inmask.shape)
    d = 3
    image = np.zeros((w, h, d), np.uint8)
    image[:, :, 0] = inmask * incodebook[0]
    image[:, :, 1] = inmask * incodebook[1]
    image[:, :, 2] = inmask * incodebook[2]
    return image

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
    print bin_array
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

# METODOS PARA PROCESAR IMAGEN

# Funcion implementada para realizar clustering k-means en imagen generica
def hsv_kmeans(img, n_colors=2, n_grays=3, clear_dct=False):
    # Retorna tensor de imagenes binarias de pixeles clusterizados mas el codebook, grises y colores separados
    # Recibe imagen mas numero de grises y colores y opcion limpiar los bajos de DCT

    # Se realiza una operacion de filtrado bilateral
    img = cv2.bilateralFilter(img, 5, 90, 370)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    w, h, d = original_shape = tuple(img.shape)

    if clear_dct!=False:
        assert d == 3
        hcol = np.zeros((w, h, d), np.uint8)
        hcol = hcol + 50
        hcol[:, :, 0] = hsv[:, :, 0]
        hcol[:, :, 1] = dct_clear(hsv[:, :, 1], clear_dct)
        hcol[:, :, 2] = dct_clear(hsv[:, :, 2], clear_dct)
        hsv=hcol

    # Se realiza un reshape a la imagen para dejarla en columnas de valores HSV de pixel
    image_array = np.reshape(hsv, (w * h, d))

    # Se separan pixeles entre imagenes en grises y en no grises
    color_idx, gray_idx = idx_gray_split(image_array, 40)
    pixel_color = image_array[color_idx]
    pixel_gray = image_array[gray_idx]

    # Se realizan los clusterings individualmente
    print("Realizando clustering con colores separados")
    color_codebook, color_labels = color_kmeans(pixel_color, n_colors)
    gray_codebook, gray_labels = gray_kmeans(pixel_gray, n_grays)
    print "Color Codeook", color_codebook
    print "Gray Codebook", gray_codebook

    # Se reconstruyen los binarios mascaras de color
    gray_masks = make_mask(np.array(gray_labels), np.array(gray_idx), w, h)
    color_masks = make_mask(np.array(color_labels), np.array(color_idx), w, h)
    return gray_masks, color_masks, gray_codebook, color_codebook

# Funcion para obtener zonas conexas
def get_conected(soldermask, serigraphy):
    #Retorna la zona de soldermask conexa, recibe soldermask y serigrafia

    # Se suman soldermask y serigrafia
    img = soldermask + serigraphy

    # Se hace operacion de closening para eliminar vacios
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Se hace operacion de erosion
    erosion = cv2.erode(img, kernel, iterations=2)
    # se obtienen los componentes conexos
    a, labels, stats, centroids = cv2.connectedComponentsWithStats(erosion)

    # Se obtiene mayor area conexa
    areas = []
    for i in np.arange(a):
        areas.append(stats[i][cv2.CC_STAT_AREA])
    lab_max = np.argmax(areas[1:-1])
    max_connected = np.array(np.equal(labels, lab_max + 1) * 255, np.uint8)

    # Se realiza operacion de apertura
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(max_connected, cv2.MORPH_OPEN, kernel)
    # Se realiza operacion de cierre
    kernel = np.ones((10, 10), np.uint8)
    back = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return back

# Funcion semiautomatica para segmentar soldermask
def solder_mask(img, n_colors=2, n_grays=3 ):
    # Retorna mascara de soldermask en imagen binaria
    # Recibe imagen, y numero de colores a separar

    # Se obtiene la imagen clusterizada en color hsv
    gray_masks, color_masks, gray_codebook,color_codebook, = hsv_kmeans(img, n_colors, n_grays)

    # Se obtiene corte de imagen, usuario debe seleccionar zona con soldermask + serigrafia
    x, y, w, h = cv2.selectROI("Roi_selector", img[0:900,0:1000])
    print "seleccionar region con soldermask +serigrafia"
    solder_reg=np.append(color_masks[:,y:y+h,x:x+w],gray_masks[:,y:y+h,x:x+w],axis=0)

    # Se obtienen colores mayoritarios
    total_reg=np.append(color_masks,gray_masks ,axis=0)
    carr = []
    for i in np.arange( np.shape(solder_reg)[0]):
        carr.append(np.sum(solder_reg[i]))
    n = np.arange(len(carr))
    Z = [x for _, x in sorted(zip(carr, n))][::-1]
    soldermask_mask = total_reg[Z[0], :, :]
    serigraphy_mask = total_reg[Z[1], :, :]

    # Y se obtiene la mayor mascara conexa de esas zonas
    back = get_conected(soldermask_mask, serigraphy_mask)
    return back

# Funcion para realizar clustering de color a imagen generica, permite cuardar la imagen colorizada
def color_cluster(img, n_colors = 2, n_grays = 3, save=False):
    # se obtienen dimensiones originales de imagen
    w, h, d = original_shape = tuple(img.shape)
    # Se obtiene la imagen clusterizada en color hsv
    gray_masks, color_masks, color_codebook, gray_codebook = hsv_kmeans(img, n_colors, n_grays)

    print "generando mascara con color"
    # Se recuperan mascaras y crea imagen reconstruida
    res_image = np.zeros(original_shape, np.uint8)
    # Se pintan mascaras
    for i in np.arange(np.shape(gray_masks)[0]):
        # llamando a funcion paint_mask
        painted_mask = paint_mask(gray_masks[i], gray_codebook[i])
        # y acumulando resultados en res_image
        res_image += painted_mask

    for i in np.arange(np.shape(color_masks)[0]):
        painted_mask = paint_mask(color_masks[i], color_codebook[i])
        res_image += painted_mask

    # Se convierte el espacio de color de vuelta a BGR
    res_image=cv2.cvtColor(res_image, cv2.COLOR_HSV2BGR)

    # Si usa opcion save, se guarda imagen
    if save:
        cv2.imwrite("reshaped.png" ,res_image)

    return res_image, color_masks, gray_masks

# Funcion para obtener bounding boxes estimadas de componentes
def get_bbs(comp_mask):

    # Se obtienen areas conexas
    n_areas, labels, stats, centroids = cv2.connectedComponentsWithStats(comp_mask)
    # Se itera sobre cada area conexa calculando y guardando bounding box
    bbs=[]
    for i in np.arange(n_areas):
        segm=np.array(np.equal( labels, np.ones(labels.shape)*i),np.uint8)
        x, y, w, h = cv2.boundingRect(segm)
        bbs.append([x,y,w,h])

    return bbs

