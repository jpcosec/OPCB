import cv2
import numpy as np


# Funcion usada para obtener una medida del angulo de orientacion de la imagen
def hough_angle(img, minLineLength):
    # Retorna angulo de orientacion en funcion de las lineas hough detectadas

    #se obtienen los bordes usando Canny, se detectan las lineas usando Hough
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, minLineLength)
    angs = []
    # se obtiene la diferencia de angulo cuantizado en 90 grados en modulo 90 para obtener una distancia al eje
    for line in lines:
        rho, theta= line[0]
        ang=np.mod(int(theta * 180 / np.pi), 90)
        angs.append(ang )
    #se retorna el angulo mas repetido
    counts = np.bincount(angs)
    return np.argmax(counts)

# Funcion para enderezar imagen usando mascara
def straighten(img, mask, threshold=100):
    #retorna imagen y mascara enderezada, recibe imagen original, mascara original y umbral opcional

    #se obtiene el angulo
    angle = hough_angle( mask, threshold)
    rows, cols, end = np.shape(img)
    #se crea matriz de rotaci0n
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    #se rotan imagenes
    dst = cv2.warpAffine(img, M, (cols, rows))
    mask =cv2.warpAffine(mask, M, (cols, rows))
    return dst, mask

# Funcion para enderezar imagen y recortar segun mascara
def cut_and_straighten(img,mask,threshold=100):
    # Retorna imagenes recortadas y enderezadas, recibe imagen, mascara y umbral opcional

    #se endereza imagen
    original, omask = straighten(img, mask,threshold)
    #se obtiene rectangulo minimo que encierra la mascara
    xo, yo, wo, ho = cv2.boundingRect(omask)
    #se recortan imagenes
    original = original[yo:yo + ho, xo:xo + wo]
    omask = omask[yo:yo + ho, xo:xo + wo]
    return original, omask
