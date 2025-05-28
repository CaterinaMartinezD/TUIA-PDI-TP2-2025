import cv2
import numpy as np
import matplotlib.pyplot as plt

def cargar_imagen(mostrar = False):
    img = cv2.imread('placa.png', cv2.IMREAD_COLOR)                             # Leemos la imagen en color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                  # Convertimos la imagen de BGR a RGB

    if (mostrar == True):
        plt.imshow(img, cmap='gray')
        plt.title('Placa de circuito impreso (PCB)')
        plt.show()

    return img

def procesar_img(img, mostrar = False):
    total_resistencia = 0
    img_copia = img.copy()
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                           # Convierte la img de BGR a escala de grises
    th, img_bin = cv2.threshold(img_gris, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    clausura = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    apertura = cv2.morphologyEx(clausura, cv2.MORPH_OPEN, kernel2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(apertura)
    for i in range(1, num_labels):
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        componente = (labels == i).astype(np.uint8)
        contornos, _ = cv2.findContours(componente, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contornos[0]
        perimetro = cv2.arcLength(cnt, True)

        rho = 4 * np.pi * area / (perimetro**2)    #factor de forma
        aspect_ratio = max(w, h) / min(w, h)       #max y min para que tome las resistencias verticales y horizontales

        if (rho < 0.5 and aspect_ratio > 2 and 8000 < area < 11000): #Resistencias (bien)
            total_resistencia += 1
            cv2.rectangle(img_copia, (x, y), (x + w, y + h), (0, 0, 255), 2)

        elif (0.4 < rho < 0.8 and aspect_ratio < 1.6 and area > 8000): # Capacitores (falta 1 y detecta 2 mal)
            cv2.rectangle(img_copia, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        #Corregir deteccion para detectar el chip y el capacitor que falta


    if (mostrar == True):
        plt.imshow(img_copia, cmap='gray')
        plt.title('Placa de circuito impreso (PCB)')
        plt.show()

    return img_copia, total_resistencia

def clasificar_img(capacitores, mostrar = False):
    pass

def contar_resistencias(total_resistencias, mostrar = False):
    if (mostrar == True):
        print(f"El total de resistencias detectadas en la placa son: {total_resistencias}")
    return

#------------------------------------------------------------------------------------------------------------------------------------
img = cargar_imagen(mostrar = False)
img_procesada, t_resistencias = procesar_img(img, mostrar = True)
#clasificar_img(t_capacitores, mostrar = False)
contar_resistencias(t_resistencias, mostrar = False)
