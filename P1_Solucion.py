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

def detectar_resistencias(img, mostrar = False):
    resistencias_coord = []
    total_resistencia = 0                                                                       
    img_copia = img.copy()                                                                      #
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # Convierte la img de BGR a escala de grises
    bin, img_bin = cv2.threshold(img_gris, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)        #

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

        rho = 4 * np.pi * area / (perimetro**2)                                                 #factor de forma
        aspect_ratio = max(w, h) / min(w, h)                                                    # max y min para que tome las resistencias verticales y horizontales

        if (rho < 0.5 and aspect_ratio > 2 and 8000 < area < 11000):                            #Resistencias (bien)
            total_resistencia += 1
            resistencias_coord.append((x, y, w, h))
            cv2.rectangle(img_copia, (x, y), (x + w, y + h), (0, 0, 255), 4)
    
    if (mostrar == True):
        plt.imshow(img_copia, cmap='gray')
        plt.title('Placa de circuito impreso (PCB): Resistencias')
        plt.show()

    return img_copia, img_bin, total_resistencia, resistencias_coord

def detectar_capacitores(img, img_bin, img_procesada, resistencias_coord, mostrar = False):
    img_resultado = img.copy()
    capacitores_coord = []
    capacitores_area = []
    img_inv = cv2.bitwise_not(img_bin)
    for x, y, w, h in resistencias_coord:
        cv2.rectangle(img_inv, (x, y), (x + w, y + h), 255, -1)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    clausura3 = cv2.morphologyEx(img_inv, cv2.MORPH_CLOSE, kernel3)
    kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clausura4 = cv2.morphologyEx(clausura3, cv2.MORPH_DILATE, kernel4)
    img_inv2 = cv2.bitwise_not(clausura4)
    num_labels_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(img_inv2)

    for i in range(1, num_labels_c):
        
        x1, y1, w1, h1 = stats_c[i, cv2.CC_STAT_LEFT], stats_c[i, cv2.CC_STAT_TOP], stats_c[i, cv2.CC_STAT_WIDTH], stats_c[i, cv2.CC_STAT_HEIGHT]
        area1 = stats_c[i, cv2.CC_STAT_AREA]
        
        componente1 = (labels_c == i).astype(np.uint8)
        contornos1, _ = cv2.findContours(componente1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos1:
            continue
        cnt1 = contornos1[0]
        perimetro1 = cv2.arcLength(cnt1, True)
        if perimetro1 == 0:
            continue
        rho1 = 4 * np.pi * area1 / (perimetro1**2)
        aspect_ratio1 = max(w1, h1) / min(w1, h1)
        
        # Condiciones para capacitores con clausura3
        if (0.65 < rho1 < 1 and aspect_ratio1 < 2 and area1 > 6000): #Capacitores, bien!
            capacitores_coord.append((x1, y1, w1, h1))
            capacitores_area.append(area1)
            cv2.rectangle(img_resultado, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 4)
            cv2.rectangle(img_procesada, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 4)

    if (mostrar == True):
        plt.imshow(img_resultado, cmap='gray')
        plt.title('Placa de circuito impreso (PCB): Capacitores')
        plt.show()

    return img_procesada, capacitores_coord, capacitores_area

def detectar_chip(img, img_procesada, resistencias_coord, capacitores_coord, mostrar = False):
    img_resultado = img.copy()
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gris, 55, 255, cv2.THRESH_BINARY_INV)

    for x, y, w, h in resistencias_coord:
        cv2.rectangle(img_bin, (x, y), (x + w, y + h), 0, -1)
    for x, y, w, h in capacitores_coord:
            cv2.rectangle(img_bin, (x, y), (x + w, y + h), 0, -1)

    B = cv2.getStructuringElement(cv2.MORPH_RECT, (45,45))
    Aclau = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, B)
    a = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    Aclau2 = cv2.morphologyEx(Aclau, cv2.MORPH_OPEN, a)

    contornos, _ = cv2.findContours(Aclau2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > 160000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.rectangle(img_procesada, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    if (mostrar == True):
        plt.imshow(img_resultado, cmap='gray')
        plt.title('Placa de circuito impreso (PCB): Capacitores')
        plt.show()

    return img_procesada

def imprimir_segmentaciones(img, mostrar = False):
    if (mostrar == True):
        plt.imshow(img, cmap='gray')
        plt.title('Placa de circuito impreso (PCB)')
        plt.show()
    return

def clasificar_capacitores(img, coordenada, area, mostrar = False):
    img_resultado = img.copy()
    for (x, y, w, h), area in zip(coordenada, area):
        if (area < 7500):
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(img_resultado, "Categoria 1", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        elif (30000 < area < 40000):
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(img_resultado, "Categoria 2", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        elif (10000 < area < 15000):
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cv2.putText(img_resultado, "Categoria 3", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
        else:
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 255), 4)
            cv2.putText(img_resultado, "Categoria 4", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

    if (mostrar == True):
        plt.imshow(img_resultado, cmap='gray')
        plt.title('Placa de circuito impreso (PCB)')
        plt.show()

    return

def contar_resistencias(total_resistencias, mostrar = False):
    if (mostrar == True):
        print(f"El total de resistencias detectadas en la placa son: {total_resistencias}")
    return

#------------------------------------------------------------------------------------------------------------------------------------
img = cargar_imagen(mostrar = False)
img_procesada, img_bin, t_resistencias, resistencias_coord = detectar_resistencias(img, mostrar = False)
img_procesada2, capacitores_coord, capacitores_area = detectar_capacitores(img, img_bin, img_procesada, resistencias_coord, mostrar = False)
img_procesada3 = detectar_chip(img, img_procesada2, resistencias_coord, capacitores_coord, mostrar = True)
imprimir_segmentaciones(img_procesada3, mostrar = False)
clasificar_capacitores(img, capacitores_coord, capacitores_area, mostrar = False)
contar_resistencias(t_resistencias, mostrar = False)
