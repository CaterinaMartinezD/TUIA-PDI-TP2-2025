import cv2
import numpy as np
import matplotlib.pyplot as plt

def cargar_imagen(mostrar = False):
    img = cv2.imread('placa.png', cv2.IMREAD_COLOR)                                             # Leemos la imagen en color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                                  # Convertimos la imagen de BGR a RGB

    if (mostrar == True):
        plt.imshow(img, cmap='gray')
        plt.title('Placa de circuito impreso (PCB)')
        plt.show()

    return img

def detectar_resistencias(img, mostrar = False):
    # Inicializa una lista y un contador para las resistencias
    resistencias_coord = []                                                                     
    total_resistencia = 0                                                                       
    img_copia = img.copy()                                                                      # Crea una copia de la imagen original
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # Convierte la img de BGR a escala de grises
    th_out, img_th = cv2.threshold(img_gris, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)      # Se aplica un umbral automático con "THRESH_OTSU"

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))                               # Kernel para aplicar una operación morfológica
    clausura = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel1)                               # Realiza una clausura morfológica
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))                                 # Kernel para aplicar una operación morfológica
    apertura = cv2.morphologyEx(clausura, cv2.MORPH_OPEN, kernel2)                              # Realiza una apertura morfológica

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(apertura)           # Obtiene las componentes conectadas de la imagen
    for i in range(1, num_labels):                                                              # Recorre sobre los objetos detectados excepto el fondo
        # Obtiene las coordenadas del bounding box
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]                                                       # Obtiene el área
        
        componente = (labels == i).astype(np.uint8)                                             # Crea una imagen binaria de 8 bits
        contornos, _ = cv2.findContours(componente, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Obtiene el contorno del componente
        cnt = contornos[0]                                                                      # Obtiene el primer contorno
        perimetro = cv2.arcLength(cnt, True)                                                    # Calcula el perímetro del contorno

        rho = 4 * np.pi * area / (perimetro**2)                                                 # Calcula el factor de forma
        aspect_ratio = max(w, h) / min(w, h)                                                    # Calcula el aspecto con max y min para que tome las resistencias verticales y horizontales

        if (rho < 0.5 and aspect_ratio > 2 and 8000 < area < 11000):                            # Aplica un filtro para detectar las resistencias
            total_resistencia += 1                                                              # Incrementa el contador de a uno
            resistencias_coord.append((x, y, w, h))                                             # Guarda las coordenadas de la resistencia en la lista
            cv2.rectangle(img_copia, (x, y), (x + w, y + h), (0, 0, 255), 4)                    # Dibuja el rectángulo en la imagen "img_copia"
    
    if (mostrar == True):
        plt.imshow(img_copia, cmap='gray')
        plt.title('Placa de circuito impreso (PCB): Resistencias')
        plt.show()

    return img_copia, img_th, total_resistencia, resistencias_coord

def detectar_capacitores(img, img_th , img_procesada, resistencias_coord, mostrar = False):
    img_resultado = img.copy()                                                                  # Crea una copia de la imagen original
    # Inicializa una lista y un contador para los capacitores
    capacitores_coord = []                                                                      
    capacitores_area = []                                                                       
    img_inv = cv2.bitwise_not(img_th)                                                           # Invierte los valores de la imagen binaria
    for x, y, w, h in resistencias_coord:                                                       # Recorre las coordenadas de la resistencia
        cv2.rectangle(img_inv, (x, y), (x + w, y + h), 255, -1)                                 # Elimina las resistencias de la imagen rellenando ese sector

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))                               # Kernel para aplicar una operación morfológica
    clausura = cv2.morphologyEx(img_inv, cv2.MORPH_CLOSE, kernel1)                              # Realiza una clausura morfológica
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))                                 # Kernel para aplicar una operación morfológica
    dilatacion = cv2.dilate(clausura, kernel2, iterations=1)                                    # Realiza una dilatación morfológica
    img_inv2 = cv2.bitwise_not(dilatacion)                                                      # Invierte los valores de la imagen binaria
    num_labels_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(img_inv2)             # Obtiene las componentes conectadas de la imagen

    for i in range(1, num_labels_c):                                                            # Recorre sobre los objetos detectados excepto el fondo
        # Obtiene las coordenadas del bounding box
        x1, y1, w1, h1 = stats_c[i, cv2.CC_STAT_LEFT], stats_c[i, cv2.CC_STAT_TOP], stats_c[i, cv2.CC_STAT_WIDTH], stats_c[i, cv2.CC_STAT_HEIGHT]
        area1 = stats_c[i, cv2.CC_STAT_AREA]                                                    # Obtiene el área
        componente1 = (labels_c == i).astype(np.uint8)                                          # Crea una imagen binaria de 8 bits
        # Obtiene el contorno del componente
        contornos1, _ = cv2.findContours(componente1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt1 = contornos1[0]                                                                    # Obtiene el primer contorno
        perimetro1 = cv2.arcLength(cnt1, True)                                                  # Calcula el perímetro del contorno
        rho1 = 4 * np.pi * area1 / (perimetro1**2)                                              # Calcula el factor de forma
        aspect_ratio1 = max(w1, h1) / min(w1, h1)                                               # Calcula la realción del aspecto
        
        if (0.65 < rho1 < 1 and aspect_ratio1 < 2 and area1 > 6000):                            # Aplica un filtro para detectar los capacitores
            capacitores_coord.append((x1, y1, w1, h1))                                          # Guarda las coordenadas de los capacitores en la lista
            capacitores_area.append(area1)                                                      # Guarda las áreas de los capacitores en la lista
            cv2.rectangle(img_resultado, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 4)          # Dibuja el rectángulo en la imagen "img_resultado"
            cv2.rectangle(img_procesada, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 4)          # Dibuja el rectángulo en la imagen "img_procesada"

    if (mostrar == True):
        plt.imshow(img_resultado, cmap='gray')
        plt.title('Placa de circuito impreso (PCB): Capacitores')
        plt.show()

    return img_procesada, capacitores_coord, capacitores_area

def detectar_chip(img, img_procesada, resistencias_coord, capacitores_coord, mostrar = False):
    img_resultado = img.copy()                                                                  # Crea una copia de la imagen original
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # Convierte la img de BGR a escala de grises
    th_out, img_th = cv2.threshold(img_gris, 55, 255, cv2.THRESH_BINARY_INV)                    # Aplica un umbral convirtiendo la imagen a binaria inversa

    # Rellena con negro las regiones donde estan las resistencias
    for x, y, w, h in resistencias_coord:                                                       
        cv2.rectangle(img_th, (x, y), (x + w, y + h), 0, -1)  
    # Rellena con negro las regiones donde estan los capacitores                    
    for x, y, w, h in capacitores_coord:
        margen = 90
        cv2.rectangle(img_th, (x - margen, y - margen), (x + w + margen, y + h + margen), 0, -1)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (45,45))                                # Kernel para aplicar una operación morfológica
    clausura = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel1)                               # Realiza una clausura morfológica
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))                                # Kernel para aplicar una operación morfológica
    apertura = cv2.morphologyEx(clausura, cv2.MORPH_OPEN, kernel2)                              # Realiza una apertura morfológica

    contornos, _ = cv2.findContours(apertura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       # Obtiene el contorno de la imagen
    for cnt in contornos:                                                                       # Recorre cada contorno detectado
        area = cv2.contourArea(cnt)                                                             # Obtiene el área de cada contorno
        if area > 160000:                                                                       # Filtro para detectar el chip según el tamaño del área
            x, y, w, h = cv2.boundingRect(cnt)                                                  # Obtiene las coordenadas del rectángulo delimitador
            # Dibuja un rectángulo en la imagen "img_resultado" y "img_procesada" 
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (255, 0, 0), 4)                
            cv2.rectangle(img_procesada, (x, y), (x + w, y + h), (255, 0, 0), 4)                
    
    if (mostrar == True):
        plt.imshow(img_resultado, cmap='gray')
        plt.title('Placa de circuito impreso (PCB): Chip')
        plt.show()

    return img_procesada

def imprimir_segmentaciones(img, mostrar = False):
    # Imprime la imagen original detectando los capacitores, resistencias y el chip
    if (mostrar == True):
        plt.imshow(img, cmap='gray')
        plt.title('Placa de circuito impreso (PCB)')
        plt.show()
    return

def clasificar_capacitores(img, coordenada, area, mostrar = False):
    img_resultado = img.copy()                                                                  # Crea una copia de la imagen original
    for (x, y, w, h), area in zip(coordenada, area):                                            # Recorre sobre las coordenadas y el área de los capacitores
        # Filtra los capacitores según el tamaño de su área
        # Dibuja un rectángulo de color y escribe un texto comentando a que categoría pertenece cada capacitor
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
    # Imprime por consola la cantidad de resistencias que hay en la placa
    if (mostrar == True):
        print(f"El total de resistencias detectadas en la placa son: {total_resistencias}")
    return

#------------------------------------------------------------------------------------------------------------------------------------
img = cargar_imagen(mostrar = False)
img_procesada, img_th, t_resistencias, resistencias_coord = detectar_resistencias(img, mostrar = False)
img_procesada2, capacitores_coord, capacitores_area = detectar_capacitores(img, img_th, img_procesada, resistencias_coord, mostrar = False)
img_procesada3 = detectar_chip(img, img_procesada2, resistencias_coord, capacitores_coord, mostrar = False)
imprimir_segmentaciones(img_procesada3, mostrar = False)
clasificar_capacitores(img, capacitores_coord, capacitores_area, mostrar = False)
contar_resistencias(t_resistencias, mostrar = False)
