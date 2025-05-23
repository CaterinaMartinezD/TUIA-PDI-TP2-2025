import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cargar_imagenes(mostrar = False):
    imagenes = []
    nombre_img = []
    carpeta_img = os.listdir('Resistencias')

    for ruta in carpeta_img:
        ruta_completa = os.path.join('Resistencias', ruta)
        img = cv2.imread(ruta_completa, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imagenes.append(img)
        nombre_img.append(ruta)

        if (mostrar == True):
            plt.imshow(img, cmap='gray')
            plt.title(ruta)
            plt.show()

    return imagenes, nombre_img

def analizar_imagen(imagenes, rutas, mostrar = False):
    puntos_imagenes = []

    for idx, (img, nombre) in enumerate(zip(imagenes, rutas)):
        img_original = img.copy()
        B, G, R = cv2.split(img)
        bordes = cv2.Canny(B, 40, 100)
        _, img = cv2.threshold(bordes, 40, 255, cv2.THRESH_BINARY)
        edges = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, np.ones((5,5), np.uint8))
        contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mayor_area = 0
        mejor_aprox = None

        for c in contornos:
            area = cv2.contourArea(c)  
            if area > mayor_area:
                approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
                if len(approx) == 4:
                    mayor_area = area
                    mejor_aprox = approx

        puntos = []
        if mejor_aprox is not None:
            for punto in mejor_aprox:
                x, y = punto[0]
                cv2.circle(img_original, (x, y), 5, (0, 255, 0), 5)
                puntos.append([int(x), int(y)]) 
                #print(f"Marcando punto en ({x},{y})")

        puntos_imagenes.append(puntos)
        if (mostrar == True):
            plt.imshow(img_original, cmap='gray')
            plt.title(f'Imagen Analizada N° {idx+1}: {nombre}')
            plt.show()
    
    return puntos_imagenes

def transformar_imagen(imagenes, puntos_img, rutas, mostrar = False):
    imagenes_transformadas = [] 

    for idx, (coordenadas, img, ruta) in enumerate(zip(puntos_img, imagenes, rutas)):
        pts_src = np.array(coordenadas)   #coordenadas desordenadas
        suma = pts_src.sum(axis=1)        # x + y  menor es sup-izq, mayor es inf-der
        resta = np.diff(pts_src, axis=1)  # y - x  menor es sup-der, mayor es inf-izq
        ordenado = np.zeros((4, 2), dtype=np.float32)
        # sup-izq | sup-der | inf-der | inf-izq
        ordenado[0] = pts_src[np.argmin(suma)]   # superior izquierda
        ordenado[2] = pts_src[np.argmax(suma)]   # inferior derecha
        ordenado[1] = pts_src[np.argmin(resta)]  # superior derecha
        ordenado[3] = pts_src[np.argmax(resta)]  # inferior izquierda
        
        ancho = int(np.sqrt(np.sum(np.power(ordenado[0]-ordenado[1],2))))
        alto = int(np.sqrt(np.sum(np.power(ordenado[1]-ordenado[2],2))))
        pts_dst = np.array([[0,0], [ancho-1,0], [ancho-1,alto-1], [0,alto-1]])  # sup-izq | sup-der | inf-der | inf-izq
        h, status = cv2.findHomography(ordenado, pts_dst)
        im_dst = cv2.warpPerspective(img, h, (ancho,alto))

        imagenes_transformadas.append(im_dst)

        if (mostrar == True):
            plt.imshow(im_dst, cmap='gray')
            plt.title(f'Imagen Transformada N° {idx+1}: {ruta}')
            plt.show()

    return imagenes_transformadas

def guardar_imagen(imagenes, rutas, mostrar = False):
    ruta_carpeta = "Resistencias_out"  # Reemplaza con la ruta deseada
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)

    for idx, (img, ruta) in enumerate(zip(imagenes, rutas)):
        lista_ruta = ['R10_a.jpg', 'R10_b.jpg', 'R10_c.jpg', 'R10_d.jpg']
        if ruta in lista_ruta:
            ruta_imagen = ruta[:5] + "_out.jpg" 
        else:
            ruta_imagen = ruta[:4] + "_out.jpg"
    
        ruta_completa = os.path.join(ruta_carpeta, ruta_imagen)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(ruta_completa, img_rgb)

        if (mostrar == True):
                plt.imshow(img, cmap='gray')
                plt.title(f'Imagen Guardada N° {idx+1}: {ruta_imagen}')
                plt.show()
    return 

#------------------------------------------------------------------------------------------------------------------------------------
imagenes, nombre_img = cargar_imagenes(mostrar = False)
fondo_detectado = analizar_imagen(imagenes, nombre_img, mostrar = False)
imagen_transformada = transformar_imagen(imagenes, fondo_detectado, nombre_img, mostrar = False)
#guardar_imagen(imagen_transformada, nombre_img, mostrar = False)
