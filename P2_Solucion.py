import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

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
        img_copia = img.copy()
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img2, 40, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
        clausura = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        contornos, _ = cv2.findContours(clausura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                cv2.circle(img_copia, (x, y), 5, (0, 255, 0), 5)
                puntos.append([int(x), int(y)]) 
                #print(f"Marcando punto en ({x},{y})")

        puntos_imagenes.append(puntos)

        if (mostrar == True):
            plt.imshow(img_copia, cmap='gray')
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
    guardar_img = []
    ruta_img = []

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
        guardar_img.append(img_rgb)
        ruta_img.append(ruta_imagen)

        if (mostrar == True):
                plt.imshow(img, cmap='gray')
                plt.title(f'Imagen Guardada N° {idx+1}: {ruta_imagen}')
                plt.show()

    return ruta_img, guardar_img

def quitar_fondo(imagenes, rutas, mostrar = False):
    resistencias = []
    r_rutas = []
    for idx, (img, ruta) in enumerate(zip(imagenes, rutas)):
        lista_rutas = ['R1_a_out.jpg', 'R2_a_out.jpg', 'R3_a_out.jpg', 'R4_a_out.jpg','R5_a_out.jpg', 
                       'R6_a_out.jpg', 'R7_a_out.jpg', 'R8_a_out.jpg', 'R9_a_out.jpg', 'R10_a_out.jpg']

        if (ruta in lista_rutas):
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
            lower_blue = np.array([100, 20, 20])     
            upper_blue = np.array([125, 255, 255])  

            mascara_fondo = cv2.inRange(img_hsv, lower_blue, upper_blue)
            mascara_objeto = cv2.bitwise_not(mascara_fondo)
            fondo_blanco = np.full_like(img, 255)
            objeto_sin_fondo = cv2.bitwise_and(img, img, mask = mascara_objeto)
            fondo = cv2.bitwise_and(fondo_blanco, fondo_blanco, mask=mascara_fondo)
            resultado = cv2.add(objeto_sin_fondo, fondo)
            img_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
            resistencias.append(img_rgb)
            r_rutas.append(ruta)

            if (mostrar == True):
                plt.imshow(img_rgb, cmap='gray')
                plt.title(f"Imagenes de resistencias: {ruta}")
                plt.show()

    return resistencias, r_rutas
    
def colores_bandas(img_hsv):

    resultado = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    valores_bajos = [
        np.array([0, 0, 0]),  # Negro
        np.array([0, 130, 0]),  # Marron
        np.array([0, 155, 100]),  # Rojo
        np.array([10, 160, 150]),  # Naranja
        np.array([20, 140, 0]),  # Amarillo
        np.array([35, 50, 50]),  # Verde
        np.array([130, 80, 70]),  # Violeta
        np.array([0, 35, 90])  # Blanco
    ]
    valores_altos = [
        np.array([179, 255, 40]),  # Negro 
        np.array([10, 255, 120]),  # Marron
        np.array([10, 255, 255]),  # Rojo
        np.array([15, 255, 255]),  # Naranja
        np.array([35, 255, 255]),  # Amarillo
        np.array([75, 255, 255]),  # Verde
        np.array([179, 255, 255]),  # Violeta
        np.array([15, 90, 190])     # Blanco
    ]
    nombre_color = ['Negro', 'Marron', 'Rojo', 'Naranja', 'Amarillo', 'Verde', 'Violeta', 'Blanco']
    bandas_detectadas = []

    for idx, (bajo, alto, nombre) in enumerate(zip(valores_bajos, valores_altos, nombre_color)):
        img_blur = cv2.GaussianBlur(img_hsv, (5, 5), 0)
        mascaras = cv2.inRange(img_blur, bajo, alto) 
        mascaras_2 = cv2.morphologyEx(mascaras, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        contornos, _ = cv2.findContours(mascaras_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            area = cv2.contourArea(c)
            if area > 500:
                x, y, w, h_c = cv2.boundingRect(c)
                bandas_detectadas.append((x, nombre))
                cv2.rectangle(resultado, (x, y), (x + w, y + h_c), (255, 0, 0), 2)
                cv2.putText(resultado, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return resultado, bandas_detectadas

def banda_tolerancia(imagenes, rutas, mostrar = False):
    for idx, (img, ruta) in enumerate(zip(imagenes, rutas)):
        img_original = img.copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_original)
        bandas = []
        print(num_labels)
        for i in range(1, num_labels):  # Ignorar fondo (etiqueta 0)
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            if area > 300:  # Filtro para evitar ruido
                bandas.append((int(cx), (x, y, w, h)))  # Guardar posición horizontal y bounding box
                cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #img = cv2.flip(img, 1)  # rotación horizontal

        if (mostrar == True):
            plt.imshow(img_original, cmap = 'gray')
            plt.title(f"Imagen N° {idx + 1}: {ruta}")
            plt.show()
    return

def detectar_bandas(imagenes, rutas, mostrar = False):
    barras_ordenadas = []

    for idx, (img, ruta) in enumerate(zip(imagenes, rutas)):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #Falta agreagar aca lo que hace banda tolerancia que gira la imagen dejando el dorado del lado izquierdo siempre

        imagen_marcada, bandas_detectadas = colores_bandas(img_hsv)
        bandas_detectadas.sort()

        for x, color in bandas_detectadas:
            barras_ordenadas.append(color)

        img_rgb = cv2.cvtColor(imagen_marcada, cv2.COLOR_BGR2RGB)
        if (mostrar == True):
            plt.imshow(img_rgb, cmap = 'gray')
            plt.title(f"Imagen N° {idx + 1}: {ruta}")
            plt.show()

    return barras_ordenadas


#------------------------------------------------------------------------------------------------------------------------------------
imagenes, nombre_img = cargar_imagenes(mostrar = False)
fondo_detectado = analizar_imagen(imagenes, nombre_img, mostrar = False)
imagen_transformada = transformar_imagen(imagenes, fondo_detectado, nombre_img, mostrar = False)
rutas_img, img_guardada = guardar_imagen(imagen_transformada, nombre_img, mostrar = False)
mod_img, rutas_mod_img = quitar_fondo(img_guardada, rutas_img, mostrar = False)
colores_img = detectar_bandas(mod_img, rutas_mod_img, mostrar = False)

probando = banda_tolerancia(mod_img, rutas_mod_img, mostrar = True)
