import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cargar_imagenes(mostrar = False):
    imagenes = []
    nombre_img = []
    carpeta_img = os.listdir('Resistencias')                                        # Obtiene la lista de archivos de la carpeta

    for ruta in carpeta_img:                                                        # Recorremos cada imagen de la carpeta
        ruta_completa = os.path.join('Resistencias', ruta)                          # Obtiene la ruta completa de la imagen
        img = cv2.imread(ruta_completa, cv2.IMREAD_COLOR)                           # Leemos la imagen en color (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                  # Convertimos la imagen de BGR a RGB
        imagenes.append(img)                                                        # Guardamos la imagen
        nombre_img.append(ruta)                                                     # Guardamos el nombre de la imagen

        if (mostrar == True):
            plt.imshow(img, cmap='gray')
            plt.title(ruta)
            plt.show()

    return imagenes, nombre_img

def analizar_imagen(imagenes, rutas, mostrar = False):
    puntos_imagenes = []

    for idx, (img, nombre) in enumerate(zip(imagenes, rutas)):                                  # Recorrer cada imagen con su nombre
        img_copia = img.copy()                                                                  # copiamos la imagen
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # Convierte la img de BGR a escala de grises
        th_out, img_th = cv2.threshold(img2, 40, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Transforma la img en una binaria inversa
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))                             # Kernel para aplicar una operación morfológica
        clausura = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel)                            # Aplica clausura a la imagen
        contornos, _ = cv2.findContours(clausura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # Detecta los contornos de la imagen
        mayor_area = 0
        mejor_aprox = None

        for c in contornos:                                                                     # Recorre los contornos encontrados
            area = cv2.contourArea(c)                                                           # Calcula el área del contorno
            if area > mayor_area:                                                               # Filtra según el tamaño del contorno
                approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)               # Aproxima el contorno con menos vértices
                if len(approx) == 4:                                                            # Verifica que el contorno tenga 4 vértices
                    mayor_area = area                                                           # Guarda el área hasta que encuentre una mejor
                    mejor_aprox = approx                                                        # Guarda la aproximación hasta que encuentre una mejor

        puntos = []                                                                             # Crea la lista donde se guardaran los puntos
        if mejor_aprox is not None:                                                             # Verifica que se encuentre un contorno válido
            for punto in mejor_aprox:                                                           # Recorre cada punto del contorno
                x, y = punto[0]                                                                 # Obtiene las coordenadas del punto
                cv2.circle(img_copia, (x, y), 5, (0, 255, 0), 5)                                # Marca un punto verde en cada vertice
                puntos.append([int(x), int(y)])                                                 # Obtiene las coordenadas x e y del punto

        puntos_imagenes.append(puntos)                                                          # Guarda los puntos en la lista

        if (mostrar == True):
            plt.imshow(img_copia, cmap='gray')
            plt.title(f'Imagen Analizada N° {idx+1}: {nombre}')
            plt.show()
    
    return puntos_imagenes

def transformar_imagen(imagenes, puntos_img, rutas, mostrar = False):
    imagenes_transformadas = []                                                                # Creación de la lista para guardar las img transformadas

    for idx, (coordenadas, img, ruta) in enumerate(zip(puntos_img, imagenes, rutas)):          
        pts_src = np.array(coordenadas)                                                        # Convierte la lista de coordenadas en una matriz
        suma = pts_src.sum(axis=1)                                                             # x + y  menor es sup-izq, mayor es inf-der
        resta = np.diff(pts_src, axis=1)                                                       # y - x  menor es sup-der, mayor es inf-izq
        ordenado = np.zeros((4, 2), dtype=np.float32)                                          # Crea una matriz vacía de 4 puntos (x,y)
        ordenado[0] = pts_src[np.argmin(suma)]                                                 # Superior izquierda
        ordenado[2] = pts_src[np.argmax(suma)]                                                 # Inferior derecha
        ordenado[1] = pts_src[np.argmin(resta)]                                                # Superior derecha
        ordenado[3] = pts_src[np.argmax(resta)]                                                # Inferior izquierda
        
        ancho = int(np.sqrt(np.sum(np.power(ordenado[0]-ordenado[1],2))))                      # Ancho: distancia horizontal (sup-izq al sup-der)
        alto = int(np.sqrt(np.sum(np.power(ordenado[1]-ordenado[2],2))))                       # Alto: distancia vertical (del sup-der al inf-der)
        pts_dst = np.array([[0,0], [ancho-1,0], [ancho-1,alto-1], [0,alto-1]])                 # Puntos: sup-izq | sup-der | inf-der | inf-izq
        h, status = cv2.findHomography(ordenado, pts_dst)                                      # Calcula la matriz de homografía
        im_dst = cv2.warpPerspective(img, h, (ancho,alto))                                     # Aplica la transformación en prespectiva

        imagenes_transformadas.append(im_dst)                                                  # Guarda la imagen transformada en la lista

        if (mostrar == True):
            plt.imshow(im_dst, cmap='gray')
            plt.title(f'Imagen Transformada N° {idx+1}: {ruta}')
            plt.show()

    return imagenes_transformadas

def guardar_imagen(imagenes, rutas, mostrar = False):
    guardar_img = []                                                                           # Se crea la lista donde se guardaran las imagenes
    ruta_img = []                                                                              # Se crea la lista donde se guardaran los nombres de las imagenes
    ruta_carpeta = "Resistencias_out"                                                          # Elegimos el nombre de la carpeta
    if not os.path.exists(ruta_carpeta):                                                       # Verifica si la carpeta existe
        os.makedirs(ruta_carpeta)                                                              # Si la carpeta no existe, se crea

    for idx, (img, ruta) in enumerate(zip(imagenes, rutas)):                                   # Recorre cada img con su nombre
        lista_ruta = ['R10_a.jpg', 'R10_b.jpg', 'R10_c.jpg', 'R10_d.jpg']                      

        #Renombra los nombres de cada imagen
        if ruta in lista_ruta:                                                                 
            ruta_imagen = ruta[:5] + "_out.jpg"                                                # Si esta en la lista, obtiene los primeros 5 datos
        else:
            ruta_imagen = ruta[:4] + "_out.jpg"                                                # Sino, obtiene los primeros 4 datos
    
        ruta_completa = os.path.join(ruta_carpeta, ruta_imagen)                                # Ruta completa donde se guardaran las imagenes
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                         # Convierte la imagen de BGR a RGB
        cv2.imwrite(ruta_completa, img_rgb)                                                    # Guarda la imagen en la ruta
        guardar_img.append(img_rgb)                                                            # Guarda la imagen la lista
        ruta_img.append(ruta_imagen)                                                           # Guarda el nombre de la imagen

        if (mostrar == True):
                plt.imshow(img, cmap='gray')
                plt.title(f'Imagen Guardada N° {idx+1}: {ruta_imagen}')
                plt.show()

    return ruta_img, guardar_img

def quitar_fondo(imagenes, rutas, mostrar = False):
    resistencias = []                                                                         # Se crea la lista que contendra las resistencias sin fondo
    r_rutas = []                                                                              # Se crea la lista que contendra los nombres de las imagenes

    for idx, (img, ruta) in enumerate(zip(imagenes, rutas)): 
        # Evalúa solamente las imagenes que se encuentran en la lista           
        lista_rutas = ['R1_a_out.jpg', 'R2_a_out.jpg', 'R3_a_out.jpg', 'R4_a_out.jpg',
                        'R5_a_out.jpg', 'R6_a_out.jpg', 'R7_a_out.jpg', 'R8_a_out.jpg', 
                        'R9_a_out.jpg', 'R10_a_out.jpg']

        if (ruta in lista_rutas):                                                              # Recorre las imagenes que se encuentran en la lista
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                                     # Convierte la imagen de BGR a HSV

            # Definimos un rango de color para detectar el fondo azul
            azul_bajo = np.array([100, 20, 20])                                                 
            azul_alto = np.array([125, 255, 255])                                              

            mascara_fondo = cv2.inRange(img_hsv, azul_bajo, azul_alto)                         # Generamos una máscara del fondo
            mascara_objeto = cv2.bitwise_not(mascara_fondo)                                    # Invierte los valores de la imagen
            fondo_blanco = np.full_like(img, 255)                                              # Crea un fondo blanco del mismo tamaño que la imagen
            objeto_sin_fondo = cv2.bitwise_and(img, img, mask = mascara_objeto)                # Elimina el fondo azul
            fondo = cv2.bitwise_and(fondo_blanco, fondo_blanco, mask=mascara_fondo)            # Obtiene el fondo blanco
            resultado = cv2.bitwise_or(objeto_sin_fondo, fondo)                                # Combina la imagen sin fondo con el fondo blanco
            img_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)                               # Convierte la imagen de BGR a RGB
            resistencias.append(img_rgb)                                                       # Guarda la imagen en la lista
            r_rutas.append(ruta)                                                               # Guarda el nombre de la imagen en la lista

            if (mostrar == True):
                plt.imshow(img_rgb, cmap='gray')
                plt.title(f"Imagenes de resistencias: {ruta}")
                plt.show()

    return resistencias, r_rutas
    
def colores_bandas(img_hsv):
    resultado = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)                                        # Convierte la imagen de HSV a RGB
    # Rangos HSV aproximados para detectar las bandas de color
    valores_bajos = [
        np.array([0, 0, 0]),                                                                    # Negro
        np.array([0, 130, 0]),                                                                  # Marron
        np.array([0, 155, 100]),                                                                # Rojo
        np.array([10, 160, 150]),                                                               # Naranja
        np.array([20, 140, 0]),                                                                 # Amarillo
        np.array([35, 50, 50]),                                                                 # Verde
        np.array([130, 80, 70]),                                                                # Violeta
        np.array([0, 35, 90])                                                                   # Blanco
    ]

    valores_altos = [
        np.array([179, 255, 40]),                                                               # Negro 
        np.array([10, 255, 120]),                                                               # Marron
        np.array([10, 255, 255]),                                                               # Rojo
        np.array([15, 255, 255]),                                                               # Naranja
        np.array([35, 255, 255]),                                                               # Amarillo
        np.array([75, 255, 255]),                                                               # Verde
        np.array([179, 255, 255]),                                                              # Violeta
        np.array([15, 90, 190])                                                                 # Blanco
    ]

    nombre_color = ['Negro', 'Marron', 'Rojo', 'Naranja', 'Amarillo', 'Verde', 'Violeta', 'Blanco']
    bandas_detectadas = []

    for idx, (bajo, alto, nombre) in enumerate(zip(valores_bajos, valores_altos, nombre_color)):
        mascaras = cv2.inRange(img_hsv, bajo, alto)                                             # Genera una mascara para cada color
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))                              # Kernel para aplicar una morfología
        mascaras_2 = cv2.morphologyEx(mascaras, cv2.MORPH_OPEN, kernel)                         # Aplica una apertura
        contornos, _ = cv2.findContours(mascaras_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encuentra los contornos de la imagen

        for c in contornos:                                                                     # Recorre los contornos encontrados
            area = cv2.contourArea(c)                                                           # Calcula el área de los contornos
            if area > 500:                                                                      # Filtra si el área es mayor a 500
                x, y, w, h_c = cv2.boundingRect(c)                                              # Obtiene las coordenadas del rectángulo delimitador
                bandas_detectadas.append((x, nombre))                                           # Guarda la coordenad x junto el nombre del color

                # Dibuja un rectángulo y escribe el nombre del color sobre la imagen
                cv2.rectangle(resultado, (x, y), (x + w, y + h_c), (0, 0, 255), 2)              
                cv2.putText(resultado, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return resultado, bandas_detectadas

def detectar_resistencia(img_hsv, ruta, mostrar = False):
    resultado = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)                                        # Convierte la imagen de HSV a RGB
    # Define el rango HSV para detectar la resistencia
    valor_bajo = np.array([10, 50, 50])
    valor_alto = np.array([25, 255, 255])                                                       
    mascara1 = cv2.inRange(img_hsv, valor_bajo, valor_alto)                                     # Genera una máscara binaria

    # Aplica la operación morfológica de clausura para rellenar huecos
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))                                
    mascara2 = cv2.morphologyEx(mascara1, cv2.MORPH_CLOSE, kernel1)       

    # Aplica la operación morfológica de apertura para eliminar ruido
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))                               
    mascara3 = cv2.morphologyEx(mascara2, cv2.MORPH_OPEN, kernel2)                              

    contornos, _ = cv2.findContours(mascara3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       # Detecta los contornos de la imagen
    resistencia = []                                                                            # Se genera una lista

    for c in contornos:                                                                         # Recorre los contornos encontrados
            x, y, w, h_c = cv2.boundingRect(c)                                                  # Obtiene las coordenadas del rectángulo delimitador
            resistencia.append((x, 'Resistencia'))                                              # Guarda la coordenad x junto "Resistencia"

            # Dibuja un rectángulo y escribe "Resistencia" sobre la imagen
            cv2.rectangle(resultado, (x, y), (x + w, y + h_c), (0, 0, 255), 2)                  
            cv2.putText(resultado, 'Resistencia', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if (mostrar == True):
        plt.imshow(resultado, cmap = 'gray')
        plt.title(f"Imagen {ruta}: ")
        plt.show()

    return resistencia

def corregir_img(b_tolerancia, b_colores, img, ruta, mostrar = False):

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)                                                  # Convierte la imagen HSV a RGB
    b_ordenada = sorted(b_colores)                                                              # Ordenala las bandas detectadas según las coordenadas x
    distancia_bandas = abs(b_ordenada[0][0] - b_ordenada[2][0])                                 # Calcula la distancia entre la primer y tercera banda
    distancia_resistencia = abs(b_ordenada[0][0] - b_tolerancia[0][0])                          # Calcula la distancia entre la resistencia y la primer banda

    if (distancia_resistencia >= distancia_bandas):                                             # Filtra si la distancia es mayor o igual a la dicha
        img = cv2.rotate(img, cv2.ROTATE_180)                                                   # Rota la imagen 180° grados

    if (mostrar == True):
        plt.imshow(img, cmap = 'gray')
        plt.title(f"Imagen corregida {ruta}")
        plt.show()

    return img

def detectar_bandas(imagenes, rutas, mostrar = False):
    bandas_ordenadas = []                                                                       # Se crea la lista para las bandas de color

    for idx, (img, ruta) in enumerate(zip(imagenes, rutas)):                                    # Recorre cada imagen con su nombre
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)                                          # Convierte la imagen de RGB a HSV
        img_marcada1, bandas_colores1 = colores_bandas(img_hsv)                                 # Llama a la función y devuelve una lista de colores de las bandas y su imagen
        resistencias = detectar_resistencia(img_hsv, ruta)                                      # Llama a la funcion y devuelve la coordenada de la resistencia
        img_orientada = corregir_img(resistencias, bandas_colores1, img_hsv, ruta)              # Llama a la función y devuelve lista de colores de las bandas y su imagen transformada

        img2 = cv2.cvtColor(img_orientada, cv2.COLOR_RGB2HSV)                                   # Convierte la imagen de RGB a HSV
        img_marcada2, bandas_colores2 = colores_bandas(img2)                                    # Vuelve a llamar a la función y detecta las bandas de colores
        bandas_ord = sorted(bandas_colores2)                                                    # Ordena las bandas de colores por su coordenada x
        colores_ordenados = []                                                                  # Se crea la lista para los colores ordenados
        for x, color in bandas_ord:                                                             # Recorre las bandas detectadas por su color y coordenada x
            colores_ordenados.append(color)                                                     # Almacena en una lista el color obtenido
        bandas_ordenadas.append((ruta, colores_ordenados))                                      # Almacena en una lista el nombre y las bandas detectadas ordenadas 

        if (mostrar == True):
            plt.imshow(img_marcada2, cmap = 'gray')
            plt.title(f"Imagen N° {idx + 1}: {ruta}")
            plt.show()

    return bandas_ordenadas

def valor_Ohms(barras_colores, mostrar = False):
    # Diccionario con el valor de las dos primeras bandas de la resistencia por color
    valor_color = {'Negro': 0, 'Marron': 1, 'Rojo': 2, 'Naranja': 3, 'Amarillo': 4,
                   'Verde': 5, 'Violeta': 7, 'Blanco': 9}
    
    # Diccionario con el multiplicador para la tercer banda de la resistencia por color
    valor_multiplicador = {'Negro': 1, 'Marron': 10, 'Rojo': 100, 'Naranja': 1000, 'Amarillo': 10000,
                           'Verde': 100000, 'Violeta': 10000000, 'Blanco': 1000000000}
    
    # Recorre por imagen y colores de la resistencia
    for img, colores in barras_colores:                                                         

        # Obtiene el valor de cada banda
        Banda1 = valor_color[colores[0]]
        Banda2 = valor_color[colores[1]]
        Banda3 = valor_multiplicador[colores[2]]

        # Calcula el valor de la resistencia
        valor_resistencia = (Banda1 * 10 + Banda2) * Banda3                                      

        if (mostrar == True):
            print("-------------------------------------")
            print(f"Resistencia: {img}")
            print(f"Banda 1: {colores[0]} - valor: {Banda1}")
            print(f"Banda 2: {colores[1]} - valor: {Banda2}")
            print(f"Banda 3: {colores[2]} - valor: {Banda3}")
            print(f"Valor total de la resistencia: {valor_resistencia} Ohms")
            print("-------------------------------------\n")
    return

#------------------------------------------------------------------------------------------------------------------------------------
imagenes, nombre_img = cargar_imagenes(mostrar = False)
fondo_detectado = analizar_imagen(imagenes, nombre_img, mostrar = False)
imagen_transformada = transformar_imagen(imagenes, fondo_detectado, nombre_img, mostrar = False)
rutas_img, img_guardada = guardar_imagen(imagen_transformada, nombre_img, mostrar = False)
mod_img, rutas_mod_img = quitar_fondo(img_guardada, rutas_img, mostrar = False)
bandas_color = detectar_bandas(mod_img, rutas_mod_img, mostrar = False)
valor_Ohms(bandas_color, mostrar = False)
