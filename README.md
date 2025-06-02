# Procesamiento de Imágenes (PDI)
## TRABAJO PRÁCTICO N° 2 - Año 2025

**Integrantes:**
- Martinez Dufour, Caterina.

## Instrucciones de instalación
### Clonar el repositorio
Para clonar el repositorio se debe ejecuta el siguiente comando en la terminal o consola:

```
git clone --depth 1 https://github.com/CaterinaMartinezD/TUIA-PDI-TP2-2025.git
```
### Crear un Entorno Virtual (venv)
Crea un entorno virtual llamado "venv" y lo activa:

```
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux
```

### Librerías
Una vez activado el entorno virtual, es necesario instalar los siguientes módulos:

```
pip install numpy 
pip install matplotlib 
pip install opencv-contrib-Python
pip install os
```

### Visualización de resultados:
Las funciones en los scripts tienen un parámetro llamado `mostrar` que por defecto está en `False`. 

- Si `mostrar = False`, el programa procesa las imágenes sin mostrar los resultados.
- Si `mostrar = True`, el programa mostrará la imágen/es o mensaje/s por consola los resultados.

Para ver las imágenes y resultados durante la ejecución hay que modificar las llamadas a las funciones en el script, cambiando `mostrar = False` por `mostrar = True`. 
