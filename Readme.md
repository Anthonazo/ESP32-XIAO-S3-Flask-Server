# Proyecto Flask: Transmisión de Video con ESP32-CAM

## Autores
[Anthony Moya](https://github.com/Anthonazo)
[Daniel Yanza](https://github.com/DanYC1503)


## Descripción general
Este proyecto es una aplicación web construida con **Flask** que transmite video desde un módulo ESP32-CAM. El proyecto permite a los usuarios ver transmisiones de video en vivo, interactuar con controles de reproducción y ajustar la configuración de la cámara directamente desde una interfaz web.
El trabajo cuenta con dos partes, la primera detalla la aplicacion de filtros y manipulacion de pixeles, usando operaciones en OPENCV, como sustraccion de fondo, ecualizacion, clahe, homomorfico, sal y pimienta ademas de suavizado con diferentes metodos, que tendran explicacion mas adelante, finalmente en la segunda parte se usan operaciones morfologicas para realzar el contraste de imagenes medicas. 

---
# Descripción de los Filtros de Procesamiento en `video_capture`

La función `video_capture` incluye varios filtros y técnicas de procesamiento de imágenes para mejorar y analizar los cuadros de video capturados. A continuación, se detallan los filtros y métodos utilizados:

## 1. Imagen en color original:
**Modo 0**: Muestra el cuadro de video en su color original sin procesamiento adicional.  

<div style="text-align: center;">
  <img src="https://drive.google.com/uc?id=1wCqkS6NDyDidoX86t-qcDjv6doypko4g" width="600"/>
</div>

## 2. Sustracción de fondo:
**Modo 1**: Utiliza un sustractor de fondo (`bg_subtractor`) permite detectar elementos en movimiento al comparar cada fotograma con
un fondo predefinido. Se utiliza el algoritmo de cv2.createBackgroundSubtractorMOG2, el cual aplica
una estimación adaptativa del fondo, facilitando la detección de cambios en la escena.

<div style="text-align: center;">
  <h2>Original<h2>
  <img src="https://drive.google.com/uc?id=1wCqkS6NDyDidoX86t-qcDjv6doypko4g" width="600"/>
</div>

<div style="text-align: center;">
  <h2>Sustracción de fondo<h2>
  <img src="https://drive.google.com/uc?id=1yVXKzvKY_dFCzoDpO6fUaBaJJePCZ2C0" width="600"/>
</div>


## 3. Equalización de histograma:
**Modo 2**: Aplica la equalización de histograma sobre la imagen en escala de grises para mejorar el contraste de una imagen en escala de grises distribuyendo de manera uniforme
sus intensidades de color. La función cv2.equalizeHist se usa para incrementar el contraste, revelando
detalles en áreas de la imagen donde los niveles de gris son similares.

<div style="text-align: center;">
  <h2>Original<h2>
  <img src="https://drive.google.com/uc?id=1wCqkS6NDyDidoX86t-qcDjv6doypko4g" width="600"/>
</div>

<div style="text-align: center;">
  <h2>Equalización de histograma<h2>
  <img src="https://drive.google.com/uc?id=1YVQXEbwrRnfMLTONyidT32VI2JS7HXyS" width="600"/>
</div>

## 4. Filtro CLAHE (Contrast Limited Adaptive Histogram Equalization):
**Modo 3**: Mejora el contraste localmente con el filtro CLAHE, que es una versión avanzada de la equalización de histograma que divide la imagen en pequeñas regiones y aplica ecualización de histograma en cada una. Esto permite aumentar el contraste en áreas locales sin afectar las áreas que ya tienen buen contraste. Se usa la función cv2.createCLAHE con un límite de recorte para controlar la amplificación del contraste.

<div style="text-align: center;">
  <h2>Original<h2>
  <img src="https://drive.google.com/uc?id=1wCqkS6NDyDidoX86t-qcDjv6doypko4g" width="600"/>
</div>

<div style="text-align: center;">
  <h2>Filtro CLAHE<h2>
  <img src="https://drive.google.com/uc?id=1ssirIox_oDnvf90s-Zg32nB_az1j61Sn" width="600"/>
</div>

## 5. Filtro Investigado: Homomórfico :
**Modo 4**: El filtro homomórfico es una técnica de procesamiento de imágenes que mejora el contraste y la visibilidad de detalles al reducir las variaciones de iluminación. Funciona en el dominio de frecuencia y utiliza la transformada de Fourier para separar los componentes de baja frecuencia (iluminación) y alta frecuencia (detalles). Tras convertir la imagen a una relación aditiva mediante una transformación logarítmica, se aplica un filtro paso-alto que atenúa las bajas frecuencias y resalta las altas, ayudando a reducir sombras y realzar detalles. Es especialmente útil en imágenes con iluminación no uniforme, como en radiografías, fotografía bajo condiciones difíciles y reconocimiento de texto en documentos antiguos.

<div style="text-align: center;">
  <h2>Original<h2>
  <img src="https://drive.google.com/uc?id=1wCqkS6NDyDidoX86t-qcDjv6doypko4g" width="600"/>
</div>

<div style="text-align: center;">
  <h2>Filtro homomórfico<h2>
  <img src="https://drive.google.com/uc?id=1jXGKT7C0qDFnSvwx0weohIWQQtkP2t14" width="600"/>
</div>

## 6. Ruido sal y pimienta y filtros de suavizado:

**Modo 5**: Este modo agrega ruido sal y pimienta a la imagen original, lo que simula interferencias en la imagen y permite evaluar la efectividad de diferentes filtros de suavizado para eliminar dicho ruido. Los filtros de suavizado aplicados son:

- **Filtro de mediana**: Este filtro calcula la mediana de los píxeles en una ventana de tamaño especificado, en lugar de calcular el promedio. Es ideal para reducir el ruido de sal y pimienta sin difuminar tanto los bordes de la imagen como lo hace el filtro de promedio.

- **Filtro gaussiano**: Aplica un suavizado basado en una distribución gaussiana (o normal), donde los píxeles cercanos al valor central reciben más peso que los píxeles distantes. Esto ayuda a reducir el ruido general de la imagen al difuminar gradualmente los cambios abruptos en los valores de píxeles.

- **Filtro de la media**: Utiliza un filtro de media para suavizar la imagen. Este filtro promedia los valores de los píxeles dentro de una vecindad definida, proporcionando un suavizado simple, pero efectivo, para reducir el ruido.

<div style="text-align: center;">
  <h2>Original</h2>
  <img src="https://drive.google.com/uc?id=1wCqkS6NDyDidoX86t-qcDjv6doypko4g" width="600"/>
</div>
    
<div style="text-align: center;">
  <h2>Desenfoque simple</h2>
  <img src="https://drive.google.com/uc?id=1N_EHsroRej94aO5tJKlwAXkB4Wn9twzb" width="600"/>
</div>


## 7. Detección de bordes:
La detección de bordes se utiliza para resaltar las áreas en una imagen donde ocurren transiciones abruptas en intensidad, lo cual es útil para identificar objetos o estructuras en la imagen. En este modo, se aplican diferentes técnicas de detección de bordes:

**Modo 6**: Implementa dos técnicas de detección de bordes, cada una con sus características particulares:

- **Detección de bordes con Media**: Este método utiliza un filtro de media (promedio) para suavizar la imagen y luego aplica un algoritmo de detección de bordes, como el de Canny o Sobel. El filtro de media ayuda a reducir el ruido antes de detectar los bordes.

<div style="text-align: center;">
  <h2>Original<h2>
  <img src="https://drive.google.com/uc?id=1wCqkS6NDyDidoX86t-qcDjv6doypko4g" width="600"/>
</div>

<div style="text-align: center;">
  <h2>Detección de bordes con Media<h2>
  <img src="https://drive.google.com/uc?id=1gFVvnBH9zxThwocgkDVr4N7Q4xF0fMlF" width="600"/>
</div>

- **Detección de bordes con Gaussiano**: Utiliza un filtro gaussiano antes de aplicar la detección de bordes. Este filtro difumina la imagen suavemente, lo que puede mejorar la precisión de la detección de bordes al eliminar pequeños detalles irrelevantes.

<div style="text-align: center;">
  <h2>Original<h2>
  <img src="https://drive.google.com/uc?id=1wCqkS6NDyDidoX86t-qcDjv6doypko4g" width="600"/>
</div>

<div style="text-align: center;">
  <h2>Detección de bordes con Gaussiano<h2>
  <img src="https://drive.google.com/uc?id=1EVBRd2IYq3nfjF_R9PlvAcEB-lS3qxxt" width="600"/>
</div>

- **Detección de bordes con blur**: En este caso, se utiliza un desenfoque general (blur) de la imagen antes de detectar los bordes. Este enfoque ayuda a reducir los detalles finos en la imagen, enfocándose más en los contornos más prominentes.

<div style="text-align: center;">
  <h2>Original<h2>
  <img src="https://drive.google.com/uc?id=1wCqkS6NDyDidoX86t-qcDjv6doypko4g" width="600"/>
</div>

<div style="text-align: center;">
  <h2>Detección de bordes con blur<h2>
  <img src="https://drive.google.com/uc?id=100FWNIRThqivm5FwrXc12lb-H1LzaXjj" width="600"/>
</div>


Estos filtros permiten aplicar una variedad de técnicas de mejora y análisis de imágenes para obtener diferentes perspectivas de la transmisión de video en tiempo real, proporcionando a los usuarios la capacidad de observar detalles específicos o realizar análisis visuales en el flujo de video.

# Parte 2 

<div style="text-align: center;">
  <h2>Imagen Medica Con Transformaciones<h2>
  <img src="https://drive.google.com/uc?export=view&id=1OwclQG9Pxjo1-iupaLAxnwW7utu9XIWQ" alt="parte2" width="1000"/>
</div>

En este apartado estan las operaciones morfologicas, con diferentes tamaños de kernels, entontrara la carpeta imgs que sera donde estan las imagenes que fueron procesadas y para cada imagen se realizo una nueva carpeta donde tendra el mejor resultado de la apliccion de las operaciones de realce de contraste, ademas de los filtros de tophat, blackhat, dilatacion y erosion aplicados para cada imagen.

```
project_root/
|-- static/
|   |-- imgs/
|   |-- chest1/
|   |-- |--best_contrast_enhanced/
        |--black_hat/
        |--contrast_enhanced/
        |--dilation/
        |--erosion/
        |--top_hat/
y asi para chest2 y chest3
```

Aplicar las operaciones morfológicas en imágenes permite realzar detalles, mejorar el contraste y destacar características importantes, especialmente en contextos como imágenes médicas. A continuación, les explicamos cada operación realizada:

### a) Erosión
La erosión es una operación morfológica que reduce el tamaño de los objetos en la imagen, "erosionando" sus bordes. Se utiliza para eliminar ruido y separar elementos cercanos. Se aplica utilizando un kernel (o máscara) que define el área de aplicación. Al aplicar una erosión con diferentes tamaños de máscaras, como una de 37x37, se puede observar cómo se reducen los detalles finos y se alisan los bordes.

### b) Dilatación
La dilatación es el proceso inverso a la erosión; expande los objetos de la imagen y rellena huecos y espacios pequeños. Esto es útil para conectar áreas disjuntas y aumentar la visibilidad de estructuras. Usar una máscara grande, como de 37x37, amplía de forma significativa las áreas brillantes y refuerza los contornos.

### c) Top Hat
La operación Top Hat resalta las áreas claras en un fondo más oscuro. Es útil para detectar pequeñas estructuras brillantes y zonas de alta intensidad.

**Resultados en la imagen:** Se puede observar cómo las estructuras internas de la radiografía, como los bordes de los pulmones y la columna, se destacan al aplicar esta operación.

### d) Black Hat
La operación Black Hat resalta las áreas más oscuras en un fondo más brillante. Permite observar detalles como sombras o estructuras de baja intensidad en la imagen original.

**Resultados en la imagen:**""** En cada fila, la operación muestra los contornos oscuros y zonas de bajo contraste, permitiendo identificar detalles finos que no son visibles en la imagen original.

### e) Imagen Original + (Top Hat – Black Hat)
La combinación de la imagen original con el resultado de Top Hat menos Black Hat ofrece una imagen con un contraste mejorado. Esta técnica permite destacar tanto las áreas oscuras como las claras, equilibrando la iluminación y haciendo más visibles los detalles.

**Resultados en la imagen:** Las imágenes "Mejorada" muestran un balance de contraste que resalta tanto las áreas más claras como las más oscuras, logrando una visualización más nítida de las estructuras internas de la radiografía.

##Tamaños de mascara utilizadas
- **Kernel 34x34:** Permite un ajuste medio que resalta detalles sin perder demasiado contraste.
- **Kernel 50x50:** Este tamaño de máscara resalta más las áreas grandes, afectando el contraste general de la imagen.
- **Kernel 38x38:** Un tamaño intermedio que equilibra la mejora de detalles y el contraste global.

## Estructura del proyecto
```
project_root/
|-- app.py
|-- static/
|   |-- imgs/
|   |-- chest1/
|   |-- chest2/
|   |-- chest3/
|-- templates/
|   |-- index.html
|-- README.md
```

## Instalación y configuración
1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/Anthonazo/ESP32-XIAO-S3-Flask-Server.git
   cd esp32-cam-flask-streaming
   ```

2. **Crear un entorno virtual e instalar dependencias necesarias**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Windows usa `venv\\Scripts\\activate`
   ```

3. **Ejecutar la aplicación Flask**:
   ```bash
   flask run
   ```

## Cómo usar

1. **Conecta tu módulo ESP32-CAM a la misma red que el servidor.**
2. **Ajusta la URL de transmisión en app.py para que coincida con la dirección IP de tu ESP32-CAM.**
3. **Accede a la interfaz web y comienza a ver la transmisión de video en vivo.**


## Conclusion

La manipulación de píxeles y el mejoramiento de imágenes mediante técnicas avanzadas son fundamentales para resaltar detalles y mejorar la calidad visual en análisis de imágenes complejas. Al aplicar métodos como la ecualización de histograma y el filtro CLAHE (Contrast Limited Adaptive Histogram Equalization), se puede aumentar el contraste y resaltar áreas de interés, especialmente en imágenes con bajo contraste o en entornos con iluminación no uniforme. La ecualización mejora la distribución de los tonos de gris, y el CLAHE ajusta el contraste de manera local, lo cual es especialmente útil en imágenes médicas o de precisión donde es necesario controlar el contraste sin perder detalles en áreas críticas.

Además, la implementación de ruido de sal y pimienta y los filtros de suavizado (como el filtro de mediana o el desenfoque gaussiano) permiten simular y gestionar la interferencia de ruido en la imagen, aumentando así la robustez del procesamiento frente a variaciones indeseadas. Técnicas avanzadas de morfología matemática, como los filtros Top-Hat y Black-Hat, son también esenciales en imágenes de alto detalle, especialmente en el ámbito médico, donde estas transformaciones ayudan a resaltar detalles específicos como estructuras y texturas en un fondo heterogéneo.
