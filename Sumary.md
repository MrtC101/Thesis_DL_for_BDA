# UnetSiames

## Summary
- Image processing
    - 16 256x256
    - 4 256x256 con Random crop
- Normalización: Normalización estándar
- Data augmentation con (Random flip y Random crop) durante entrenamiento on the fly
- Dos UNet siamesas con un 3er decodificador para clasificación.
- 1500 epochs
- 0.0005 learning rate
- 32 de batch size 
- loss function "CrossEntropyLoss"
- ReduceLROnPlateau con paciencia de 1200 epochs
- Métricas "f1-score"
- Optimizador "Adam"
- weights for building segmentation pixel class. [1,15]
- weights for building damage class [1,35,70,150,120]
- weights for loss functions (3 salidas) [0,0,1]
- weights initializer con xavier_uniform_.

### pipeline sequence
Preprocesamiento:
1. Crear la imagen "target" de los desastres, por cada conjunto train,test,hold y tier3 aparte.
2. Tomar todos los conjuntos train,test,hold y tier3 y hace 2 splits de los path a los archivos de imagen, uno de 80%/10%/10% y otro de 90%/10%/0%.
3. Realiza un recortado de todas las imágenes en 16 "parches" de 256x256 y además genera 4 mediante recortes aleatorios (*random crop*) del mismo tamaño. Es decir, genera 20 recortes.
4. Calcula la media y la desviación estándar de los colores RGB de todas las imágenes del dataset.(Se usa para normalizar durante el entrenamiento)
5. Se crean n conjuntos de 5 un "shards" para los recortes de imágenes pre, post, target, class mask, y la imagen original dividiendo en n partes el conjunto de entrenamiento y validación.
Entrenamiento:
1. Se cargan los shards
2. Se define la arquitectura del modelo que es una UNet siamesa que da como salida 3 mascaras.
3. Se itera la cantidad de épocas especificadas, 
    1. Por cada época se realiza el forwardpropagation y el backpropagation según la cantidad de pasos en función del batch size definido. Al final de cada step se calculan 2 matrices de confusión una para evaluar la mascara de segmentación binaria y otra para evaluar la clasificación de daños.
    2. Se calcula la métrica  f1-score en función de la matriz de confusión.
    3. Se realiza la validación. Se repiten todos los paso que se realizo durante el entrenamiento, pero con un conjunto de datos diferente y sin actualizar los pesos del modelo.
Inferencia:
1. Carga los datos de test
2. Carga el modelo y su checkpoint 
3. llama la función `validate()`
4. Computa la métrica "f1-score" para la salida de la mascara de segmentación, la mascara de clase y la clasificación de daños.
5. Almacena todos los resultados en un csv.

# Script sequence
## Preprocessing
1. ### feature engineering  
    `create_label_mask.py` 
    - Genera las imágenes target para pre y post desastre a partir del atributo `'xy'` de los *.json* que existen por cada desastre. Admite 2 parámetros: *-b* que indica el grosor de los bordes de los polígonos y *-o* que indica si se quieren re generar aquellos incluidos en el xBD.
2. ### Data spiting  
    `class_distribution_and_splits.ipynb`
    - Cuenta la cantidad de polígonos y su area utilizando los archivos *.json* que están incluidos en xBD para todas las imágenes pre y post desastre. 
    - Genera como salida el archivo `areas_and_counts_by_disaster.json`.
    Hace un plot de la cantidad por tipo de daño "no-subtype","no-damage","minor-damage","mayor-damage","destroyed" y "un-classified".
    - Crea splits con diferentes proporciones por area de desastre: 
        - Solo desastres relacionados con viento 80%/10%/10% `wind_disaster_splits.json`
        - Todos los desastres excepto un subconjunto LOO (flooding) 80%/20%/0% `LOO_(subconjunto_eliminado)_subset_disaster_splits.json`
        - Repetición pero para todos los desastres 80%/20%/0%. La salida son varios archivos donde se excluye un desastre.
        - Un split de 90%/10%/0% de los path a las imágenes 1024x1024 `final_mdl_all_disaster_splits.json`
        - Un split de 80%/10%/10% de los path a las imágenes 1024x1024 `all_disaster_splits.json`  
3. ### Data resizing
    `make_smaller_tiles.py`  
     - Define una clase que hereda de `torch.utils.data.Dataset` llamada **SliceDataset**. Esta clase define 2 métodos importantes.
        - `slice_tile()` realiza un recorte de la imagen de 1024x1024 a 20 imágenes 256x256 donde 4 de ellas son obtenidas mediante random crop y las otras 16 son los trozos correspondientes a un grillado de la imagen.
        - `__getitem__()` Se encarga de devolver las 4 imágenes (pre,post,target,class) para cada desastre. Pero, llama a **slice_tile()** y almacena los recortes generados a los cuales se refieren como "chips" o "patches". 
    - En función de los archivos *.json* creados en el paso anterior y crea 2 nuevos .json `all_disaster_splits_sliced_img_augmented_20.json` y `final_mdl_all_disaster_splits_sliced_img_augmented_20.json` a los que les añade los nuevos path a las carpetas que contendrán los recortes generados al iterar sobre el  `torch.utils.data.Dataloader` inicializado con las clases `SliceDataset`.
    - Utilizando 3 threads para cada dataset itera sobre los `DataLoaders` aplicándoles el método estático `iterate_and_slice()`. **(El slicing se realiza on the fly cuando se accede a una imagen de un dataset)**  
    
    `compute_mean_stddev.ipynb`
    - Crea una clase llamada `AverageMeter` utilizada para contar, sumar y calcular el promedio de una serie de números. 
    - Utiliza `AverageMeter` para calcula la media y desviación estándar del color rojo, verde y azul de todas las imágenes de xBD.
    - Guarda los resultados en un `all_disaster_mean_stddev_tiles_0_1.json`.
    
    `make_data_shards.py` 
    - Se define la clase `DisasterDataset` que hereda de `torch.utils.data.Dataset`
    - A partir de los paths en `final_mdl_all_disaster_splits_sliced_img_augmented_20.json` se crea un `DisasterDataset` y un `torch.utils.data.DataLoader` para los conjuntos "*train*" y "*validation*".
    - Solo se crean shards para *train* y *validation*, no para *test*. Un shard es un `np.array` que contiene matrices de dimension (256,256,3) resultantes de la conversión de las imágenes jpeg a `np.array`. Tanto para train como para test se llama a las funciones: `create_shards()` y `save_shards()`.
    - `create_shards()` Divide los indices del conjunto de datos en n partes de igual tamaño. A partir de estos indices genera 5 shards utilizando la función `np.stack()` sobre una lista que contiene las imágenes como `np.array`. Crea un shard para imágenes pre, post, target, class mask y uno para la imagen sin recortar.
    - `save_shards()` genera los archivos *.npy* que contienen los shards generados.  
    **(Los archivos *.npy* no son eficientes en termino de espacio, pero con este formato pueden saber cuanto ocupará cada shard en memoria y se ahorran el tiempo de conversión de imagen a numpy.)** 

## Model training
4. ### Training
    `dataset_shard_load`
    - Define otra clase nueva con el mismo nombre **DisasterDataset**. 
    - `__init__()`: Carga un conjunto de 5 shards, el diccionario con la desviación estándar y dos booleanos que indican si se aplica normalización y/o transformaciones **(Data augmentation on the fly)**.
    - `__getitem__()`: 
        - Si esta habilitado, se aplican las transformaciones "Random horizontal flip" y "Random vertical flip" sobre las 4 imagen que componen el elemento **i** del dataset.
        - Si esta habilitado, se aplica una transformación estándar con `torchvision.transform.Normalize()` sobre los "chips" pre y post desastre. Sino se dividen por 255.
        - Se devuelve un diccionario que contiene las 4 imágenes "chips" del elemento *i* y las 2 imágenes pre y post originales.  
    
    `end_to_end_Siam_UNet.py`
    - Se define una clase `SiamUnet` que hereda de `torch.nn.Module`
        - `_block()`: Define el bloque básico utilizado en toda la red
            - Todas las capas se definen con un método llamado `_block()`. El método recibe como parámetros, la cantidad de canales entrante, la cantidad de canales de salida y el nombre del bloque. 
            - Cada bloque está conformado por la siguiente secuencia:
                1. Capa convencional 2D padding 1 kernel 3
                2. BatchNormalization 2D
                3. ReLU
                4. Capa convencional 2D padding 1 kernel 3
                5. BatchNormalization 2D
                6. ReLU
        - `__init__()`: En el contractor se definen todas las capas y bloques que serán utilizados en la red, concretamente aquellos que forman los codificador, el bottleneck y un decodificador de la red. 
            - El codificador está compuesto por una sucesión de bloques construidos con `_block()` y capas `torch.nn.MaxPool2d`.
            - El bottleneck es un solo bloque `_block()`.
            - El decodificador está compuesto por una sucesión de `_block()` y capas `torch.nn.ConvTranspose2d`.
            - Una sola capa de salida con kernel 1 de tipo `torch.nn.Conv2d`
            - Se define un decodificador más que es utilizado para la clasificación
        - `foward()`: Esté método se reescribe con el fin de que relacionar las capas y la secuencia que realizan los datos del modelo.
            - Se define una UNet siamesa donde ingresa la imagen pre y post desastre y como salida genera dos mascaras de segmentación semántica.
            - Se define un decodificador para clasificación que tendrá como salida la mascara de clases.
                La primera capa del decodificador de clasificación se genera haciendo la resta del bottleneck 1 al bottleneck 2. Las capas siguientes realizan la siguiente secuencia que se repite hasta que se escala a 1024x1024:
                1. upconv2D
                2. diff = enc2-enc1
                3. concatenación con dec anterior
                4. bloqueConv
            - Como salida retorna la máscara de segmentación de la imagen pre desastre, la imagen post desastre y la mascara de clases generadas por cada decodificador. 
    
    `train.py`  
    - `main()`
        1. Carga los datos del conjunto de train y val en `DisasterDataset`'s, y los `DataLoader`'s, crea `DataFrame`'s para guardar métricas.
        2. Crea 4 directorios para guardar "/checkpoints", "/logs", "/evals" y "/configs".
        3. Crea el modelo especificado el dispositivo donde va a correr ("cpu" o "cuda").
        4. Si existe un archivo en el directorio de checkpoints se cargan los pesos con `reinitialize_Siamese()` y crea un nuevo optimizador Adam que puede tener diferente **learning rate**.
        5. Crea un scheduler con `torch.optim.lr_scheduler.ReduceLROnPlateau` con paciencia de 2000 épocas. Este método modifica el **learning rate** del modelo si la función de perdida no tienen progreso después de cierta cantidad de épocas.
        8. Define 3 funciones de perdida (una para cada salida) con `torch.nn.CrossEntropyLoss`y le pasa los ***pesos para cada clase***.
        9. Por cada época se realiza la siguiente secuencia
            1. `train()`
            2. `compute_eval_metrics()` y calcula f1-score para la mascara de clases
            3. `compute_eval_metrics()` y calcula f1-score para la mascara de segmentación
            4. `validation()`
            5. Añade un paso al scheduler.
            6. `compute_eval_metrics()` y calcula f1-score para la mascara de clases
            7. `compute_eval_metrics()` y calcula f1-score para la mascara de segmentación
            8. Almacena la época con mejor f1-score y crea un checkpoint 
    - `train()`
        - Crea `DataFrame` para almacenar una matriz de confusión para daño y edificios, crea 3 objetos de tipo `AverageMeter` para contar y calcular el promedio de las función de perdida para cada salida. 
        - Itera por cada batch, realizando una conversión del `torch.Tensor` con `torch.Tensor.to(device)` al dispositivo que este ejecutando el código.
        - Calcula con `scores = model(x_pre,x_post)`(**forwardpropagation**)
        - Aplica mediante operación de matrices una softmax a todas la imágenes de salida.
        - Calcula  la función de perdida para cada salida `CrossEntropyLoss` y la almacena.**La salida de la segmentación semántica de la imagen post-desastre se valida con la mascara de segmentación pre-desastre**
        - `loss.backward()`y`optimizer.step()`
        - Computa la predicción de las 3 salidas, luego construye 2 matrices de confusion una para detección de edificios y otra para detección de daños. ***Ignora la salida de la predicción del post-procesamiento***
        - Crea 2 matrices de confusión con `compute_confusion_mtrx()` una para la mascara de segmentación y otra para la mascara de clasificación 
    - `validation()`
        - El inicio es similar a train(), pero se llama a `torch.no_grad()` para realizar inferencia más eficientemente.
        - `model.eval()` y `scores = model(x_pre,x_post)`
        - Aplica `torch.nn.softmax` a todas las salidas.
        - Calcula la función de perdida para cada salida.
        - Crea la matriz de confusión para daño y edificios.

## Model inference & evaluation
5. ### evaluation
    `eval_building_level.py`
    - `_evaluate_tile()` Está función implementa un método para poder calcular TP,FP y FN a nivel de edificio, es decir que se hace una evaluación object-level. El problema es que varios edificios quedan juntos por lo que la matriz de confusión realiza un subestimación.
    Parámetros:
        - `pred_polygons_and_class` Una lista de polígonos shapely predichos
        - `label_polygons_and_class` Una lista de polígonos shapely de la class mask
        - "allowed_classes" las clases a evaluar
        - iou_threshold: limite inferior de la métrica IoU para la cual se va a considerar un polígono predicho como true positive
        - La salida son 2 listas que se usan para formar una matriz de confusión.
    - Las 2 listas de `shapely.polygon` que representan los edificios de la predicción y los de la mascara de clase se pueden obtener con los siguientes métodos:
        - `get_label_and_pred_polygons_for_tile_json_input`() Extrae los polygons de la predicción creando un ground truth con np.where y utiliza raster.features.shapes para extraer cada edificios (pueden quedar pegados). SOlo toma aquellos clusters de pixeles que tengan un area mayor a 20. Utiliza majority vote para considerar la clasificación del polígono con la clase que se le ha asignado a la mayor cantidad de pixeles del polígono. Devuelve 2 listas, con tuplas (polygon,class) una lista con las predicciones y otra lista con el target.
        - `get_label_and_pred_polygons_for_tile_mask_input`() Realiza el mismo procedimiento que el método anterior pero para 2 imágenes.
    
    `DisasterDataset`
    - Aplica las transformaciones en las imágenes de test. (Random resize,crop,horizontal flip,vertical flip)
    
    `inference.py`
    - Carga los datos de test
    - Carga el modelo y su checkpoint 
    - llama la función `validate()`
    - Computa las métricas para los niveles de daño y para la detección de edificios.
    
    `validate()`
    - con `model.eval()` y `torch.no_grad()` realiza la inferencia del modelo sobre los datos de test.
    - Aplica softmax a la salida.
    - Guarda las predicciones en disco.
    - Utiliza la `_evaluate_tile()` para crear una matriz confusión.
    - La salida son 3 matrices de confusión:
        1. `confusion_mtrx_df_val_dmg` Está matriz refleja el desempeño del modelo para inferir niveles de daño. (pixel-level classification)
        2. `confusion_mtrx_df_val_bld` Está matriz refleja el desempeño del modelo para inferir edificios y fondo. (pixel-level detection)
        3. `confusion_mtrx_df_val_dmg_building_level` Está matriz refleja el desempeño del modelo para inferir niveles de daño en edificios.(object-level classification)

## Mejoras que haría
### Mejoras en el código
1. Las clases Dataset se encuentran re definidas varias veces con algunas diferencias. (Crearía una sola clase y realizaría herencia. Pondría nombres diferentes para cada uno)
2. Hay pedazos de código repetido, sobre todo en train.py. (Modulizaría creando métodos más generales.)
3. El archivo train.py es muy denso. (tomaría gran parte de los métodos y los implementaría dentro de la clase del modelo)
4. Crearía una sola clase que se encargue de implmentar todos los métodos relacionados con la carga de archivos y paths.
5. Colocaría todos los métodos relacionados con la carga de loaders o dataset en otra clase.
6. Todo lo relacionado con visualización lo colocaría en la clase de visualización.`prepare_for_vis()`
7. Colocaría todos los métodos para crear las matrices de confusión y calcular métricas en una clase especifica.
8. En archvio train.py dejaría los métodos (train,validate)
9. Cambiaría el directorio de la siguiente manera:
    - constants (todos los resultados que no cambian)
    - public_datasets (contiene los datasets raw)
    - data (todo relacionado con datos raw y preprocesamiento)  
        - preprocessing (Procesado de datos)
        - DataManagement (Creacíon de los datasets y dataloaders de todo el proyecto)
        - visualization (Visualización de datos y resultados)
    - model (definición del modelo en una clase con todos sus métodos relacionados)
        - weights (Pesos)
    - metrics (Clases que se utilizen para el calculo de métricas y matrices de confusión)
    - train (El método que realiza el entrenamiento del modelo)
        - progress (logs,checkpoints y todos los datos generados por el entrenamiento)
    - evaluation (Todo lo relacionado con la evaluación del modelo sobre el conjunto test)
    - inference(Código para la inferencia sobre datos nuevos)
    - report (Documentación)
        - figures (figuras de la documentación)
10. Aplicar patrones de diseño a las clases y métodos para evitar que todas los 
mensajes del logger se encuentren mezclados entre todo el código
11. Quitar ai4utils, directamente implementarlo con matplotlib.
### Mejoras en el pipeline
11. Creo que no se ha realizado un tratamiento del desbalance de los datos. EL desbalance de los datos debería aplicarse antes de realizar el slicing. (Random oversampling + data augmentation ?)
12. Previo al entrenamiento final, realizar solo con el conjunto de entrenamiento la técnica k-folding. Si se aplica sobre un conjunto de datos mucho más pequeño entonces utilizar un sampling estratificado del conjunto de entrenamiento.
13. Realizar una búsqueda de hiperparámetros.
14. Lograr una forma de poder aplicar object-level o building-level evaluation. (poder separar los edificios entre si (segmentación de instancias?))
15. Crear métricas para visualización de la función de perdida durante el entrenamiento y la función de perdida de validación