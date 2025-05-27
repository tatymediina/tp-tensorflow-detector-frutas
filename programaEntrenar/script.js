// Inicializa el array donde se almacenarán las imágenes y sus etiquetas
let dataset = [];

/**
 * Lee las imágenes cargadas desde el input y las almacena junto con su clase (nombre de carpeta)
 * @param {FileList} files - Lista de archivos seleccionados desde el navegador
 */
function leerArchivos(files) {
  return new Promise((resolve) => {
    let procesadas = 0; // Contador de imágenes procesadas

    // Iteramos sobre cada archivo seleccionado
    for (const file of files) {
      const lector = new FileReader(); // Creamos un lector de archivos

      // Cuando el lector termine de leer el archivo (imagen)
      lector.onload = function (e) {
        const img = new Image(); // Creamos un objeto Image del DOM

        // Cuando la imagen haya sido cargada en memoria
        img.onload = function () {
          // Extraemos el nombre de la carpeta (clase/etiqueta)
          const label = file.webkitRelativePath.split("/")[1];

          // Guardamos la imagen y su etiqueta en el dataset
          dataset.push({ img, label });

          // Aumentamos el contador y resolvemos la promesa cuando estén todas
          procesadas++;
          if (procesadas === files.length) resolve();
        };

        // Establecemos la fuente de la imagen usando base64
        img.src = e.target.result;
      };

      // Leemos el archivo como Data URL (base64)
      lector.readAsDataURL(file);
    }
  });
}

/**
 * Función principal que entrena el modelo con las imágenes cargadas
 */
async function entrenarModelo() {
  // Obtenemos los archivos cargados desde el input HTML
  const files = document.getElementById("upload").files;

  // Leemos y procesamos las imágenes
  await leerArchivos(files);

  // Extraemos las etiquetas únicas (una por clase de fruta)
  const etiquetas = [...new Set(dataset.map(d => d.label))];

  // Mapeamos cada etiqueta a un número (ej: manzana → 0)
  const etiquetaToIndex = Object.fromEntries(etiquetas.map((e, i) => [e, i]));

  // Creamos arreglos para almacenar tensores de imágenes y etiquetas
  const xs = [];
  const ys = [];

  // Procesamos cada imagen del dataset
  for (const { img, label } of dataset) {
    const tensorImg = tf.browser.fromPixels(img)  // Convertimos la imagen a tensor
      .resizeNearestNeighbor([64, 64])            // Redimensionamos a 64x64 píxeles
      .toFloat()                                  // Convertimos a float32
      .div(255.0);                                // Normalizamos valores entre 0 y 1

    xs.push(tensorImg);                           // Agregamos imagen procesada
    ys.push(etiquetaToIndex[label]);              // Agregamos su etiqueta numérica
  }

  // Apilamos las imágenes en un solo tensor 4D (batch, alto, ancho, canales)
  const xsStacked = tf.stack(xs);

  // Convertimos las etiquetas a formato one-hot para clasificación
  const ysOneHot = tf.oneHot(tf.tensor1d(ys, "int32"), etiquetas.length);

  // Creamos un modelo secuencial (una capa sigue a la otra)
  const model = tf.sequential();

  // Capa convolucional para detectar patrones visuales (bordes, formas)
  model.add(tf.layers.conv2d({
    inputShape: [64, 64, 3], // Entrada: imagen RGB 64x64
    filters: 16,             // 16 filtros de convolución
    kernelSize: 3,           // Tamaño del filtro 3x3
    activation: 'relu'       // Activación ReLU: filtra valores negativos
  }));

  // Capa de reducción de resolución (pooling)
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Aplana la imagen 2D a vector 1D
  model.add(tf.layers.flatten());

  // Capa densa oculta con 64 neuronas y activación ReLU
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));

  // Capa de salida con tantas neuronas como clases (una por etiqueta)
  model.add(tf.layers.dense({
    units: etiquetas.length,
    activation: 'softmax' // Softmax devuelve probabilidades para clasificación
  }));

  // Compilamos el modelo, definiendo cómo se entrena
  model.compile({
    optimizer: 'adam',                  // Algoritmo de optimización rápido
    loss: 'categoricalCrossentropy',   // Pérdida para clasificación multiclase
    metrics: ['accuracy']              // Métrica que muestra precisión
  });

  // Entrenamos el modelo usando los datos procesados
  await model.fit(xsStacked, ysOneHot, {
    epochs: 10, // Número de ciclos de entrenamiento (puede aumentarse)
    callbacks: {
      // Callback para mostrar el avance del entrenamiento en la interfaz
      onEpochEnd: (epoch, logs) => {
        document.getElementById('output').textContent +=
          `Época ${epoch + 1}: Pérdida = ${logs.loss.toFixed(4)}, Precisión = ${logs.acc.toFixed(4)}\n`;
		  //console.log("perdida "+logs.loss.toFixed(4));
      }
    }
  });

  // Guardamos el modelo entrenado en el navegador (IndexedDB)
  await model.save('indexeddb://modelo-frutas');

  await model.save('downloads://modelo-frutas');


  // Avisamos al usuario que el entrenamiento finalizó con éxito
  alert('✅ Modelo entrenado y guardado exitosamente.');
}
