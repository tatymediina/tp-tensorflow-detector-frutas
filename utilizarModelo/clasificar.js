let modelo;
const etiquetas = ["manzana", "banana", "pera"]; // Reemplazá por las clases reales de tu dataset
const resultado = document.getElementById('resultado');
const preview = document.getElementById('preview');

// Cargar el modelo desde IndexedDB
async function cargarModelo() {
  try {
    modelo = await tf.loadLayersModel('./modelo-frutas.json');
    resultado.textContent = "✅ Modelo cargado correctamente.";
  } catch (e) {
    resultado.textContent = "❌ No se pudo cargar el modelo. Asegurate de haberlo entrenado antes.";
  }
}

cargarModelo();

document.getElementById('imagen').addEventListener('change', async function (e) {
  const archivo = e.target.files[0];
  if (!archivo) return;

  const img = new Image();
  img.src = URL.createObjectURL(archivo);
  preview.src = img.src;

  img.onload = async () => {
    const tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([64, 64])
      .toFloat()
      .div(255.0)
      .expandDims();

    const pred = modelo.predict(tensor);
    const index = (await pred.argMax(1).data())[0];
    const prob = (await pred.max().data())[0];

    resultado.textContent = `🔍 Detectado: ${etiquetas[index]} (${(prob * 100).toFixed(2)}% seguro)`;
  };
});
