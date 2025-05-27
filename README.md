## Entrenamiento del modelo

Este proyecto utiliza TensorFlow.js para entrenar un modelo de clasificación de frutas. Por defecto, el modelo **no está incluido** en este repositorio para evitar archivos binarios pesados.

### Como clonar este repositorio

1. Clonar el repositorio
```bash
git clone https://github.com/tatymediina/tp-tensorflow-detector-frutas.git
```
2. Acceder a la carpeta clonada
```bash
cd  tp-tensorflow-detector-frutas.git
```
### ¿Cómo entrenar el modelo?

1. Accede a la carpeta programaEntrenar`desde la terminar:
```bash
cd ./programaEntrenar
```
2. Ejecutar en la terminal para crear las carpetas con las imágenes de frutas.
```bash
python traer.frutas.py
```
3. Ejecuta el `index.html` con el comando o podes abrir manualmente en tu navegador
```bash
start index.html
```
4. Subí las imágenes desde el input (usa la opción "carpetas" si querés agrupar por clases).

#### Nota: 
El código está configurado para que sean 3 imagenes por cada fruta. Podés cambiarlo en el archivo `traer-fruta.py`en la última función donde dice max_num

3. Presioná el botón "Entrenar modelo".

4. Una vez finalizado, el modelo se guardará automáticamente como archivos descargables:
   - `modelo-frutas.json`
   - `modelo-frutas.weights.bin`

5. Copiá esos archivos a la carpeta `utilizarModelo`.

6. Colocarse en la carpeta `utilicarModelo`
```bash
cd ..

cd ./utilizarModelo
```

7. Ejecutar 
```
python -m http.server

```
8. Abrí desde tu navegador:
`http://localhost:8000`

9. Subí una imágen y observa cuánto el modelo se acerca.
