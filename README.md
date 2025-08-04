# Museum Network Analysis - UBA

Este repositorio contiene el desarrollo completo de los Trabajos Prácticos Nº1 y Nº2 de la materia Álgebra Lineal Computacional (Licenciatura en Ciencia de Datos, UBA). El objetivo fue modelar y analizar redes de museos en CABA aplicando herramientas algebraicas, computacionales y heurísticas espectrales.

## 🧠 Contenido

- `TP_notebook.ipynb`: Notebook con análisis de PageRank, factorización LU, detección de comunidades mediante métodos espectrales y visualización geográfica.
- `template_funciones.py` y `template_funciones_2.py: estructura base del TP, funciones auxiliares para ambos trabajos (LU, PageRank, corte mínimo, modularidad).
- `ALC_2025-TP1-PRyMuseos.pdf`: Enunciado de la primer parte del TP.
- `ALC_1C2025-TP2-Comunidades.pdf`: Enunciado de la segunda parte del TP.
- `visitas.txt`: Datos ilustrativos/ejemplo de visitas a museos.

## 📊 Métodos implementados

### TP1: PageRank y Caminatas Aleatorias
- Construcción de redes dirigidas y pesadas.
- Matriz de transiciones con distancia geográfica.
- PageRank con amortiguamiento.
- Factorización LU sin inversión directa.
- Análisis de sensibilidad con número de condición.

### TP2: Detección de Comunidades
- Matriz Laplaciana y de modularidad.
- Autovalores y autovectores.
- Método de la potencia y potencia inversa.
- Deflación de Hotelling.
- Heurísticas de corte mínimo y modularidad.

## 🗺️ Visualizaciones

- Mapas interactivos con tamaño de nodos proporcional a PageRank.
- Evolución de rankings según parámetros m (cantidad de enlaces) y alpha (coeficiente de dumping).
- Detección de comunidades en grafos reales.

## 🧑‍💻 Tecnologías

- Python 3
- NumPy, SciPy, Pandas, GeoPandas
- Matplotlib, NetworkX
- Jupyter Notebook

## 🎓 Autores

Arango Joaquin
Morrone Valentina
Peix Francisco
