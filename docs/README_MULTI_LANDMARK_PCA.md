# 🧬 Sistema de Análisis PCA Multi-Landmark

Sistema completo para análisis de componentes principales (PCA) y eigenfaces de todos los landmarks L1-L15 del dataset médico.

## 📁 Archivos Principales

### Scripts de Análisis
- **`multi_landmark_pca_analysis.py`** - Script principal con clase `MultiLandmarkPCAProcessor`
- **`run_multi_landmark_analysis.py`** - Script de ejecución interactivo
- **`pca_eigenfaces_analysis.py`** - Script original para L1 (mantenido intacto)

## 🚀 Uso Rápido

### Opción 1: Ejecución Interactiva
```bash
python3 run_multi_landmark_analysis.py
```

### Opción 2: Ejecución Directa
```bash
python3 multi_landmark_pca_analysis.py
```

### Opción 3: Uso Programático
```python
from multi_landmark_pca_analysis import MultiLandmarkPCAProcessor

# Procesar todos los landmarks
processor = MultiLandmarkPCAProcessor()
summary = processor.process_all_landmarks()
processor.generate_consolidated_reports()

# Procesar landmarks específicos
summary = processor.process_all_landmarks(["L1", "L5", "L10"])
```

## 📊 Funcionalidades

### ✅ Procesamiento Automático
- **Detección automática** de dimensiones por landmark
- **Procesamiento completo** de 15 landmarks (L1-L15)
- **Gestión inteligente de memoria** para 10,035 imágenes totales
- **Manejo robusto de errores** por landmark

### ✅ Análisis Científico Completo
Para cada landmark:
- 🧮 **Análisis PCA** con validaciones matemáticas
- 🎭 **Eigenfaces** con normalización global científica
- 📈 **Gráficos de varianza** acumulada e individual
- 🔄 **Reconstrucciones** con diferentes números de componentes
- 🗺️ **Proyección 2D** en espacio de componentes principales
- 💾 **Modelo entrenado** guardado para uso futuro

### ✅ Reportes Consolidados
- 📊 **Comparación de varianza** entre landmarks
- 🖼️ **Grid de eigenfaces principales** de todos los landmarks
- 📋 **Análisis estadístico** comparativo
- 📄 **Reportes JSON** detallados

## 📁 Estructura de Resultados

```
output_pca_analysis_all_landmarks/
├── L1/                                  # Análisis completo del landmark L1
│   ├── pure_images/                     # Imágenes sin títulos
│   │   ├── mean_face.png               # Imagen promedio
│   │   ├── eigenfaces_grid_5.png       # Grid de 5 eigenfaces
│   │   ├── eigenfaces_grid_10.png      # Grid de 10 eigenfaces
│   │   └── individual_reconstructions/ # Reconstrucciones individuales
│   ├── titled_versions/                # Imágenes con títulos y gráficos
│   │   ├── eigenfaces_from_individuals_opencv.png
│   │   ├── cumulative_variance.png
│   │   ├── reconstructions_titled.png
│   │   └── pca_2d_projection.png
│   ├── individual_eigenfaces/          # Eigenfaces individuales
│   │   ├── eigenface_1.png
│   │   ├── eigenface_2.png
│   │   └── ... (hasta eigenface_10.png)
│   ├── trained_model.npz              # Modelo PCA entrenado
│   └── analysis_report.json           # Reporte detallado
├── L2/ ... L15/                       # Estructura idéntica para cada landmark
├── consolidated_reports/              # Reportes comparativos
│   ├── comparison_summary.json        # Resumen estadístico completo
│   ├── variance_comparison.png        # Gráficos comparativos
│   └── landmarks_eigenfaces_grid.png  # Grid de eigenfaces principales
├── final_processing_summary.json     # Resumen final completo
└── processing_log.txt                # Log detallado de procesamiento
```

## 🔧 Configuración y Requisitos

### Dependencias
```bash
pip install numpy opencv-python matplotlib seaborn scikit-learn pandas tqdm pathlib2
```

### Estructura de Entrada Requerida
```
output_landmarks_complete/
├── L1/
│   ├── COVID-100_L1.png
│   ├── Normal-102_L1.png
│   └── Viral_Pneumonia-101_L1.png
├── L2/ ... L15/  # Estructura similar
```

## 📈 Especificaciones Técnicas

### Capacidades
- **15 landmarks** procesados automáticamente
- **669 imágenes** por landmark (10,035 total)
- **Dimensiones variables** detectadas automáticamente
- **Procesamiento secuencial** con gestión de memoria

### Rendimiento Estimado
- ⏱️ **Tiempo**: 15-30 minutos para todos los landmarks
- 🧠 **Memoria**: Pico de 2-3GB (procesamiento secuencial)
- 💾 **Disco**: ~5-10GB de resultados completos

### Dimensiones por Landmark (Detectadas Automáticamente)
| Landmark | Dimensiones | Píxeles | Componentes PCA |
|----------|-------------|---------|-----------------|
| L1       | 200×159     | 31,800  | 668            |
| L2       | 186×93      | 17,298  | 668            |
| L5       | 164×153     | 25,092  | 668            |
| L10      | 209×157     | 32,813  | 668            |
| L15      | 177×102     | 18,054  | 668            |
| ...      | ...         | ...     | ...            |

## 🧪 Validaciones Científicas

### Calidad del Análisis
- ✅ **Validaciones matemáticas** de ortogonalidad de componentes
- ✅ **Normalización global** científicamente correcta
- ✅ **Reproducibilidad** garantizada con seeds fijos
- ✅ **Consistencia píxel-por-píxel** en visualizaciones

### Verificaciones Automáticas
- 🔍 Centrado correcto de datos
- 🔍 Ortogonalidad de componentes principales
- 🔍 Normalización de eigenfaces
- 🔍 Reconstrucción perfecta con todos los componentes

## 📊 Casos de Uso

### 1. Análisis Exploratorio
```python
# Procesar algunos landmarks para exploración rápida
processor = MultiLandmarkPCAProcessor()
summary = processor.process_all_landmarks(["L1", "L5", "L10"])
```

### 2. Análisis Completo del Dataset
```python
# Procesar todos los landmarks para análisis completo
processor = MultiLandmarkPCAProcessor()
summary = processor.process_all_landmarks()  # Todos los L1-L15
```

### 3. Comparación de Landmarks Específicos
```python
# Comparar landmarks anatómicamente relacionados
processor = MultiLandmarkPCAProcessor()
summary = processor.process_all_landmarks(["L1", "L2", "L3"])  # Región específica
```

## 🔧 Personalización

### Modificar Rutas
```python
processor = MultiLandmarkPCAProcessor(
    landmarks_base_path="custom_landmarks_path",
    output_base_path="custom_output_path"
)
```

### Procesar Subconjunto
```python
# Solo landmarks impares
landmarks = [f"L{i}" for i in range(1, 16, 2)]  # L1, L3, L5, ...
summary = processor.process_all_landmarks(landmarks)
```

## 📝 Logs y Monitoreo

### Log de Procesamiento
El archivo `processing_log.txt` contiene:
- Timestamps detallados
- Progreso por landmark
- Estadísticas de procesamiento
- Errores y advertencias

### Monitoreo en Tiempo Real
```bash
# Seguir el log en tiempo real
tail -f output_pca_analysis_all_landmarks/processing_log.txt
```

## 🚨 Solución de Problemas

### Error de Memoria
- Cerrar otras aplicaciones que consuman memoria
- El procesamiento es secuencial, cada landmark se libera de memoria

### Landmark Faltante
- Verificar que exista el directorio `output_landmarks_complete/LX/`
- Confirmar que contenga archivos PNG

### Dimensiones Incorrectas
- El sistema detecta automáticamente las dimensiones
- Verifica la integridad de las imágenes PNG

## 🎯 Características Destacadas

- 🔄 **Procesamiento secuencial** con limpieza de memoria automática
- 📊 **Reportes consolidados** para comparación entre landmarks
- 🎛️ **Detección automática** de dimensiones y configuración
- 🛡️ **Manejo robusto** de errores por landmark
- 📈 **Visualizaciones científicas** con estándares de la literatura
- 💾 **Modelos guardados** listos para análisis posteriores

---

**Desarrollado por**: Sistema de desarrollo con IA especializada  
**Fecha**: 2025-08-19  
**Basado en**: `pca_eigenfaces_analysis.py` (análisis original para L1)