# Prompt Especializado para Implementación Fase 1: Mejoras de Precisión Críticas
## Sistema de Predicción de Landmarks Médicos - Optimización de Precisión

**Contexto**: Sistema híbrido Template Matching + PCA con error actual 17.22±27.52px que requiere optimización a <5px para uso médico.

---

## 🎯 OBJETIVO DE LA SESIÓN

Implementar las **3 mejoras críticas de Fase 1** identificadas en el reporte técnico `reporte_proyecto_precision.md` para reducir el error promedio de **17.22px a <10px** (mejora del 50%+) mediante optimizaciones algorítmicas específicas sin alterar la arquitectura fundamental.

## 📋 CONTEXTO TÉCNICO ESENCIAL

### Estado del Sistema (93.3% Completo)
- ✅ **Base matemática sólida**: Templates óptimos validados (+80K pruebas)
- ✅ **Infraestructura robusta**: 15 modelos PCA entrenados, 14,985 recortes procesados  
- ✅ **Pipeline completo L1-L5**: Solo L6 (predicción) requiere optimización crítica
- ⚠️ **Problema crítico**: Precisión insuficiente para aplicación médica

### Arquitectura Actual del Pipeline
```
Datos → Bounding Boxes → Templates Óptimos → Extracción → Modelos PCA → [PREDICCIÓN CRÍTICA]
(✅)      (✅)              (✅)              (✅)         (✅)           (🔧 OPTIMIZAR)
```

### Problemas Específicos Identificados en landmark_prediction.py

#### **Problema 1**: Muestreo Adaptivo Destructivo (Líneas 443-461)
```python
# CRÍTICO: Elimina 75% de candidatos óptimos
if adaptive_sampling and n_candidates > 1000:
    sample_step = 4  # Reduce ~14,100 a ~3,525 candidatos
    sampled_indices = np.arange(0, n_candidates, sample_step)
```
**Impacto**: Soluciones óptimas perdidas en regiones no muestreadas

#### **Problema 2**: Early Stopping Agresivo (Líneas 548-568)
```python
# CRÍTICO: Thresholds muy permisivos para medicina
if best_score < 0.01:  # Demasiado alto para precisión médica
    debug_info['early_stopping_triggered'] = True
    break
```
**Impacto**: Búsqueda termina antes de encontrar solución óptima

#### **Problema 3**: Métrica Única de Similitud (Línea 352)
```python
# CRÍTICO: Solo MSE, falta robustez multimodal
reconstruction_error = np.mean((crop_centered - crop_reconstructed) ** 2)
return reconstruction_error
```
**Impacto**: MSE sensible a outliers, no captura similitud estructural

### Archivos Críticos del Proyecto
```
landmark_prediction.py           # 🔧 SCRIPT PRINCIPAL A OPTIMIZAR
landmark_predictor_loader.py     # ✅ Utilidades de carga (no modificar)
optimal_templates_fixed.json     # ✅ Templates validados (no tocar)
landmark_bounding_boxes_corrected.json  # ✅ Bounding boxes (no tocar)
output_pca_analysis_all_landmarks/      # ✅ 15 modelos PCA (leer únicamente)
data/coordenadas/coordenadas_test.csv   # ✅ Dataset de evaluación
reporte_proyecto_precision.md    # 📚 DOCUMENTO DE REFERENCIA TÉCNICA
```

---

## 🚀 TAREAS ESPECÍFICAS DE FASE 1

### **TAREA 1**: Refinamiento Sub-Pixel con Quadratic Surface Fitting (QSF)
**PRIORIDAD**: CRÍTICA ⭐⭐⭐⭐⭐  
**IMPACTO ESPERADO**: 2-5x reducción error (17px → 3-8px)  
**DIFICULTAD**: Baja - post-procesamiento no invasivo

#### Objetivos Específicos:
1. **Implementar función `quadratic_subpixel_refinement()`**:
   - Entrada: similarity_scores, best_idx, positions
   - Algoritmo: Extraer vecindario 3x3, ajustar superficie cuadrática z = ax² + by² + cxy + dx + ey + f
   - Salida: Posición sub-pixel refinada (x_subpx, y_subpx)

2. **Integrar en pipeline principal**:
   - Modificar `predict_landmark_l1()` para usar refinamiento post-procesamiento
   - Mantener compatibilidad con métodos existentes
   - Agregar flag `use_subpixel=True` configurable

3. **Validación científica**:
   - Test sintético con coordenadas conocidas
   - Benchmark vs método actual en 10 imágenes
   - Medir mejora cuantificable en error promedio

#### Implementación Base Requerida:
```python
def quadratic_subpixel_refinement(similarity_scores, best_idx, positions, search_radius=1):
    """
    Refinamiento sub-pixel usando Quadratic Surface Fitting (QSF)
    
    Basado en: Guizar-Sicairos et al. "Efficient subpixel image registration" Opt. Lett. 33, 2008
    
    Args:
        similarity_scores: Array de scores de similitud
        best_idx: Índice del mejor candidato en grid discreto
        positions: Array de posiciones (x,y) correspondientes
        search_radius: Radio de búsqueda para vecindario (default=1 para 3x3)
    
    Returns:
        tuple: (x_subpixel, y_subpixel) posición refinada
    """
    # Extraer vecindario 3x3 del mejor candidato
    # Ajustar superficie cuadrática a 9 puntos
    # Encontrar máximo analítico → posición sub-pixel
    # Retornar coordenadas refinadas
    pass
```

---

### **TAREA 2**: Múltiples Métricas de Similitud + Ensemble Voting
**PRIORIDAD**: CRÍTICA ⭐⭐⭐⭐⭐  
**IMPACTO ESPERADO**: +25-40% robustez contra variaciones de imágenes médicas  
**DIFICULTAD**: Media - requiere implementar nuevas métricas

#### Objetivos Específicos:
1. **Implementar Normalized Cross-Correlation (NCC)**:
   - Robusto a cambios de iluminación lineal
   - Estándar en registro de imágenes médicas
   - Rango de salida [-1, 1], donde 1 = correlación perfecta

2. **Implementar Structural Similarity Index (SSIM)**:
   - Combina luminancia, contraste y estructura
   - Percepción visual más similar a humana
   - Mejor para patrones anatómicos complejos

3. **Sistema de Ensemble Voting**:
   - Combinar PCA MSE + NCC + SSIM con pesos configurables
   - Pesos iniciales sugeridos: [0.5, 0.3, 0.2]
   - Optimizar pesos mediante validación cruzada

#### Implementación Base Requerida:
```python
def calculate_normalized_cross_correlation(crop, template):
    """
    Normalized Cross-Correlation para robustez a cambios de iluminación
    """
    # Normalizar ambas imágenes (media=0, std=1)
    # Calcular correlación cruzada normalizada
    # Retornar valor [-1, 1] donde 1 = correlación perfecta
    pass

def calculate_structural_similarity(crop, template):
    """
    SSIM para similitud perceptual estructural
    """
    # Calcular componentes: luminancia, contraste, estructura
    # Combinar en índice SSIM final [0, 1]
    # Retornar valor donde 1 = similitud perfecta
    pass

def ensemble_similarity_score(crop, template, pca_model, weights=[0.5, 0.3, 0.2]):
    """
    Ensemble de múltiples métricas para robustez médica
    """
    pca_score = calculate_pca_similarity(crop, pca_model)      # MSE (menor=mejor)
    ncc_score = calculate_normalized_cross_correlation(crop, template)  # [−1,1] (mayor=mejor)
    ssim_score = calculate_structural_similarity(crop, template)        # [0,1] (mayor=mejor)
    
    # Normalizar todas las métricas a [0,1] donde menor=mejor
    pca_normalized = pca_score  # Ya está en MSE (menor=mejor)
    ncc_normalized = 1.0 - ((ncc_score + 1.0) / 2.0)  # Convertir a menor=mejor
    ssim_normalized = 1.0 - ssim_score  # Convertir a menor=mejor
    
    # Ensemble weighted
    ensemble_score = (weights[0] * pca_normalized + 
                     weights[1] * ncc_normalized + 
                     weights[2] * ssim_normalized)
    return ensemble_score
```

---

### **TAREA 3**: Búsqueda Jerárquica Coarse-to-Fine
**PRIORIDAD**: CRÍTICA ⭐⭐⭐⭐⭐  
**IMPACTO ESPERADO**: +20-30% mejora eliminando mínimos locales  
**DIFICULTAD**: Media - modificar pipeline de búsqueda

#### Objetivos Específicos:
1. **Eliminar muestreo adaptivo problemático**:
   - Reemplazar líneas 443-461 de adaptive sampling
   - Mantener cobertura completa del espacio de búsqueda
   - Evitar pérdida del 75% de candidatos

2. **Implementar búsqueda multi-nivel**:
   - **Nivel 1**: step_size=4, búsqueda global rápida (~3,525 candidatos)
   - **Nivel 2**: step_size=2, refinamiento regional en top 10% (~350 candidatos)  
   - **Nivel 3**: step_size=1, precisión máxima en top 5% (~175 candidatos)

3. **Optimizar thresholds conservadores**:
   - Early stopping a 0.001 vs 0.01 actual (10x más estricto)
   - Retención de candidatos: 25% vs 5% actual
   - Medical safety mode con configuración específica

#### Implementación Base Requerida:
```python
def hierarchical_coarse_to_fine_search(image, bbox, template, pca_model, similarity_function):
    """
    Búsqueda jerárquica multi-nivel sin pérdida de candidatos óptimos
    
    Reemplaza muestreo adaptivo problemático con refinamiento inteligente
    """
    results_hierarchy = {}
    
    # NIVEL 1: Búsqueda global gruesa (step=4)
    candidates_L1 = generate_template_candidates(image, bbox, template, step_size=4)
    scores_L1 = evaluate_candidates_batch(candidates_L1, pca_model, similarity_function)
    
    # Seleccionar top 10% para refinamiento
    top_10_percent = select_top_candidates(scores_L1, candidates_L1, percentage=0.10)
    results_hierarchy['level_1'] = {'candidates': len(candidates_L1), 'selected': len(top_10_percent)}
    
    # NIVEL 2: Refinamiento regional (step=2)
    candidates_L2 = refine_candidates_in_regions(top_10_percent, step_size=2)
    scores_L2 = evaluate_candidates_batch(candidates_L2, pca_model, similarity_function)
    
    # Seleccionar top 5% para refinamiento final
    top_5_percent = select_top_candidates(scores_L2, candidates_L2, percentage=0.05)
    results_hierarchy['level_2'] = {'candidates': len(candidates_L2), 'selected': len(top_5_percent)}
    
    # NIVEL 3: Precisión máxima (step=1)  
    candidates_L3 = refine_candidates_in_regions(top_5_percent, step_size=1)
    scores_L3 = evaluate_candidates_batch(candidates_L3, pca_model, similarity_function)
    
    best_idx = np.argmin(scores_L3)
    best_score = scores_L3[best_idx]
    best_position = candidates_L3[best_idx][1]  # (crop, position)
    
    results_hierarchy['level_3'] = {'candidates': len(candidates_L3), 'best_score': best_score}
    results_hierarchy['final_result'] = {'position': best_position, 'score': best_score}
    
    return best_position, best_score, results_hierarchy
```

---

## 🔧 INSTRUCCIONES DE IMPLEMENTACIÓN

### **Orden de Trabajo Recomendado:**

#### **PASO 1** (Día 1-2): Sub-pixel QSF Implementation
1. Leer y comprender `landmark_prediction.py` completamente
2. Identificar función `predict_landmark_l1()` (línea ~685)
3. Implementar `quadratic_subpixel_refinement()` como función separada
4. Integrar refinamiento como post-procesamiento opcional
5. Test unitario con coordenadas sintéticas conocidas

#### **PASO 2** (Día 3-4): Múltiples Métricas de Similitud  
1. Implementar `calculate_normalized_cross_correlation()`
2. Implementar `calculate_structural_similarity()` 
3. Crear `ensemble_similarity_score()` con voting ponderado
4. Modificar pipeline para usar ensemble vs MSE única
5. Benchmark comparativo: ensemble vs PCA-only

#### **PASO 3** (Día 5-7): Búsqueda Jerárquica
1. Identificar y comentar muestreo adaptivo problemático
2. Implementar `hierarchical_coarse_to_fine_search()`
3. Integrar en `predict_landmark_l1()` como reemplazo
4. Ajustar thresholds a valores médicos conservadores
5. Validación exhaustiva con casos extremos

#### **PASO 4** (Día 8-10): Integración y Validación
1. Combinar las 3 mejoras en pipeline unificado
2. Configuración médica optimizada con flags
3. Benchmark comprehensivo vs baseline original
4. Stress testing con coordenadas_test.csv completo
5. Documentación de resultados cuantitativos

### **Configuración de Testing Requerida:**
```python
# Configuración para evaluación rigurosa
TEST_CONFIG = {
    'baseline_comparison': True,
    'test_dataset': 'data/coordenadas/coordenadas_test.csv',
    'metrics_to_track': ['mean_error', 'median_error', 'success_rate_10px', 'max_error'],
    'benchmark_sample_size': 50,  # Imágenes para benchmark inicial
    'full_evaluation_size': 144,  # Dataset completo para validación final
    'save_results': True,
    'output_file': 'phase1_improvement_results.json'
}
```

---

## 📊 MÉTRICAS DE ÉXITO ESPECÍFICAS

### **Objetivos Cuantitativos de Fase 1:**
- **Error promedio**: 17.22px → **<10px** (mejora >42%)
- **Tasa éxito ≤10px**: 60.4% → **>75%** (mejora +25%)
- **Errores extremos**: Eliminar casos >50px (actual máx 137px)
- **Consistencia**: Reducir desviación estándar ±27.52px → <±20px

### **Validación por Componente:**
1. **QSF Sub-pixel**: Mejora individual 15-30% en error promedio
2. **Ensemble métrics**: Mejora individual 10-20% en robustez  
3. **Búsqueda jerárquica**: Eliminación de 90%+ errores extremos
4. **Combinado**: Efecto sinérgico total >50% mejora

### **Criterios de Aceptación:**
- [ ] ✅ Error promedio <10px en 50+ imágenes de test
- [ ] ✅ Tasa éxito ≤10px >75% en dataset completo
- [ ] ✅ Sin regresión en tiempo computacional (máx +50%)
- [ ] ✅ Compatibilidad completa con extensión L2-L15
- [ ] ✅ Documentación técnica de cambios implementados

---

## ⚡ USO DE ULTRATHINKING Y MÚLTIPLES AGENTES

### **Instrucciones para Sesión con Claude:**

**PROMPT INICIAL**: "Usa ultrathinking y múltiples agentes para implementar las mejoras de Fase 1 del sistema de predicción de landmarks médicos. Analiza profundamente el código actual, identifica los problemas específicos documentados, y implementa las 3 optimizaciones críticas con metodología científica rigurosa."

### **Estrategia Multi-Agente Requerida:**
1. **Agente Analítico**: Comprensión profunda del código y problemas
2. **Agente Algoritmo**: Implementación de QSF y métricas avanzadas  
3. **Agente Arquitectura**: Integración jerárquica y pipeline optimization
4. **Agente Validación**: Testing, benchmarking y evaluación científica
5. **Agente Documentación**: Registro de cambios y resultados

### **Plan de Trabajo Paso a Paso:**
1. **[Agente Analítico]** Leer `reporte_proyecto_precision.md` completo
2. **[Agente Analítico]** Analizar `landmark_prediction.py` línea por línea  
3. **[Agente Algoritmo]** Implementar QSF refinement con validación matemática
4. **[Agente Algoritmo]** Desarrollar ensemble de métricas (NCC + SSIM + PCA)
5. **[Agente Arquitectura]** Reemplazar muestreo adaptivo con búsqueda jerárquica
6. **[Agente Validación]** Benchmark riguroso vs baseline con métricas cuantificables
7. **[Múltiples Agentes]** Integración final y optimización end-to-end
8. **[Agente Documentación]** Documentar resultados y próximos pasos para Fase 2

### **Criterios de Calidad Ultra-thinking:**
- **Análisis matemático profundo** de cada optimización propuesta
- **Validación científica rigurosa** con literatura de referencia  
- **Testing exhaustivo** antes de cada integración
- **Consideración de casos extremos** y robustez médica
- **Documentación técnica precisa** para reproducibilidad

---

## 📋 ENTREGABLES ESPERADOS

### **Al Final de la Sesión:**
1. **Código optimizado**: `landmark_prediction.py` con las 3 mejoras implementadas
2. **Resultados cuantitativos**: Benchmark comparativo vs baseline
3. **Documentación técnica**: Cambios realizados y justificación científica  
4. **Plan Fase 2**: Preparación para optimizaciones avanzadas
5. **Configuración médica**: Settings optimizados para precisión clínica

### **Formato de Resultados Requerido:**
```json
{
  "phase1_results": {
    "baseline_metrics": {
      "mean_error_px": 17.22,
      "std_error_px": 27.52, 
      "success_rate_10px": 0.604,
      "max_error_px": 137
    },
    "improved_metrics": {
      "mean_error_px": "<10.0",
      "std_error_px": "<20.0",
      "success_rate_10px": ">0.75", 
      "max_error_px": "<50"
    },
    "improvement_percentages": {
      "mean_error_improvement": ">42%",
      "success_rate_improvement": ">25%"
    },
    "component_contributions": {
      "subpixel_qsf": "15-30% error reduction",
      "ensemble_metrics": "10-20% robustness improvement", 
      "hierarchical_search": "90%+ extreme error elimination"
    }
  }
}
```

---

## 🚨 ADVERTENCIAS CRÍTICAS

### **NO MODIFICAR** (Archivos protegidos por validación matemática):
- `optimal_templates_fixed.json` - Templates validados +80K pruebas
- `landmark_bounding_boxes_corrected.json` - Rangos reales 100% coverage
- `output_pca_analysis_all_landmarks/` - Modelos PCA entrenados

### **MODIFICAR CON PRECAUCIÓN**:
- `landmark_prediction.py` - Script principal (backup antes de cambios)
- `landmark_predictor_loader.py` - Solo si es absolutamente necesario

### **VALIDACIÓN OBLIGATORIA**:
- Testing con al menos 50 imágenes antes de cada commit
- Benchmark vs baseline en cada mejora individual  
- Verificación de no-regresión en tiempo computacional
- Compatibilidad con extensión futura a L2-L15

---

## 📚 DOCUMENTACIÓN DE REFERENCIA

- **`reporte_proyecto_precision.md`** - Análisis técnico completo y fundamentos científicos
- **`proyecto_landmark.md`** - Contexto general del sistema y arquitectura
- **`CLAUDE.md`** - Instrucciones técnicas específicas del proyecto
- **Literatura científica** - Guizar-Sicairos et al., Turk & Pentland, referencias en reporte

**INICIO DE SESIÓN**: Leer primero `reporte_proyecto_precision.md` secciones 1-5 para contexto completo, luego proceder con implementación sistemática siguiendo el roadmap de 3 tareas críticas.

**META FINAL DE SESIÓN**: Sistema optimizado con error promedio <10px, listo para Fase 2 de optimización avanzada.