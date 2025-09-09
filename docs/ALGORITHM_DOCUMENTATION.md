# 📐 Especificación Formal del Algoritmo de Templates Óptimos

## Resumen Ejecutivo

**Algoritmo**: Generación de Templates Óptimos mediante Extensiones desde Anclaje  
**Complejidad**: O(1) por landmark  
**Garantía**: 100% de validación en cualquier posición del anclaje  
**Optimización**: Maximización matemática del área del template  

---

## 1. Definición Formal del Problema

### 1.1 Entrada
- **Imagen**: `I` de dimensiones `W × H` píxeles (299×299)
- **Bounding Box**: `B = [x_min, x_max, y_min, y_max]` del landmark
- **Restricción**: Template debe permanecer dentro de `I` para cualquier anclaje en `B`

### 1.2 Objetivo
Encontrar template `T` con:
- **Máxima área**: `area(T) = max{posible}`
- **Anclaje flexible**: `A ∈ B` (cualquier posición en bounding box)
- **Validez garantizada**: `T ∪ A ⊆ I` para todo `A ∈ B`

---

## 2. Fundamento Matemático

### 2.1 Representación del Template

**Definición como extensiones**:
```
T(L,R,U,D) = {(x,y) | -L ≤ x ≤ R, -U ≤ y ≤ D}
```

Donde:
- `L`: Extensión izquierda desde anclaje
- `R`: Extensión derecha desde anclaje  
- `U`: Extensión arriba desde anclaje
- `D`: Extensión abajo desde anclaje

### 2.2 Posicionamiento del Template

Para anclaje `A = (ax, ay)`, el template se posiciona como:
```
T_positioned = {(ax + x, ay + y) | (x,y) ∈ T(L,R,U,D)}
```

### 2.3 Restricción de Límites

Para que `T_positioned ⊆ I` para todo `A ∈ B`:
```
ax - L ≥ 0        ∀ ax ∈ [x_min, x_max]
ax + R ≤ W-1      ∀ ax ∈ [x_min, x_max]  
ay - U ≥ 0        ∀ ay ∈ [y_min, y_max]
ay + D ≤ H-1      ∀ ay ∈ [y_min, y_max]
```

---

## 3. Algoritmo de Optimización

### 3.1 Cálculo de Extensiones Máximas

**Teorema**: Las extensiones máximas que satisfacen las restricciones son:

```python
L_max = x_min           # Distancia mínima al borde izquierdo
R_max = (W-1) - x_max   # Distancia mínima al borde derecho
U_max = y_min           # Distancia mínima al borde superior  
D_max = (H-1) - y_max   # Distancia mínima al borde inferior
```

**Demostración**:
- Para `L_max = x_min`: Si `ax ≥ x_min`, entonces `ax - L_max = ax - x_min ≥ 0` ✓
- Para `R_max = (W-1) - x_max`: Si `ax ≤ x_max`, entonces `ax + R_max = ax + (W-1) - x_max ≤ W-1` ✓
- Análogo para `U_max` y `D_max`

### 3.2 Template Óptimo

**Dimensiones**:
```
width_optimal = L_max + R_max + 1
height_optimal = U_max + D_max + 1
area_optimal = width_optimal × height_optimal
```

**Punto de anclaje sugerido** (centroide del bounding box):
```
anchor_x = (x_min + x_max) // 2
anchor_y = (y_min + y_max) // 2
```

---

## 4. Pseudocódigo del Algoritmo

```
FUNCIÓN calcular_template_óptimo(bbox, image_width, image_height):
    // Extraer límites del bounding box
    x_min, x_max, y_min, y_max = bbox
    
    // Calcular extensiones máximas
    L_max = x_min
    R_max = (image_width - 1) - x_max
    U_max = y_min
    D_max = (image_height - 1) - y_max
    
    // Calcular dimensiones del template
    width = L_max + R_max + 1
    height = U_max + D_max + 1
    area = width × height
    
    // Calcular anclaje sugerido
    anchor_x = (x_min + x_max) // 2
    anchor_y = (y_min + y_max) // 2
    
    RETORNAR {
        extensiones: {L_max, R_max, U_max, D_max},
        dimensiones: {width, height, area},
        anclaje: {anchor_x, anchor_y}
    }
FIN FUNCIÓN
```

---

## 5. Propiedades del Algoritmo

### 5.1 Corrección
**Teorema**: El algoritmo genera templates válidos.
**Prueba**: Por construcción, las extensiones máximas garantizan que para cualquier anclaje `A ∈ B`, el template posicionado `T ∪ A ⊆ I`.

### 5.2 Optimalidad
**Teorema**: El template generado tiene área máxima.
**Prueba**: Cualquier extensión mayor en cualquier dirección violaría las restricciones de límites para algún anclaje en `B`.

### 5.3 Complejidad Computacional
- **Tiempo**: O(1) por landmark (cálculos aritméticos constantes)
- **Espacio**: O(1) de almacenamiento adicional

### 5.4 Escalabilidad
- **Landmarks**: O(n) para n landmarks
- **Imagen**: Independiente del tamaño de imagen (solo afecta constantes)

---

## 6. Casos Especiales y Manejo de Errores

### 6.1 Bounding Box en Esquinas
Si el bounding box está en una esquina de la imagen:
- Algunas extensiones pueden ser 0
- El algoritmo sigue siendo válido
- Template será rectangular (no necesariamente cuadrado)

### 6.2 Bounding Box Ocupando Toda la Imagen
Si `B = [0, W-1, 0, H-1]`:
- `L_max = R_max = U_max = D_max = 0`
- Template será un solo píxel
- Comportamiento correcto y esperado

### 6.3 Validación de Entrada
```python
FUNCIÓN validar_entrada(bbox, W, H):
    x_min, x_max, y_min, y_max = bbox
    
    ASSERT x_min ≥ 0 AND x_max < W
    ASSERT y_min ≥ 0 AND y_max < H  
    ASSERT x_min ≤ x_max
    ASSERT y_min ≤ y_max
FIN FUNCIÓN
```

---

## 7. Implementación de Referencia

### 7.1 Función Principal

```python
def calcular_template_optimo(bbox, image_width=299, image_height=299):
    """
    Calcula el template óptimo para un bounding box dado.
    
    Args:
        bbox: Dict con 'x', 'y', 'width', 'height' del bounding box
        image_width: Ancho de la imagen (default: 299)
        image_height: Alto de la imagen (default: 299)
        
    Returns:
        Dict con extensiones, dimensiones y anclaje óptimos
    """
    # Convertir a coordenadas de límites
    x_min = bbox['x']
    y_min = bbox['y'] 
    x_max = x_min + bbox['width'] - 1
    y_max = y_min + bbox['height'] - 1
    
    # Calcular extensiones máximas
    L_max = x_min
    R_max = (image_width - 1) - x_max
    U_max = y_min
    D_max = (image_height - 1) - y_max
    
    # Calcular dimensiones
    width = L_max + R_max + 1
    height = U_max + D_max + 1
    area = width * height
    
    # Calcular anclaje sugerido
    anchor_x = (x_min + x_max) // 2
    anchor_y = (y_min + y_max) // 2
    
    return {
        'template_extensions': {
            'left': L_max,
            'right': R_max, 
            'up': U_max,
            'down': D_max
        },
        'template_dimensions': {
            'width': width,
            'height': height,
            'area': area
        },
        'anchor_point': {
            'x': anchor_x,
            'y': anchor_y
        }
    }
```

### 7.2 Función de Validación

```python
def validar_template(extensions, anchor, bbox, image_width=299, image_height=299):
    """Valida que el template funcione en cualquier posición del anclaje."""
    x_min, x_max = bbox['x'], bbox['x'] + bbox['width'] - 1
    y_min, y_max = bbox['y'], bbox['y'] + bbox['height'] - 1
    
    for anchor_x in range(x_min, x_max + 1):
        for anchor_y in range(y_min, y_max + 1):
            # Calcular límites del template posicionado
            left = anchor_x - extensions['left']
            right = anchor_x + extensions['right']
            top = anchor_y - extensions['up']
            bottom = anchor_y + extensions['down']
            
            # Verificar que esté dentro de límites
            if not (0 <= left <= right < image_width and 
                    0 <= top <= bottom < image_height):
                return False
    
    return True
```

---

## 8. Referencias y Fundamentos Teóricos

### 8.1 Teoría de Optimización
- **Programación lineal**: Maximización con restricciones lineales
- **Geometría computacional**: Intersección de regiones factibles

### 8.2 Procesamiento de Imágenes
- **Template matching**: Aplicación en búsqueda de patrones
- **Registro de imágenes**: Alineación mediante templates

### 8.3 Visión por Computadora  
- **Detección de características**: Landmarks como puntos de interés
- **Análisis de regiones**: Bounding boxes y regiones de interés

---

*Documento generado automáticamente por el sistema de documentación especializado - 2025-08-15*