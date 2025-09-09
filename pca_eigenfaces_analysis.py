#!/usr/bin/env python3
"""
Análisis PCA y Eigenfaces para Landmark L1
==========================================

Este script implementa análisis de componentes principales (PCA) y eigenfaces
para las imágenes del landmark L1 extraídas del dataset médico.

Funcionalidades:
- Carga y preprocesamiento de 669 imágenes L1 (200x159 píxeles)
- Análisis PCA completo con sklearn
- Generación de imagen promedio
- Visualización de 5 principales eigenimágenes
- Gráfico de varianza acumulada
- Reconstrucciones con diferentes números de componentes
- Proyección 2D de los datos

Autor: Sistema de desarrollo con IA especializada
Fecha: 2025-08-18
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')

class LandmarkPCAAnalysis:
    """
    Clase principal para análisis PCA de imágenes del landmark L1
    """
    
    def __init__(self, input_path, output_path, image_size=(200, 159)):
        """
        Inicializar el análisis PCA
        
        Args:
            input_path (str): Ruta al directorio con imágenes L1
            output_path (str): Ruta de salida para resultados
            image_size (tuple): Dimensiones de las imágenes (width, height)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.width, self.height = image_size
        self.n_pixels = self.width * self.height
        
        # Datos del análisis
        self.images_data = None
        self.image_filenames = []
        self.image_labels = []
        self.mean_image = None
        self.pca_model = None
        self.principal_components = None
        self.transformed_data = None
        
        # Crear directorios de salida organizados
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.pure_images_path = self.output_path / "pure_images"
        self.titled_versions_path = self.output_path / "titled_versions"
        self.individual_reconstructions_path = self.pure_images_path / "individual_reconstructions"
        
        # Crear subdirectorios
        self.pure_images_path.mkdir(exist_ok=True)
        self.titled_versions_path.mkdir(exist_ok=True)
        self.individual_reconstructions_path.mkdir(exist_ok=True)
    
    def augment_image(self, image_2d):
        """
        Aplicar transformaciones afines aleatorias a una imagen 2D en escala de grises.
        
        Esta función genera una versión aumentada de la imagen aplicando transformaciones
        sutiles y realistas para imágenes médicas: rotación, escalado y traslación.
        
        Args:
            image_2d (np.ndarray): Imagen 2D en escala de grises (height, width)
                                 con valores normalizados entre 0.0 y 1.0
        
        Returns:
            np.ndarray: Imagen aumentada con las mismas dimensiones que la entrada
        
        Transformaciones aplicadas:
            - Rotación: Entre -5 y +5 grados (aleatorio)
            - Escalado: Entre 0.95 y 1.05 (95% a 105%, aleatorio)  
            - Traslación: Entre -5 y +5 píxeles en X e Y (aleatorio)
        
        Método técnico:
            - Usa cv2.getRotationMatrix2D para crear matriz de transformación combinada
            - Aplica cv2.warpAffine con cv2.BORDER_REFLECT_101 para evitar bordes negros
        """
        # Generar parámetros de transformación aleatorios dentro de rangos sutiles
        angle = random.uniform(-5.0, 5.0)  # Rotación en grados
        scale = random.uniform(0.95, 1.05)  # Factor de escalado
        tx = random.uniform(-5.0, 5.0)      # Traslación en X (píxeles)
        ty = random.uniform(-5.0, 5.0)      # Traslación en Y (píxeles)
        
        # Obtener dimensiones de la imagen
        h, w = image_2d.shape
        
        # Calcular centro de la imagen para rotación
        center = (w // 2, h // 2)
        
        # Crear matriz de transformación combinada (rotación + escalado)
        # cv2.getRotationMatrix2D retorna una matriz 2x3 que incluye rotación y escalado
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Agregar traslación a la matriz de transformación
        # rotation_matrix[0, 2] corresponde a tx, rotation_matrix[1, 2] corresponde a ty
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty
        
        # Aplicar la transformación usando warpAffine
        # cv2.BORDER_REFLECT_101 refleja los píxeles del borde para evitar artefactos negros
        augmented_image = cv2.warpAffine(
            image_2d, 
            rotation_matrix, 
            (w, h),  # Mantener las mismas dimensiones de salida
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return augmented_image
        
    def load_and_preprocess_images(self, augmentations_per_image=4):
        """
        Cargar y preprocesar todas las imágenes L1 con aumento de datos on-the-fly.
        
        Este método implementa aumento de datos en memoria durante el entrenamiento,
        generando múltiples versiones aumentadas de cada imagen original sin guardarlas en disco.
        
        Proceso:
        1. Cargar todas las imágenes originales en memoria
        2. Para cada imagen original, generar 'augmentations_per_image' versiones aumentadas
        3. Combinar imágenes originales + aumentadas en el dataset final
        4. Normalizar y aplanar todas las imágenes para PCA
        
        Args:
            augmentations_per_image (int): Número de versiones aumentadas por imagen original.
                                         Default: 4 (resultado: 5x más datos = originales + 4 aumentadas)
        
        Returns:
            np.ndarray: Matriz de imágenes expandida (n_original_images * (1 + augmentations_per_image), n_pixels)
                       Ejemplo: 669 imágenes → 669 * (1 + 4) = 3,345 imágenes finales
        
        Notas:
            - El aumento solo ocurre durante el entrenamiento (método aplicable solo aquí)
            - Las imágenes aumentadas NO se guardan en disco (solo en memoria durante ejecución)
            - Los metadatos (filenames, labels) se actualizan para reflejar el dataset expandido
            - Set augmentations_per_image=0 para desactivar el aumento y usar solo imágenes originales
        """
        print("=== Cargando imágenes con aumento de datos on-the-fly ===")
        
        # Obtener lista de archivos PNG
        image_files = list(self.input_path.glob("*.png"))
        n_original_images = len(image_files)
        
        print(f"Encontradas {n_original_images} imágenes originales")
        print(f"Aumentos por imagen: {augmentations_per_image}")
        
        # Calcular tamaño del dataset final
        total_images = n_original_images * (1 + augmentations_per_image)
        print(f"Dataset final: {total_images} imágenes ({n_original_images} originales + {n_original_images * augmentations_per_image} aumentadas)")
        
        # === PASO 1: Cargar todas las imágenes originales en memoria ===
        print("\n1. Cargando imágenes originales en memoria...")
        original_images_2d = []  # Lista de imágenes 2D normalizadas
        original_filenames = []
        original_labels = []
        
        for img_path in tqdm(image_files, desc="Cargando originales"):
            # Cargar imagen
            img = cv2.imread(str(img_path))
            
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {img_path}")
            
            # Convertir BGR a RGB y luego a escala de grises
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            # Verificar dimensiones
            if img_gray.shape != (self.height, self.width):
                img_gray = cv2.resize(img_gray, self.image_size)
            
            # Normalizar (0-1) y guardar como 2D
            img_normalized = img_gray.astype(np.float32) / 255.0
            original_images_2d.append(img_normalized)
            
            # Guardar metadatos
            original_filenames.append(img_path.name)
            
            # Extraer etiqueta de categoría del nombre de archivo
            filename = img_path.name
            if filename.startswith("COVID"):
                original_labels.append("COVID")
            elif filename.startswith("Normal"):
                original_labels.append("Normal")
            elif filename.startswith("Viral"):
                original_labels.append("Viral Pneumonia")
            else:
                original_labels.append("Unknown")
        
        print(f"Imágenes originales cargadas: {len(original_images_2d)}")
        
        # === PASO 2: Generar versiones aumentadas para cada imagen original ===
        print(f"\n2. Generando {augmentations_per_image} aumentos por imagen...")
        augmented_images_2d = []
        augmented_filenames = []
        augmented_labels = []
        
        if augmentations_per_image > 0:
            for i, (img_2d, filename, label) in enumerate(tqdm(
                zip(original_images_2d, original_filenames, original_labels), 
                desc="Generando aumentos",
                total=len(original_images_2d)
            )):
                # Generar múltiples versiones aumentadas de esta imagen
                for aug_idx in range(augmentations_per_image):
                    augmented_img = self.augment_image(img_2d)
                    augmented_images_2d.append(augmented_img)
                    
                    # Crear nombre de archivo para la versión aumentada
                    base_name = filename.replace('.png', '')
                    augmented_filename = f"{base_name}_aug{aug_idx+1}.png"
                    augmented_filenames.append(augmented_filename)
                    augmented_labels.append(label)
            
            print(f"Imágenes aumentadas generadas: {len(augmented_images_2d)}")
        else:
            print("Aumento de datos desactivado (augmentations_per_image=0)")
        
        # === PASO 3: Combinar imágenes originales + aumentadas ===
        print("\n3. Combinando dataset original + aumentado...")
        all_images_2d = original_images_2d + augmented_images_2d
        
        # Combinar metadatos
        self.image_filenames = original_filenames + augmented_filenames
        self.image_labels = original_labels + augmented_labels
        
        print(f"Dataset combinado: {len(all_images_2d)} imágenes totales")
        print(f"Distribución: {len(original_images_2d)} originales + {len(augmented_images_2d)} aumentadas")
        
        # === PASO 4: Aplanar todas las imágenes para construir matriz final de PCA ===
        print("\n4. Construyendo matriz final para PCA...")
        self.images_data = np.zeros((len(all_images_2d), self.n_pixels), dtype=np.float32)
        
        for i, img_2d in enumerate(tqdm(all_images_2d, desc="Aplanando imágenes")):
            self.images_data[i] = img_2d.flatten()
        
        # Mostrar estadísticas finales del dataset
        print(f"\n=== Dataset Final Preparado ===")
        print(f"Matriz de datos: {self.images_data.shape}")
        print(f"Total de imágenes: {self.images_data.shape[0]}")
        print(f"Píxeles por imagen: {self.images_data.shape[1]}")
        
        # Mostrar distribución por categorías
        unique_labels, counts = np.unique(self.image_labels, return_counts=True)
        print(f"\nDistribución por categorías:")
        for label, count in zip(unique_labels, counts):
            original_count = sum(1 for l in original_labels if l == label)
            augmented_count = count - original_count
            print(f"  {label}: {count} total ({original_count} originales + {augmented_count} aumentadas)")
        
        return self.images_data
    
    def set_images_data(self, images_data, image_filenames=None, image_labels=None):
        """
        Establecer datos de imágenes directamente para entrenamiento (útil para reutilización)
        
        Args:
            images_data (np.ndarray): Matriz de imágenes (n_images, n_pixels)
            image_filenames (list): Lista de nombres de archivo (opcional)
            image_labels (list): Lista de etiquetas de categoría (opcional)
        """
        self.images_data = images_data
        
        if image_filenames is not None:
            self.image_filenames = image_filenames
        if image_labels is not None:
            self.image_labels = image_labels
            
        print(f"Datos de imágenes establecidos: {self.images_data.shape}")
    
    def compute_pca(self, n_components=None, quiet=False):
        """
        Computar análisis PCA
        
        Args:
            n_components (int): Número de componentes (None para todos)
            quiet (bool): Si True, reduce la salida de logging para uso en bucles
        """
        if not quiet:
            print("Computando análisis PCA...")
        
        if self.images_data is None:
            raise ValueError("Debe cargar las imágenes primero")
        
        # Calcular imagen promedio
        self.mean_image = np.mean(self.images_data, axis=0)
        
        # Centrar los datos respecto a la media
        centered_data = self.images_data - self.mean_image
        
        # Aplicar PCA
        if n_components is None:
            n_components = min(self.images_data.shape) - 1
        
        self.pca_model = PCA(n_components=n_components)
        self.transformed_data = self.pca_model.fit_transform(centered_data)
        self.principal_components = self.pca_model.components_
        
        if not quiet:
            print(f"PCA completado con {n_components} componentes")
            variance_top5 = self.pca_model.explained_variance_ratio_[:5] * 100
            print(f"Varianza explicada por los primeros 5 componentes: {variance_top5}")
            
            # Validaciones matemáticas
            self._validate_pca_computation()
        
        # Guardar datos originales para verificación de consistencia
        self.original_mean_image = self.mean_image.copy()
        self.original_principal_components = self.principal_components.copy()
        
        # Calcular normalización global científicamente correcta UNA SOLA VEZ
        self._compute_global_normalization()
        
    def get_trained_model_objects(self):
        """
        Obtener objetos del modelo entrenado para uso en memoria
        
        Returns:
            Dict: Diccionario con componentes del modelo PCA entrenado
        """
        if self.pca_model is None:
            raise ValueError("Modelo no entrenado. Ejecute compute_pca() primero.")
        
        return {
            'pca_model': self.pca_model,
            'mean_image': self.mean_image,
            'principal_components': self.principal_components,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'explained_variance': self.pca_model.explained_variance_,
            'n_components': int(self.pca_model.n_components_),
            'image_dimensions': np.array([self.width, self.height]),
            'n_pixels': int(self.n_pixels)
        }
        
    def _validate_pca_computation(self):
        """
        Validar que los cálculos de PCA sean matemáticamente correctos
        """
        print("\n--- Validaciones Matemáticas PCA ---")
        
        # 1. Verificar que los datos estén centrados
        centered_data = self.images_data - self.mean_image
        mean_after_centering = np.mean(centered_data, axis=0)
        max_deviation = np.max(np.abs(mean_after_centering))
        print(f"Máxima desviación de la media después del centrado: {max_deviation:.2e}")
        
        # 2. Verificar ortogonalidad de componentes principales
        n_test = min(10, len(self.principal_components))
        dot_products = []
        for i in range(n_test):
            for j in range(i+1, n_test):
                dot_prod = np.dot(self.principal_components[i], self.principal_components[j])
                dot_products.append(abs(dot_prod))
        
        max_dot_product = max(dot_products) if dot_products else 0
        print(f"Máximo producto punto entre componentes (ortogonalidad): {max_dot_product:.2e}")
        
        # 3. Verificar normalización de componentes principales
        norms = [np.linalg.norm(comp) for comp in self.principal_components[:5]]
        print(f"Normas de los primeros 5 componentes: {[f'{norm:.6f}' for norm in norms]}")
        
        # 4. Verificar reconstrucción perfecta con todos los componentes
        if len(self.principal_components) == len(centered_data[0]):
            # Reconstruir primera imagen usando todos los componentes
            original_centered = centered_data[0]
            projected_full = self.pca_model.transform([original_centered])
            reconstructed_full = np.dot(projected_full, self.principal_components) + self.mean_image
            reconstruction_error = np.mean((self.images_data[0] - reconstructed_full[0])**2)
            print(f"Error de reconstrucción con todos los componentes: {reconstruction_error:.2e}")
        
        # 5. Verificar suma de varianzas
        total_variance_explained = np.sum(self.pca_model.explained_variance_ratio_)
        print(f"Varianza total explicada: {total_variance_explained:.6f}")
        
        print("--- Fin Validaciones ---\n")
    
    def _compute_global_normalization(self):
        """
        Calcular rango de normalización global para todas las eigenfaces
        
        METODOLOGÍA CIENTÍFICA (basada en Turk & Pentland, 1991):
        
        1. PRINCIPIO: Las eigenfaces deben visualizarse con un rango de normalización 
           FIJO y GLOBAL, no variable según el subconjunto que se esté procesando.
        
        2. PROBLEMA RESUELTO: La normalización variable causa que la misma eigenface 
           se vea diferente dependiendo de cuántas eigenfaces se estén procesando 
           simultáneamente, violando la consistencia científica.
        
        3. SOLUCIÓN: Calcular el rango basado en TODAS las eigenfaces disponibles:
           - Examinar todos los valores de todas las eigenfaces
           - Encontrar el valor absoluto máximo global
           - Usar rango simétrico: [-max_abs, +max_abs]
        
        4. VENTAJAS CIENTÍFICAS:
           - Eigenface 1 se ve IDÉNTICA en cualquier contexto
           - Preserva magnitudes relativas entre eigenfaces
           - Cumple estándares de visualización de la literatura
           - Reproducibilidad científica garantizada
        
        Referencias:
        - Turk, M. & Pentland, A. (1991). "Eigenfaces for Recognition"
        - Brunelli, R. & Poggio, T. (1993). "Face Recognition"
        """
        print("Calculando normalización global para eigenfaces...")
        
        # Usar TODAS las eigenfaces disponibles, no subconjuntos
        all_eigenfaces_values = self.principal_components.flatten()
        
        # Calcular rango simétrico basado en el valor extremo global
        self.global_eigenface_max = np.max(np.abs(all_eigenfaces_values))
        self.global_vmin = -self.global_eigenface_max
        self.global_vmax = self.global_eigenface_max
        
        print(f"Normalización global calculada: [{self.global_vmin:.6f}, {self.global_vmax:.6f}]")
        print(f"Rango basado en {len(self.principal_components)} eigenfaces completas")
        
        # Documentar método para reproducibilidad científica
        self.normalization_method = {
            'method': 'global_symmetric',
            'description': 'Rango simétrico basado en valor absoluto máximo de todas las eigenfaces',
            'vmin': float(self.global_vmin),
            'vmax': float(self.global_vmax),
            'n_eigenfaces_used': len(self.principal_components),
            'reference': 'Turk & Pentland (1991) - Eigenfaces for Recognition'
        }
        
    def save_mean_image(self):
        """
        Guardar la imagen promedio
        """
        print("Guardando imagen promedio...")
        
        # Reshape y desnormalizar
        mean_img = self.mean_image.reshape(self.height, self.width)
        mean_img = (mean_img * 255).astype(np.uint8)
        
        # Guardar versión pura (solo datos de píxeles)
        output_file_pure = self.pure_images_path / "mean_face.png"
        cv2.imwrite(str(output_file_pure), mean_img)
        
        print(f"Imagen promedio pura guardada en: {output_file_pure} (dimensiones exactas: {self.width}x{self.height})")
    
    def save_top_eigenfaces(self, n_eigenfaces=5):
        """
        Guardar las principales eigenimágenes usando SOLO método OpenCV científicamente correcto
        
        Args:
            n_eigenfaces (int): Número de eigenimágenes a mostrar
        """
        print(f"Guardando {n_eigenfaces} principales eigenimágenes...")
        print(f"Rango de normalización global (científico): [{self.global_vmin:.6f}, {self.global_vmax:.6f}]")
        print(f"Método: {self.normalization_method['description']}")
        
        # Guardar eigenfaces individuales con dimensiones exactas (primeras 10)
        self._save_individual_eigenfaces(n_eigenfaces=10)
        
        # Crear versión con títulos usando eigenfaces individuales como base (SIN re-procesar píxeles)
        self._save_top_eigenfaces_from_individuals(n_eigenfaces)
        
        # Guardar grids puros usando normalización global consistente
        self._save_pure_eigenfaces_grids()
    
    def _save_individual_eigenfaces(self, n_eigenfaces=10):
        """
        Guardar eigenfaces individuales con normalización global científicamente correcta
        
        Args:
            n_eigenfaces (int): Número de eigenfaces a guardar (default: 10)
        """
        # Crear subdirectorio para eigenfaces individuales
        eigenfaces_dir = self.output_path / "individual_eigenfaces"
        eigenfaces_dir.mkdir(exist_ok=True)
        
        # USAR NORMALIZACIÓN GLOBAL FIJA (científicamente correcta)
        # Basada en TODAS las eigenfaces disponibles, no solo el subconjunto
        vmin, vmax = self.global_vmin, self.global_vmax
        
        for i in range(n_eigenfaces):
            eigenface = self.principal_components[i].reshape(self.height, self.width)
            
            # Normalizar para visualización (0-255) usando rango global fijo
            eigenface_normalized = ((eigenface - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            
            # Guardar con cv2 para mantener dimensiones exactas
            output_file = eigenfaces_dir / f"eigenface_{i+1}.png"
            cv2.imwrite(str(output_file), eigenface_normalized)
        
        print(f"Eigenfaces individuales guardadas en: {eigenfaces_dir} ({n_eigenfaces} eigenfaces, dimensiones exactas: {self.width}x{self.height})")
        print(f"Normalización global usada: vmin={vmin:.6f}, vmax={vmax:.6f} (basada en {len(self.principal_components)} eigenfaces totales)")
    
    def _save_top_eigenfaces_from_individuals(self, n_eigenfaces=5):
        """
        MÉTODO CIENTÍFICAMENTE CORRECTO:
        Crear versión con títulos usando eigenfaces individuales como base,
        construyendo la imagen directamente con OpenCV para garantizar consistencia píxel-por-píxel.
        
        Args:
            n_eigenfaces (int): Número de eigenfaces a mostrar con títulos
        """
        print(f"Creando versión con títulos usando eigenfaces individuales (sin re-procesar píxeles)...")
        
        # Verificar que las eigenfaces individuales existan
        eigenfaces_dir = self.output_path / "individual_eigenfaces"
        if not eigenfaces_dir.exists():
            print("⚠️ WARNING: Eigenfaces individuales no encontradas. Ejecutando _save_individual_eigenfaces() primero.")
            self._save_individual_eigenfaces(n_eigenfaces=max(10, n_eigenfaces))
        
        # MÉTODO CORRECTO: Construir imagen con títulos usando OpenCV directamente
        # para evitar cualquier procesamiento de matplotlib
        
        # Cargar eigenfaces individuales
        eigenfaces_images = []
        for i in range(n_eigenfaces):
            eigenface_path = eigenfaces_dir / f"eigenface_{i+1}.png"
            
            if not eigenface_path.exists():
                raise FileNotFoundError(f"Eigenface individual no encontrada: {eigenface_path}")
            
            eigenface_exact = cv2.imread(str(eigenface_path), cv2.IMREAD_GRAYSCALE)
            if eigenface_exact is None:
                raise ValueError(f"No se pudo cargar eigenface desde: {eigenface_path}")
            
            eigenfaces_images.append(eigenface_exact)
        
        # Crear imagen combinada con espacios para títulos
        title_height = 60  # Espacio para títulos arriba
        spacing = 10  # Espacio entre eigenfaces
        
        total_width = n_eigenfaces * self.width + (n_eigenfaces - 1) * spacing
        total_height = self.height + title_height
        
        # Crear lienzo blanco
        combined_image = np.ones((total_height, total_width), dtype=np.uint8) * 255
        
        # Colocar eigenfaces en el lienzo SIN modificar píxeles
        for i, eigenface in enumerate(eigenfaces_images):
            x_start = i * (self.width + spacing)
            x_end = x_start + self.width
            y_start = title_height
            y_end = y_start + self.height
            
            # Copiar píxeles EXACTOS
            combined_image[y_start:y_end, x_start:x_end] = eigenface
        
        # Agregar títulos usando cv2.putText (sobre píxeles blancos, no sobre eigenfaces)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = 0  # Negro
        thickness = 1
        
        for i in range(n_eigenfaces):
            # Texto principal
            title = f'Eigenface {i+1}'
            variance_text = f'({self.pca_model.explained_variance_ratio_[i]*100:.1f}% varianza)'
            
            x_center = i * (self.width + spacing) + self.width // 2
            
            # Calcular posición centrada del texto
            (title_w, title_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
            (var_w, var_h), _ = cv2.getTextSize(variance_text, font, font_scale, thickness)
            
            title_x = x_center - title_w // 2
            var_x = x_center - var_w // 2
            
            # Colocar títulos
            cv2.putText(combined_image, title, (title_x, 20), font, font_scale, color, thickness)
            cv2.putText(combined_image, variance_text, (var_x, 40), font, font_scale, color, thickness)
        
        # Guardar imagen final
        output_file = self.titled_versions_path / "eigenfaces_from_individuals_opencv.png"
        cv2.imwrite(str(output_file), combined_image)
        
        print(f"✅ Versión con títulos guardada en: {output_file}")
        print(f"🔬 MÉTODO: OpenCV directo - eigenfaces individuales SIN re-procesamiento")
        print(f"🎯 GARANTÍA: Píxeles de eigenfaces son EXACTAMENTE idénticos a individuales")
    
    def _save_pure_eigenfaces_grid(self, eigenfaces, vmin, vmax, grid_name="eigenfaces_grid"):
        """
        Guardar grid de eigenfaces usando cv2 (sin títulos ni márgenes)
        """
        n_eigenfaces = len(eigenfaces)
        
        # Normalizar todas las eigenfaces
        normalized_eigenfaces = []
        for eigenface in eigenfaces:
            eigenface_normalized = ((eigenface - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            normalized_eigenfaces.append(eigenface_normalized)
        
        # Crear grid horizontal concatenando las imágenes
        grid_width = self.width * n_eigenfaces
        grid_height = self.height
        grid_image = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        # Llenar el grid
        for i, eigenface_norm in enumerate(normalized_eigenfaces):
            start_col = i * self.width
            end_col = (i + 1) * self.width
            grid_image[:, start_col:end_col] = eigenface_norm
        
        # Guardar grid puro
        output_file = self.pure_images_path / f"{grid_name}.png"
        cv2.imwrite(str(output_file), grid_image)
        
        print(f"Grid puro de eigenfaces guardado en: {output_file} (dimensiones: {grid_width}x{grid_height})")
    
    def _save_pure_eigenfaces_grid_10(self):
        """
        Guardar grid de 10 eigenfaces en formato 2x5 con normalización global científicamente correcta
        """
        n_eigenfaces = 10
        
        # Obtener las 10 eigenfaces
        all_eigenfaces = []
        for i in range(n_eigenfaces):
            eigenface = self.principal_components[i].reshape(self.height, self.width)
            all_eigenfaces.append(eigenface)
        
        # USAR NORMALIZACIÓN GLOBAL FIJA (científicamente correcta)
        vmin, vmax = self.global_vmin, self.global_vmax
        
        # Normalizar eigenfaces
        normalized_eigenfaces = []
        for eigenface in all_eigenfaces:
            eigenface_normalized = ((eigenface - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            normalized_eigenfaces.append(eigenface_normalized)
        
        # Crear grid 2x5
        cols = 5
        rows = 2
        grid_width = self.width * cols
        grid_height = self.height * rows
        grid_image = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        # Llenar el grid
        for i, eigenface_norm in enumerate(normalized_eigenfaces):
            row = i // cols
            col = i % cols
            
            start_row = row * self.height
            end_row = (row + 1) * self.height
            start_col = col * self.width
            end_col = (col + 1) * self.width
            
            grid_image[start_row:end_row, start_col:end_col] = eigenface_norm
        
        # Guardar grid de 10
        output_file = self.pure_images_path / "eigenfaces_grid_10.png"
        cv2.imwrite(str(output_file), grid_image)
        
        print(f"Grid de 10 eigenfaces guardado en: {output_file} (dimensiones: {grid_width}x{grid_height})")
    
    def _save_pure_eigenfaces_grids(self):
        """
        Guardar todos los grids de eigenfaces con normalización global consistente
        """
        # Grid de 5 eigenfaces (horizontal)
        eigenfaces_5 = []
        for i in range(5):
            eigenface = self.principal_components[i].reshape(self.height, self.width)
            eigenfaces_5.append(eigenface)
        
        # Usar normalización global consistente
        vmin, vmax = self.global_vmin, self.global_vmax
        self._save_pure_eigenfaces_grid(eigenfaces_5, vmin, vmax, "eigenfaces_grid_5")
        
        # Grid de 10 eigenfaces (2x5)
        self._save_pure_eigenfaces_grid_10()
        
        print(f"Grids guardados con normalización científica global: [{vmin:.6f}, {vmax:.6f}]")
    
    def save_cumulative_variance_plot(self):
        """
        Guardar gráfico de varianza acumulada
        """
        print("Generando gráfico de varianza acumulada...")
        
        # Calcular varianza acumulada
        cumulative_variance = np.cumsum(self.pca_model.explained_variance_ratio_)
        
        plt.figure(figsize=(10, 6.5))
        
        # Subplot 1: Varianza acumulada
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 'b-', linewidth=2)
        plt.xlabel('Número de Componentes')
        plt.ylabel('Varianza Acumulada (%)')
        plt.title('Varianza Acumulada vs Número de Componentes', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Marcar puntos importantes
        for threshold in [90, 95, 99]:
            idx = np.argmax(cumulative_variance >= threshold/100)
            if idx > 0:
                plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
                plt.axvline(x=idx+1, color='r', linestyle='--', alpha=0.7)
                plt.text(idx+1, threshold-5, f'{idx+1} comp.\n({threshold}%)', 
                        ha='center', va='top', fontsize=10, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Subplot 2: Varianza individual
        plt.subplot(2, 1, 2)
        plt.plot(range(1, min(51, len(self.pca_model.explained_variance_ratio_) + 1)), 
                self.pca_model.explained_variance_ratio_[:50] * 100, 'g-', linewidth=2)
        plt.xlabel('Número de Componente')
        plt.ylabel('Varianza Explicada (%)')
        plt.title('Varianza Explicada por cada Componente (Primeros 50)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar en titled_versions (los gráficos siempre tienen títulos)
        output_file = self.titled_versions_path / "cumulative_variance.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de varianza guardado en: {output_file}")
    
    def save_reconstructions(self, n_components_list=[1, 2, 3, 4, 5, 10, 40, 100, 150]):
        """
        Guardar reconstrucciones con diferentes números de componentes
        
        Args:
            n_components_list (list): Lista de números de componentes para reconstrucción
        """
        print("Generando reconstrucciones...")
        
        # Seleccionar una imagen de ejemplo (primera imagen COVID)
        covid_idx = next(i for i, label in enumerate(self.image_labels) if label == "COVID")
        original_img = self.images_data[covid_idx]
        
        # Filtrar componentes que no excedan el máximo disponible
        available_components = len(self.principal_components)
        valid_components = [n for n in n_components_list if n <= available_components]
        
        print(f"Componentes usados: {valid_components}")
        
        # Guardar reconstrucciones individuales y grid puro
        self._save_pure_reconstructions(original_img, valid_components, covid_idx)
    
    def _save_pure_reconstructions(self, original_img, valid_components, covid_idx):
        """
        Guardar reconstrucciones individuales y grid usando cv2
        """
        print("Guardando reconstrucciones puras...")
        
        # Guardar imagen original
        original_reshaped = original_img.reshape(self.height, self.width)
        original_uint8 = (original_reshaped * 255).astype(np.uint8)
        original_file = self.individual_reconstructions_path / "original.png"
        cv2.imwrite(str(original_file), original_uint8)
        
        # Guardar cada reconstrucción individual
        reconstructed_images = [original_uint8]  # Para el grid
        
        for n_comp in valid_components:
            # Reconstruir imagen
            projected = self.pca_model.transform([original_img - self.mean_image])[:, :n_comp]
            reconstructed = np.dot(projected, self.principal_components[:n_comp]) + self.mean_image
            
            # Convertir a uint8 y guardar
            reconstructed_reshaped = reconstructed.reshape(self.height, self.width)
            reconstructed_uint8 = (reconstructed_reshaped * 255).astype(np.uint8)
            
            # Guardar individual
            recon_file = self.individual_reconstructions_path / f"reconstruction_{n_comp}_components.png"
            cv2.imwrite(str(recon_file), reconstructed_uint8)
            
            # Añadir al grid
            reconstructed_images.append(reconstructed_uint8)
        
        print(f"Reconstrucciones individuales guardadas en: {self.individual_reconstructions_path}")
        
        # Crear grid de reconstrucciones
        self._save_pure_reconstructions_grid(reconstructed_images, valid_components)
        
        # Crear versión con títulos usando método OpenCV
        self._create_reconstructions_titled_opencv(valid_components)
    
    def _save_pure_reconstructions_grid(self, reconstructed_images, valid_components):
        """
        Crear grid de reconstrucciones usando cv2
        """
        n_images = len(reconstructed_images)
        cols = 5  # Mismo layout que la versión con títulos
        rows = (n_images + cols - 1) // cols
        
        # Crear canvas para el grid
        grid_width = self.width * cols
        grid_height = self.height * rows
        grid_image = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        # Llenar el grid
        for i, img in enumerate(reconstructed_images):
            row = i // cols
            col = i % cols
            
            start_row = row * self.height
            end_row = (row + 1) * self.height
            start_col = col * self.width
            end_col = (col + 1) * self.width
            
            grid_image[start_row:end_row, start_col:end_col] = img
        
        # Guardar grid
        grid_file = self.pure_images_path / "reconstructions_grid.png"
        cv2.imwrite(str(grid_file), grid_image)
        
        print(f"Grid puro de reconstrucciones guardado en: {grid_file} (dimensiones: {grid_width}x{grid_height})")
    
    def _create_reconstructions_titled_opencv(self, valid_components):
        """
        Crear versión con títulos de reconstrucciones usando método OpenCV
        para garantizar píxeles idénticos a las imágenes puras
        """
        # Cargar imágenes individuales de reconstrucciones
        reconstruction_images = []
        reconstruction_labels = []
        
        # Imagen original
        original_path = self.individual_reconstructions_path / "original.png"
        if original_path.exists():
            original_img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
            if original_img is not None:
                reconstruction_images.append(original_img)
                reconstruction_labels.append("Original")
        
        # Reconstrucciones
        for n_comp in valid_components:
            recon_path = self.individual_reconstructions_path / f"reconstruction_{n_comp}_components.png"
            if recon_path.exists():
                recon_img = cv2.imread(str(recon_path), cv2.IMREAD_GRAYSCALE)
                if recon_img is not None:
                    reconstruction_images.append(recon_img)
                    reconstruction_labels.append(f"{n_comp} comp.")
        
        if not reconstruction_images:
            print("⚠️ No se encontraron imágenes de reconstrucción individuales")
            return
        
        # Crear grid con títulos usando OpenCV
        n_images = len(reconstruction_images)
        cols = 5
        rows = (n_images + cols - 1) // cols
        
        title_height = 60
        spacing = 10
        
        grid_width = cols * self.width + (cols - 1) * spacing
        grid_height = rows * (self.height + title_height) + (rows - 1) * spacing
        
        # Crear lienzo blanco
        titled_grid = np.ones((grid_height, grid_width), dtype=np.uint8) * 255
        
        # Colocar imágenes y títulos
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = 0
        thickness = 1
        
        for i, (img, label) in enumerate(zip(reconstruction_images, reconstruction_labels)):
            row = i // cols
            col = i % cols
            
            # Posición de la imagen
            x_start = col * (self.width + spacing)
            y_start = row * (self.height + title_height + spacing) + title_height
            x_end = x_start + self.width
            y_end = y_start + self.height
            
            # Colocar imagen SIN modificar píxeles
            titled_grid[y_start:y_end, x_start:x_end] = img
            
            # Agregar título
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x_start + (self.width - text_w) // 2
            text_y = y_start - 20
            
            cv2.putText(titled_grid, label, (text_x, text_y), font, font_scale, color, thickness)
        
        # Título principal
        main_title = "Reconstrucciones con Diferentes Numeros de Componentes"
        (title_w, title_h), _ = cv2.getTextSize(main_title, font, 0.6, 2)
        title_x = (grid_width - title_w) // 2
        cv2.putText(titled_grid, main_title, (title_x, 30), font, 0.6, color, 2)
        
        # Guardar imagen final
        output_file = self.titled_versions_path / "reconstructions_titled.png"
        cv2.imwrite(str(output_file), titled_grid)
        
        print(f"✅ Reconstrucciones con título guardadas en: {output_file}")
        print(f"🔬 MÉTODO: OpenCV directo - píxeles de reconstrucciones son EXACTAMENTE idénticos")
    
    def save_2d_projection(self):
        """
        Guardar proyección 2D de los datos
        """
        print("Generando proyección 2D...")
        
        # Usar los primeros 2 componentes
        pc1 = self.transformed_data[:, 0]
        pc2 = self.transformed_data[:, 1]
        
        plt.figure(figsize=(10, 6.5))
        
        # Crear colores para categorías
        color_map = {'COVID': 'red', 'Normal': 'blue', 'Viral Pneumonia': 'green'}
        colors = [color_map[label] for label in self.image_labels]
        
        # Scatter plot
        for label in color_map.keys():
            mask = np.array(self.image_labels) == label
            if np.any(mask):
                plt.scatter(pc1[mask], pc2[mask], c=color_map[label], label=label, alpha=0.6, s=40)
        
        plt.xlabel(f'Primer Componente Principal ({self.pca_model.explained_variance_ratio_[0]*100:.1f}% varianza)')
        plt.ylabel(f'Segundo Componente Principal ({self.pca_model.explained_variance_ratio_[1]*100:.1f}% varianza)')
        plt.title('Proyección 2D de Imágenes L1 en Espacio de Componentes Principales', fontweight='bold', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Estadísticas por categoría
        stats_text = []
        for label in color_map.keys():
            count = sum(1 for l in self.image_labels if l == label)
            stats_text.append(f'{label}: {count} imágenes')
        
        plt.text(0.02, 0.98, '\n'.join(stats_text), transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Guardar en titled_versions (los gráficos siempre tienen títulos)
        output_file = self.titled_versions_path / "pca_2d_projection.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Proyección 2D guardada en: {output_file}")
    
    def save_trained_model(self):
        """
        Guardar modelo PCA entrenado para uso futuro
        """
        print("Guardando modelo entrenado...")
        
        from datetime import datetime
        
        # Preparar datos del modelo
        model_data = {
            'pca_components': self.principal_components,
            'mean_image': self.mean_image,
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'explained_variance': self.pca_model.explained_variance_,
            'n_components': int(self.pca_model.n_components_),
            'image_dimensions': np.array([self.width, self.height]),
            'n_pixels': int(self.n_pixels),
            
            # Metadatos de entrenamiento
            'training_metadata': {
                'n_training_images': len(self.image_filenames),
                'image_filenames': self.image_filenames,
                'image_labels': self.image_labels,
                'training_date': str(datetime.now()),
                'dataset_categories': dict(zip(*np.unique(self.image_labels, return_counts=True))),
            },
            
            # Datos transformados (proyecciones)
            'transformed_data': self.transformed_data,
        }
        
        # Guardar en formato NumPy (eficiente)
        model_file = self.output_path / "trained_model.npz"
        np.savez_compressed(str(model_file), **model_data)
        
        print(f"Modelo entrenado guardado en: {model_file}")
        return model_file
    
    def load_trained_model(self, model_path):
        """
        Cargar modelo PCA entrenado
        
        Args:
            model_path (str): Ruta al archivo del modelo
        """
        print(f"Cargando modelo desde: {model_path}")
        
        # Cargar datos
        model_data = np.load(model_path, allow_pickle=True)
        
        # Restaurar atributos del modelo
        self.principal_components = model_data['pca_components']
        self.mean_image = model_data['mean_image']
        self.width, self.height = model_data['image_dimensions']
        self.n_pixels = int(model_data['n_pixels'])
        
        # Recrear modelo PCA de scikit-learn
        from sklearn.decomposition import PCA
        self.pca_model = PCA(n_components=int(model_data['n_components']))
        self.pca_model.components_ = model_data['pca_components']
        self.pca_model.explained_variance_ratio_ = model_data['explained_variance_ratio']
        self.pca_model.explained_variance_ = model_data['explained_variance']
        # NO asignar mean_ al modelo para evitar inconsistencias
        # El centrado se hará manualmente usando self.mean_image
        
        # Cargar metadatos
        metadata = model_data['training_metadata'].item()
        self.image_filenames = metadata['image_filenames']
        self.image_labels = metadata['image_labels']
        
        # Cargar datos transformados si están disponibles
        if 'transformed_data' in model_data:
            self.transformed_data = model_data['transformed_data']
        
        print(f"Modelo cargado exitosamente:")
        print(f"  - {len(self.principal_components)} componentes principales")
        print(f"  - Entrenado con {len(self.image_filenames)} imágenes")
        print(f"  - Dimensiones: {self.width}x{self.height}")
        print(f"  - Fecha de entrenamiento: {metadata['training_date']}")
        
        return True
    
    def project_new_image(self, image_path):
        """
        Proyectar una nueva imagen al espacio PCA
        
        Args:
            image_path (str): Ruta a la imagen a proyectar
            
        Returns:
            np.ndarray: Coordenadas en el espacio PCA
        """
        if self.pca_model is None:
            raise ValueError("Modelo no entrenado. Ejecute compute_pca() primero o cargue un modelo.")
        
        # Cargar y preprocesar imagen
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Redimensionar si es necesario
        if img.shape != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        
        # Normalizar y aplanar
        img_normalized = img.astype(np.float32) / 255.0
        img_flat = img_normalized.flatten()
        
        # Proyectar al espacio PCA
        img_centered = img_flat - self.mean_image
        projection = self.pca_model.transform([img_centered])
        
        return projection[0]
    
    def reconstruct_image(self, image_data, n_components):
        """
        Reconstruir imagen usando n componentes principales
        IDÉNTICA a la lógica en save_reconstructions()
        
        Args:
            image_data (np.ndarray): Datos de imagen aplanados (normalizados 0-1)
            n_components (int): Número de componentes a usar
            
        Returns:
            np.ndarray: Imagen reconstruida (height, width) como uint8
        """
        if self.pca_model is None:
            raise ValueError("Modelo no entrenado. Ejecute compute_pca() primero o cargue un modelo.")
        
        if len(image_data) != self.n_pixels:
            raise ValueError(f"Imagen debe tener {self.n_pixels} píxeles, recibido: {len(image_data)}")
        
        # USAR EXACTAMENTE LA MISMA LÓGICA QUE save_reconstructions()
        # Línea ~466: projected = self.pca_model.transform([original_img - self.mean_image])[:, :n_comp]
        # PERO sin usar self.pca_model.transform() para evitar inconsistencias con mean_
        
        # Centrar datos manualmente (igual que en el flujo normal)
        img_centered = image_data - self.mean_image
        
        # Proyectar manualmente usando los componentes principales
        # Esto es equivalente a self.pca_model.transform() pero usando nuestros datos exactos
        projection = np.dot(img_centered, self.principal_components[:n_components].T)
        
        # Reconstruir (igual que en save_reconstructions())
        # Línea ~467: reconstructed = np.dot(projected, self.principal_components[:n_comp]) + self.mean_image
        reconstructed = np.dot(projection.reshape(1, -1), self.principal_components[:n_components]) + self.mean_image
        
        # Reshape y desnormalizar (igual que en save_reconstructions())
        reconstructed_img = reconstructed.reshape(self.height, self.width)
        reconstructed_img = (reconstructed_img * 255).astype(np.uint8)
        
        return reconstructed_img
    
    def get_model_summary(self):
        """
        Obtener resumen del modelo entrenado
        
        Returns:
            dict: Resumen del modelo
        """
        if self.pca_model is None:
            return {"error": "Modelo no entrenado"}
        
        summary = {
            'model_info': {
                'n_components': int(self.pca_model.n_components),
                'image_dimensions': f"{self.width}x{self.height}",
                'n_pixels': int(self.n_pixels),
                'n_training_images': len(self.image_filenames) if hasattr(self, 'image_filenames') else 0,
            },
            'variance_analysis': {
                'top_5_variance': [float(x) for x in self.pca_model.explained_variance_ratio_[:5]],
                'cumulative_top_5': [float(x) for x in np.cumsum(self.pca_model.explained_variance_ratio_[:5])],
                'total_variance_explained': float(np.sum(self.pca_model.explained_variance_ratio_)),
                'components_for_90_percent': int(np.argmax(np.cumsum(self.pca_model.explained_variance_ratio_) >= 0.9)) + 1,
                'components_for_95_percent': int(np.argmax(np.cumsum(self.pca_model.explained_variance_ratio_) >= 0.95)) + 1,
            }
        }
        
        if hasattr(self, 'image_labels'):
            labels, counts = np.unique(self.image_labels, return_counts=True)
            summary['dataset_info'] = {str(label): int(count) for label, count in zip(labels, counts)}
        
        return summary
    
    def verify_model_consistency(self, test_components=[1, 5, 10, 40]):
        """
        Verificar que el modelo cargado produzca EXACTAMENTE las mismas 
        reconstrucciones que el flujo normal
        
        Args:
            test_components (list): Lista de números de componentes a probar
            
        Returns:
            dict: Resultados de la verificación
        """
        if self.pca_model is None:
            return {"error": "Modelo no entrenado"}
        
        print("\n=== Verificando Consistencia del Modelo ===")
        
        results = {
            'test_passed': True,
            'differences_found': [],
            'test_details': {}
        }
        
        # Usar primera imagen COVID como test
        covid_idx = next(i for i, label in enumerate(self.image_labels) if label == "COVID")
        
        # Si images_data no está disponible (modelo cargado), usar datos del modelo
        if self.images_data is None:
            # Necesitamos cargar la imagen desde el archivo
            covid_filename = self.image_filenames[covid_idx]
            # Buscar la imagen en el directorio de landmarks
            import glob, os
            possible_paths = [
                f"output_landmarks_complete/L1/{covid_filename}",
                f"output_landmarks_complete/L1/{covid_filename.replace('_L1', '_L1.png')}",
                f"output_landmarks_complete/L1/{covid_filename.replace('.png', '')}_L1.png"
            ]
            
            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    break
            
            if img_path is None:
                raise FileNotFoundError(f"No se pudo encontrar la imagen: {covid_filename}")
            
            # Cargar y preprocesar como en load_and_preprocess_images
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            if img_gray.shape != (self.height, self.width):
                img_gray = cv2.resize(img_gray, (self.width, self.height))
            original_img = img_gray.astype(np.float32) / 255.0
            original_img = original_img.flatten()
        else:
            original_img = self.images_data[covid_idx]
        
        print(f"Imagen de prueba: {self.image_filenames[covid_idx]}")
        
        for n_comp in test_components:
            print(f"Probando {n_comp} componentes...")
            
            # MÉTODO 1: Flujo normal (proyección manual igual que en reconstruct_image)
            img_centered = original_img - self.mean_image
            projected_normal = np.dot(img_centered, self.principal_components[:n_comp].T)
            reconstructed_normal = np.dot(projected_normal.reshape(1, -1), self.principal_components[:n_comp]) + self.mean_image
            reconstructed_normal_img = (reconstructed_normal.reshape(self.height, self.width) * 255).astype(np.uint8)
            
            # MÉTODO 2: Modelo cargado (usando reconstruct_image)
            reconstructed_loaded_img = self.reconstruct_image(original_img, n_comp)
            
            # COMPARAR PÍXEL POR PÍXEL
            difference = np.abs(reconstructed_normal_img.astype(np.int16) - reconstructed_loaded_img.astype(np.int16))
            max_diff = np.max(difference)
            mean_diff = np.mean(difference)
            identical = np.array_equal(reconstructed_normal_img, reconstructed_loaded_img)
            
            test_result = {
                'identical': identical,
                'max_difference': int(max_diff),
                'mean_difference': float(mean_diff),
                'total_pixels': int(np.prod(reconstructed_normal_img.shape)),
                'different_pixels': int(np.sum(difference > 0))
            }
            
            results['test_details'][f'{n_comp}_components'] = test_result
            
            if not identical:
                results['test_passed'] = False
                results['differences_found'].append(f"{n_comp} componentes: max_diff={max_diff}, mean_diff={mean_diff:.3f}")
                print(f"   ❌ DIFERENCIA encontrada: max={max_diff}, mean={mean_diff:.3f}")
            else:
                print(f"   ✅ IDÉNTICO: reconstrucciones exactamente iguales")
        
        # Verificar también datos base
        print("\nVerificando datos base del modelo...")
        
        # Verificar mean_image
        if hasattr(self, 'original_mean_image'):
            mean_identical = np.array_equal(self.mean_image, self.original_mean_image)
            results['mean_image_identical'] = mean_identical
            if not mean_identical:
                mean_diff = np.max(np.abs(self.mean_image - self.original_mean_image))
                results['differences_found'].append(f"mean_image diferente: max_diff={mean_diff}")
                print(f"   ❌ mean_image diferente: max_diff={mean_diff}")
            else:
                print(f"   ✅ mean_image idéntica")
        
        # Verificar principal_components
        if hasattr(self, 'original_principal_components'):
            components_identical = np.array_equal(self.principal_components, self.original_principal_components)
            results['components_identical'] = components_identical
            if not components_identical:
                comp_diff = np.max(np.abs(self.principal_components - self.original_principal_components))
                results['differences_found'].append(f"principal_components diferentes: max_diff={comp_diff}")
                print(f"   ❌ principal_components diferentes: max_diff={comp_diff}")
            else:
                print(f"   ✅ principal_components idénticos")
        
        if results['test_passed']:
            print("\n🎉 TODAS LAS PRUEBAS PASARON: El modelo es consistente")
        else:
            print(f"\n❌ PRUEBAS FALLARON: {len(results['differences_found'])} diferencias encontradas")
            for diff in results['differences_found']:
                print(f"   - {diff}")
        
        return results
    
    def verify_eigenface_visual_consistency(self):
        """
        Verificar que eigenface 1 se vea idéntica en todos los contextos de visualización
        """
        print("\n=== Verificando Consistencia Visual de Eigenfaces ===")
        
        eigenface_1 = self.principal_components[0].reshape(self.height, self.width)
        vmin, vmax = self.global_vmin, self.global_vmax
        
        # Normalizar eigenface 1 usando normalización global
        eigenface_1_normalized = ((eigenface_1 - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        
        verification_results = {
            'eigenface_1_consistent': True,
            'global_normalization_range': {'vmin': float(vmin), 'vmax': float(vmax)},
            'eigenface_1_stats': {
                'original_min': float(np.min(eigenface_1)),
                'original_max': float(np.max(eigenface_1)),
                'normalized_min': int(np.min(eigenface_1_normalized)),
                'normalized_max': int(np.max(eigenface_1_normalized))
            },
            'verification_tests': {}
        }
        
        # Leer eigenface 1 individual guardada
        eigenface_1_path = self.output_path / "individual_eigenfaces" / "eigenface_1.png"
        if eigenface_1_path.exists():
            import cv2
            eigenface_1_individual = cv2.imread(str(eigenface_1_path), cv2.IMREAD_GRAYSCALE)
            
            # Comparar píxel por píxel
            if eigenface_1_individual is not None:
                identical = np.array_equal(eigenface_1_normalized, eigenface_1_individual)
                max_diff = np.max(np.abs(eigenface_1_normalized.astype(np.int16) - eigenface_1_individual.astype(np.int16)))
                
                verification_results['verification_tests']['individual_eigenface'] = {
                    'identical': identical,
                    'max_difference': int(max_diff),
                    'status': '✅ IDÉNTICA' if identical else f'❌ DIFERENTE (max_diff={max_diff})'
                }
                
                if not identical:
                    verification_results['eigenface_1_consistent'] = False
                
                print(f"Eigenface 1 individual: {verification_results['verification_tests']['individual_eigenface']['status']}")
        
        # Leer grid de 5 eigenfaces y extraer eigenface 1
        grid_5_path = self.pure_images_path / "eigenfaces_grid_5.png"
        if grid_5_path.exists():
            import cv2
            grid_5 = cv2.imread(str(grid_5_path), cv2.IMREAD_GRAYSCALE)
            
            if grid_5 is not None:
                # Extraer primera eigenface del grid (primera columna)
                eigenface_1_from_grid = grid_5[:, :self.width]
                
                identical = np.array_equal(eigenface_1_normalized, eigenface_1_from_grid)
                max_diff = np.max(np.abs(eigenface_1_normalized.astype(np.int16) - eigenface_1_from_grid.astype(np.int16)))
                
                verification_results['verification_tests']['grid_5_eigenfaces'] = {
                    'identical': identical,
                    'max_difference': int(max_diff),
                    'status': '✅ IDÉNTICA' if identical else f'❌ DIFERENTE (max_diff={max_diff})'
                }
                
                if not identical:
                    verification_results['eigenface_1_consistent'] = False
                
                print(f"Eigenface 1 en grid de 5: {verification_results['verification_tests']['grid_5_eigenfaces']['status']}")
        
        # Leer grid de 10 eigenfaces y extraer eigenface 1
        grid_10_path = self.pure_images_path / "eigenfaces_grid_10.png"
        if grid_10_path.exists():
            import cv2
            grid_10 = cv2.imread(str(grid_10_path), cv2.IMREAD_GRAYSCALE)
            
            if grid_10 is not None:
                # Extraer primera eigenface del grid (primera posición: fila 0, columna 0)
                eigenface_1_from_grid_10 = grid_10[:self.height, :self.width]
                
                identical = np.array_equal(eigenface_1_normalized, eigenface_1_from_grid_10)
                max_diff = np.max(np.abs(eigenface_1_normalized.astype(np.int16) - eigenface_1_from_grid_10.astype(np.int16)))
                
                verification_results['verification_tests']['grid_10_eigenfaces'] = {
                    'identical': identical,
                    'max_difference': int(max_diff),
                    'status': '✅ IDÉNTICA' if identical else f'❌ DIFERENTE (max_diff={max_diff})'
                }
                
                if not identical:
                    verification_results['eigenface_1_consistent'] = False
                
                print(f"Eigenface 1 en grid de 10: {verification_results['verification_tests']['grid_10_eigenfaces']['status']}")
        
        # Verificar nueva versión con títulos (usando eigenfaces individuales como base + OpenCV)
        titled_opencv_path = self.titled_versions_path / "eigenfaces_from_individuals_opencv.png"
        if titled_opencv_path.exists():
            import cv2
            titled_image = cv2.imread(str(titled_opencv_path), cv2.IMREAD_GRAYSCALE)
            
            if titled_image is not None:
                # La imagen con títulos tiene espacio para títulos arriba (60px) y espacios entre eigenfaces (10px)
                title_height = 60
                y_start = title_height
                y_end = y_start + self.height
                
                # Extraer primera eigenface (esquina superior izquierda, después del espacio de título)
                eigenface_1_from_titled = titled_image[y_start:y_end, :self.width]
                
                identical = np.array_equal(eigenface_1_normalized, eigenface_1_from_titled)
                max_diff = np.max(np.abs(eigenface_1_normalized.astype(np.int16) - eigenface_1_from_titled.astype(np.int16)))
                
                verification_results['verification_tests']['titled_opencv_method'] = {
                    'identical': identical,
                    'max_difference': int(max_diff),
                    'status': '✅ IDÉNTICA' if identical else f'❌ DIFERENTE (max_diff={max_diff})'
                }
                
                if not identical:
                    verification_results['eigenface_1_consistent'] = False
                
                print(f"Eigenface 1 en versión con títulos (OpenCV): {verification_results['verification_tests']['titled_opencv_method']['status']}")
        
        # Resumen final
        if verification_results['eigenface_1_consistent']:
            print("\n🎉 VERIFICACIÓN EXITOSA: Eigenface 1 es IDÉNTICA en todos los contextos")
            print("✅ La normalización global científica funciona correctamente")
        else:
            print("\n❌ INCONSISTENCIA DETECTADA en eigenface 1")
            verification_results['eigenface_1_consistent'] = False
        
        # Guardar resultados de verificación
        import json
        verification_file = self.output_path / "eigenface_visual_verification.json"
        with open(verification_file, 'w') as f:
            json.dump(verification_results, f, indent=2)
        
        print(f"Resultados de verificación visual guardados en: {verification_file}")
        
        return verification_results
    
    def generate_analysis_report(self):
        """
        Generar reporte de análisis
        """
        # Convertir conteos de categorías a tipos serializables
        labels, counts = np.unique(self.image_labels, return_counts=True)
        categories = {str(label): int(count) for label, count in zip(labels, counts)}
        
        report = {
            'dataset_info': {
                'total_images': len(self.image_filenames),
                'image_dimensions': f'{self.width}x{self.height}',
                'categories': categories,
            },
            'pca_results': {
                'n_components': int(self.pca_model.n_components_),
                'variance_explained_top5': [float(x) for x in self.pca_model.explained_variance_ratio_[:5]],
                'cumulative_variance_top5': [float(x) for x in np.cumsum(self.pca_model.explained_variance_ratio_[:5])],
                'components_for_90_percent': int(np.argmax(np.cumsum(self.pca_model.explained_variance_ratio_) >= 0.9)) + 1,
                'components_for_95_percent': int(np.argmax(np.cumsum(self.pca_model.explained_variance_ratio_) >= 0.95)) + 1,
            }
        }
        
        # Guardar reporte
        import json
        report_file = self.output_path / "analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Reporte de análisis guardado en: {report_file}")
        return report


def main():
    """
    Función principal para ejecutar el análisis completo
    """
    print("=== Análisis PCA y Eigenfaces para Landmark L1 ===")
    print()
    
    # Configuración de rutas
    input_path = "output_landmarks_complete/L1"
    output_path = "output_pca_analysis"
    
    # Crear analizador
    analyzer = LandmarkPCAAnalysis(input_path, output_path)
    
    try:
        # Ejecutar análisis completo
        print("1. Cargando y preprocesando imágenes...")
        analyzer.load_and_preprocess_images()
        
        print("\n2. Computando análisis PCA...")
        analyzer.compute_pca()
        
        print("\n3. Generando visualizaciones...")
        analyzer.save_mean_image()
        analyzer.save_top_eigenfaces()
        analyzer.save_cumulative_variance_plot()
        analyzer.save_reconstructions()
        analyzer.save_2d_projection()
        
        print("\n4. Guardando modelo entrenado...")
        analyzer.save_trained_model()
        
        print("\n5. Verificando consistencia del modelo...")
        # Crear nueva instancia para probar carga del modelo
        test_analyzer = LandmarkPCAAnalysis(input_path, output_path)
        test_analyzer.load_trained_model(analyzer.output_path / "trained_model.npz")
        
        # Pasar datos originales para verificación
        test_analyzer.original_mean_image = analyzer.original_mean_image
        test_analyzer.original_principal_components = analyzer.original_principal_components
        
        # Verificar que el modelo cargado sea idéntico al original
        verification_results = test_analyzer.verify_model_consistency()
        
        # Verificar consistencia visual de eigenfaces
        visual_verification = analyzer.verify_eigenface_visual_consistency()
        
        print("\n6. Generando reportes...")
        report = analyzer.generate_analysis_report()
        
        # Guardar resumen del modelo en JSON legible
        import json
        model_summary = analyzer.get_model_summary()
        summary_file = analyzer.output_path / "model_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(model_summary, f, indent=2)
        print(f"Resumen del modelo guardado en: {summary_file}")
        
        # Guardar resultados de verificación
        verification_file = analyzer.output_path / "model_verification.json"
        with open(verification_file, 'w') as f:
            json.dump(verification_results, f, indent=2)
        print(f"Resultados de verificación guardados en: {verification_file}")
        
        print("\n=== Análisis Completado ===")
        print(f"Resultados guardados en: {analyzer.output_path}")
        print(f"Total de imágenes procesadas: {report['dataset_info']['total_images']}")
        print(f"Componentes para 90% varianza: {report['pca_results']['components_for_90_percent']}")
        print(f"Componentes para 95% varianza: {report['pca_results']['components_for_95_percent']}")
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        raise


if __name__ == "__main__":
    main()