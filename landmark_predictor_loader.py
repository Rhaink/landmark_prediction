#!/usr/bin/env python3
"""
Sistema de Predicción de Landmarks Médicos
==========================================

Este script implementa funcionalidades de predicción para landmarks médicos
utilizando los modelos PCA entrenados previamente.

Funcionalidades:
- Carga de modelos PCA entrenados para landmarks L1-L15
- Predicción de coordenadas de landmarks en nuevas imágenes
- Proyección de imágenes al espacio PCA
- Reconstrucción y análisis de landmarks

Autor: Sistema de desarrollo con IA especializada
Fecha: 2025-08-20
Basado en: pca_eigenfaces_analysis.py y multi_landmark_pca_analysis.py
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
from sklearn.decomposition import PCA
import json
import pandas as pd

class LandmarkPredictor:
    """
    Clase principal para predicción de landmarks médicos usando modelos PCA entrenados
    """
    
    def __init__(self, models_base_path="output_pca_analysis_all_landmarks"):
        """
        Inicializar el predictor de landmarks
        
        Args:
            models_base_path (str): Ruta base donde están los modelos entrenados
        """
        self.models_base_path = Path(models_base_path)
        
        # Validar que el directorio de modelos existe
        if not self.models_base_path.exists():
            raise FileNotFoundError(f"Directorio de modelos no encontrado: {self.models_base_path}")
        
        # Configurar logging
        self._setup_logging()
        
        # Almacenar modelos cargados
        self.loaded_models = {}
        
        # Almacenar datos cargados
        self.test_coordinates = None
        self.dataset_images = None
        self.bounding_boxes_config = None
        self.optimal_templates_config = None
        
        # Detectar modelos disponibles
        self.available_landmarks = self._detect_available_models()
        
        self.logger.info(f"LandmarkPredictor inicializado")
        self.logger.info(f"Modelos disponibles: {self.available_landmarks}")
        self.logger.info(f"Directorio de modelos: {self.models_base_path}")
    
    def _setup_logging(self):
        """
        Configurar sistema de logging
        """
        # Configurar logging básico
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== Inicio del Sistema de Predicción de Landmarks ===")
    
    def _detect_available_models(self) -> List[str]:
        """
        Detectar automáticamente los modelos entrenados disponibles (L1-L15)
        
        Returns:
            List[str]: Lista de landmarks con modelos disponibles
        """
        landmarks = []
        
        for i in range(1, 16):  # L1 a L15
            landmark_dir = self.models_base_path / f"L{i}"
            model_file = landmark_dir / "trained_model.npz"
            
            if landmark_dir.exists() and model_file.exists():
                landmarks.append(f"L{i}")
                self.logger.info(f"Modelo detectado para L{i}: {model_file}")
            else:
                self.logger.warning(f"Modelo L{i}: no encontrado en {landmark_dir}")
        
        return landmarks
    
    def load_pca_model(self, landmark: str) -> Dict:
        """
        Cargar modelo PCA entrenado para un landmark específico
        
        Args:
            landmark (str): Nombre del landmark (ej: "L1")
            
        Returns:
            Dict: Información del modelo cargado
        """
        if landmark not in self.available_landmarks:
            raise ValueError(f"Modelo para {landmark} no está disponible. Landmarks disponibles: {self.available_landmarks}")
        
        self.logger.info(f"Cargando modelo PCA para {landmark}...")
        
        # Ruta al modelo
        model_path = self.models_base_path / landmark / "trained_model.npz"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {model_path}")
        
        try:
            # Cargar datos del modelo
            model_data = np.load(model_path, allow_pickle=True)
            
            # Extraer componentes principales del modelo
            model_info = {
                'landmark': landmark,
                'pca_components': model_data['pca_components'],
                'mean_image': model_data['mean_image'],
                'explained_variance_ratio': model_data['explained_variance_ratio'],
                'explained_variance': model_data['explained_variance'],
                'n_components': int(model_data['n_components']),
                'image_dimensions': tuple(model_data['image_dimensions']),
                'n_pixels': int(model_data['n_pixels']),
                'model_path': str(model_path)
            }
            
            # Cargar metadatos de entrenamiento
            if 'training_metadata' in model_data:
                metadata = model_data['training_metadata'].item()
                model_info['training_metadata'] = {
                    'n_training_images': metadata['n_training_images'],
                    'training_date': metadata['training_date'],
                    'dataset_categories': metadata['dataset_categories']
                }
            
            # Cargar datos transformados si están disponibles
            if 'transformed_data' in model_data:
                model_info['transformed_data'] = model_data['transformed_data']
            
            # Recrear modelo PCA de scikit-learn para uso futuro
            pca_model = PCA(n_components=model_info['n_components'])
            pca_model.components_ = model_info['pca_components']
            pca_model.explained_variance_ratio_ = model_info['explained_variance_ratio']
            pca_model.explained_variance_ = model_info['explained_variance']
            model_info['pca_model'] = pca_model
            
            # Almacenar en cache
            self.loaded_models[landmark] = model_info
            
            # Información de carga exitosa
            width, height = model_info['image_dimensions']
            n_components = model_info['n_components']
            variance_top5 = model_info['explained_variance_ratio'][:5] * 100
            
            self.logger.info(f"✅ Modelo {landmark} cargado exitosamente:")
            self.logger.info(f"  - Dimensiones: {width}x{height} píxeles")
            self.logger.info(f"  - Componentes PCA: {n_components}")
            self.logger.info(f"  - Imágenes de entrenamiento: {model_info.get('training_metadata', {}).get('n_training_images', 'N/A')}")
            self.logger.info(f"  - Varianza explicada (top 5): {[f'{v:.1f}%' for v in variance_top5]}")
            
            if 'training_metadata' in model_info:
                self.logger.info(f"  - Fecha de entrenamiento: {model_info['training_metadata']['training_date']}")
                self.logger.info(f"  - Categorías: {model_info['training_metadata']['dataset_categories']}")
            
            return model_info
            
        except Exception as e:
            error_msg = f"Error cargando modelo {landmark}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_model_info(self, landmark: str) -> Dict:
        """
        Obtener información de un modelo cargado
        
        Args:
            landmark (str): Nombre del landmark
            
        Returns:
            Dict: Información del modelo
        """
        if landmark not in self.loaded_models:
            raise ValueError(f"Modelo {landmark} no está cargado. Use load_pca_model() primero.")
        
        return self.loaded_models[landmark]
    
    def is_model_loaded(self, landmark: str) -> bool:
        """
        Verificar si un modelo está cargado
        
        Args:
            landmark (str): Nombre del landmark
            
        Returns:
            bool: True si el modelo está cargado
        """
        return landmark in self.loaded_models
    
    def get_loaded_landmarks(self) -> List[str]:
        """
        Obtener lista de landmarks con modelos cargados
        
        Returns:
            List[str]: Lista de landmarks cargados
        """
        return list(self.loaded_models.keys())
    
    def load_all_available_models(self) -> Dict[str, bool]:
        """
        Cargar todos los modelos disponibles
        
        Returns:
            Dict[str, bool]: Estado de carga por landmark
        """
        self.logger.info(f"Cargando todos los modelos disponibles: {self.available_landmarks}")
        
        loading_results = {}
        
        for landmark in self.available_landmarks:
            try:
                self.load_pca_model(landmark)
                loading_results[landmark] = True
                self.logger.info(f"✅ {landmark} cargado exitosamente")
            except Exception as e:
                loading_results[landmark] = False
                self.logger.error(f"❌ Error cargando {landmark}: {e}")
        
        successful_loads = sum(loading_results.values())
        total_models = len(loading_results)
        
        self.logger.info(f"Carga completada: {successful_loads}/{total_models} modelos cargados exitosamente")
        
        return loading_results
    
    def load_test_coordinates(self, csv_path="data/coordenadas/coordenadas_test.csv") -> Dict:
        """
        Cargar datos de prueba desde archivo CSV
        
        Args:
            csv_path (str): Ruta al archivo CSV de coordenadas de prueba
            
        Returns:
            Dict: Datos de coordenadas organizados por imagen
        """
        self.logger.info(f"Cargando datos de prueba desde: {csv_path}")
        
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Archivo CSV no encontrado: {csv_file}")
        
        try:
            # Leer CSV sin header
            df = pd.read_csv(csv_file, header=None)
            
            # Validar formato esperado (32 columnas)
            if df.shape[1] != 32:
                raise ValueError(f"Formato CSV incorrecto. Se esperan 32 columnas, encontradas: {df.shape[1]}")
            
            self.logger.info(f"CSV cargado: {len(df)} muestras, {df.shape[1]} columnas")
            
            # Procesar cada fila
            test_data = {}
            category_counts = {"COVID": 0, "Normal": 0, "Viral Pneumonia": 0, "Unknown": 0}
            
            for index, row in df.iterrows():
                # Extraer filename (última columna)
                filename = row.iloc[-1]
                
                # Extraer coordenadas (columnas 1-30, excluyendo índice y filename)
                coords_raw = row.iloc[1:-1].values
                
                # Validar que tenemos 30 valores (15 pares x,y)
                if len(coords_raw) != 30:
                    self.logger.warning(f"Coordenadas incompletas para {filename}: {len(coords_raw)} valores")
                    continue
                
                # Convertir a pares (x,y)
                landmarks = []
                for i in range(0, 30, 2):
                    x = float(coords_raw[i])
                    y = float(coords_raw[i+1])
                    landmarks.append((x, y))
                
                # Determinar categoría desde filename
                category = "Unknown"
                if filename.startswith("COVID"):
                    category = "COVID"
                elif filename.startswith("Normal"):
                    category = "Normal"
                elif filename.startswith("Viral"):
                    category = "Viral Pneumonia"
                
                category_counts[category] += 1
                
                # Almacenar datos
                test_data[filename] = {
                    'landmarks': landmarks,
                    'category': category,
                    'index': int(row.iloc[0]),
                    'n_landmarks': len(landmarks)
                }
            
            # Almacenar en cache
            self.test_coordinates = {
                'data': test_data,
                'metadata': {
                    'csv_path': str(csv_file),
                    'total_samples': len(test_data),
                    'category_distribution': category_counts,
                    'loaded_at': str(datetime.now())
                }
            }
            
            self.logger.info(f"✅ Datos de prueba cargados exitosamente:")
            self.logger.info(f"  - Total de muestras: {len(test_data)}")
            self.logger.info(f"  - Distribución por categoría: {category_counts}")
            self.logger.info(f"  - Landmarks por imagen: {len(landmarks)}")
            
            return self.test_coordinates
            
        except Exception as e:
            error_msg = f"Error cargando datos de prueba: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_dataset_images(self, dataset_path="data/dataset", categories=None) -> Dict:
        """
        Cargar imágenes originales del dataset por categoría
        
        Args:
            dataset_path (str): Ruta al directorio del dataset
            categories (Optional[List[str]]): Categorías específicas a cargar
            
        Returns:
            Dict: Imágenes organizadas por categoría
        """
        self.logger.info(f"Cargando imágenes del dataset desde: {dataset_path}")
        
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Directorio del dataset no encontrado: {dataset_dir}")
        
        # Categorías disponibles
        available_categories = ["COVID", "Normal", "Viral_Pneumonia"]
        if categories is None:
            categories = available_categories
        
        try:
            dataset_images = {}
            total_images = 0
            
            for category in categories:
                if category not in available_categories:
                    self.logger.warning(f"Categoría no reconocida: {category}")
                    continue
                
                category_dir = dataset_dir / category
                if not category_dir.exists():
                    self.logger.warning(f"Directorio de categoría no encontrado: {category_dir}")
                    continue
                
                self.logger.info(f"Cargando categoría: {category}")
                
                # Buscar archivos PNG
                png_files = list(category_dir.glob("*.png"))
                
                if not png_files:
                    self.logger.warning(f"No se encontraron imágenes PNG en: {category_dir}")
                    continue
                
                category_images = {}
                
                for img_path in png_files:
                    try:
                        # Cargar imagen
                        img = cv2.imread(str(img_path))
                        
                        if img is None:
                            self.logger.warning(f"No se pudo cargar imagen: {img_path}")
                            continue
                        
                        # Convertir BGR a RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Validar dimensiones (esperadas: 299x299)
                        if img_rgb.shape[:2] != (299, 299):
                            self.logger.warning(f"Dimensiones incorrectas en {img_path.name}: {img_rgb.shape[:2]}")
                            # Redimensionar si es necesario
                            img_rgb = cv2.resize(img_rgb, (299, 299))
                        
                        # Almacenar imagen
                        category_images[img_path.name] = img_rgb
                        
                    except Exception as e:
                        self.logger.error(f"Error cargando {img_path}: {e}")
                        continue
                
                dataset_images[category] = category_images
                total_images += len(category_images)
                
                self.logger.info(f"  - {category}: {len(category_images)} imágenes cargadas")
            
            # Almacenar en cache
            self.dataset_images = {
                'data': dataset_images,
                'metadata': {
                    'dataset_path': str(dataset_dir),
                    'categories_loaded': list(dataset_images.keys()),
                    'total_images': total_images,
                    'loaded_at': str(datetime.now())
                }
            }
            
            self.logger.info(f"✅ Imágenes del dataset cargadas exitosamente:")
            self.logger.info(f"  - Total de imágenes: {total_images}")
            self.logger.info(f"  - Categorías: {list(dataset_images.keys())}")
            
            return self.dataset_images
            
        except Exception as e:
            error_msg = f"Error cargando imágenes del dataset: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_bounding_boxes_config(self, json_path="landmark_bounding_boxes_corrected.json") -> Dict:
        """
        Cargar configuración de bounding boxes desde archivo JSON
        
        Args:
            json_path (str): Ruta al archivo JSON de bounding boxes
            
        Returns:
            Dict: Configuración de bounding boxes
        """
        self.logger.info(f"Cargando configuración de bounding boxes desde: {json_path}")
        
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Archivo JSON no encontrado: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                bbox_config = json.load(f)
            
            # Validar estructura esperada
            required_keys = ['metadata', 'global_statistics', 'landmark_bounding_boxes']
            for key in required_keys:
                if key not in bbox_config:
                    raise ValueError(f"Clave requerida '{key}' no encontrada en JSON")
            
            # Validar landmarks L1-L15
            landmarks_found = []
            landmark_boxes = bbox_config['landmark_bounding_boxes']
            
            for i in range(1, 16):
                landmark_key = f"L{i}"
                if landmark_key in landmark_boxes:
                    landmarks_found.append(landmark_key)
                else:
                    self.logger.warning(f"Landmark {landmark_key} no encontrado en configuración")
            
            # Almacenar en cache
            self.bounding_boxes_config = {
                'config': bbox_config,
                'landmarks_available': landmarks_found,
                'loaded_from': str(json_file),
                'loaded_at': str(datetime.now())
            }
            
            # Información de metadatos
            metadata = bbox_config['metadata']
            
            self.logger.info(f"✅ Configuración de bounding boxes cargada exitosamente:")
            self.logger.info(f"  - Imágenes analizadas: {metadata.get('total_images_analyzed', 'N/A')}")
            self.logger.info(f"  - Landmarks por imagen: {metadata.get('landmarks_per_image', 'N/A')}")
            self.logger.info(f"  - Algoritmo: {metadata.get('algorithm', 'N/A')}")
            self.logger.info(f"  - Landmarks encontrados: {len(landmarks_found)} ({landmarks_found[:5]}...)")
            
            return self.bounding_boxes_config
            
        except Exception as e:
            error_msg = f"Error cargando configuración de bounding boxes: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_optimal_templates_config(self, json_path="optimal_templates_fixed.json") -> Dict:
        """
        Cargar configuración de templates óptimos desde archivo JSON
        
        Args:
            json_path (str): Ruta al archivo JSON de templates óptimos
            
        Returns:
            Dict: Configuración de templates óptimos
        """
        self.logger.info(f"Cargando configuración de templates óptimos desde: {json_path}")
        
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Archivo JSON no encontrado: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                templates_config = json.load(f)
            
            # Validar estructura esperada
            required_keys = ['metadata', 'global_statistics', 'optimal_templates_corrected']
            for key in required_keys:
                if key not in templates_config:
                    raise ValueError(f"Clave requerida '{key}' no encontrada en JSON")
            
            # Validar landmarks L1-L15
            landmarks_found = []
            optimal_templates = templates_config['optimal_templates_corrected']
            
            for i in range(1, 16):
                landmark_key = f"L{i}"
                if landmark_key in optimal_templates:
                    landmarks_found.append(landmark_key)
                    
                    # Validar estructura del template
                    template = optimal_templates[landmark_key]
                    required_template_keys = ['anchor_point', 'template_extensions', 'template_dimensions']
                    
                    for template_key in required_template_keys:
                        if template_key not in template:
                            self.logger.warning(f"Clave '{template_key}' no encontrada en template {landmark_key}")
                else:
                    self.logger.warning(f"Template {landmark_key} no encontrado en configuración")
            
            # Almacenar en cache
            self.optimal_templates_config = {
                'config': templates_config,
                'landmarks_available': landmarks_found,
                'loaded_from': str(json_file),
                'loaded_at': str(datetime.now())
            }
            
            # Información de metadatos
            metadata = templates_config['metadata']
            
            self.logger.info(f"✅ Configuración de templates óptimos cargada exitosamente:")
            self.logger.info(f"  - Landmarks procesados: {metadata.get('total_landmarks_processed', 'N/A')}")
            self.logger.info(f"  - Algoritmo: {metadata.get('algorithm', 'N/A')}")
            self.logger.info(f"  - Método de búsqueda: {metadata.get('search_method', 'N/A')}")
            self.logger.info(f"  - Templates encontrados: {len(landmarks_found)} ({landmarks_found[:5]}...)")
            
            if 'processing_summary' in metadata:
                summary = metadata['processing_summary']
                self.logger.info(f"  - Área promedio de template: {summary.get('avg_template_area', 'N/A')}")
                self.logger.info(f"  - Área máxima de template: {summary.get('max_template_area', 'N/A')}")
            
            return self.optimal_templates_config
            
        except Exception as e:
            error_msg = f"Error cargando configuración de templates óptimos: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)


def main():
    """
    Función principal para pruebas del sistema de predicción
    """
    print("=" * 80)
    print("🔮 SISTEMA DE PREDICCIÓN DE LANDMARKS MÉDICOS")
    print("=" * 80)
    print()
    
    try:
        # Crear predictor
        predictor = LandmarkPredictor()
        
        print("\n🧠 === CARGA DE MODELOS PCA ===")
        # Cargar modelo L1 como ejemplo
        print("Cargando modelo PCA para L1...")
        model_info = predictor.load_pca_model("L1")
        
        print("\n📊 Información del modelo L1:")
        print(f"  - Componentes PCA: {model_info['n_components']}")
        print(f"  - Dimensiones: {model_info['image_dimensions']}")
        print(f"  - Varianza explicada (primeros 5): {[f'{v:.1f}%' for v in model_info['explained_variance_ratio'][:5] * 100]}")
        
        # Mostrar estadísticas de entrenamiento si están disponibles
        if 'training_metadata' in model_info:
            metadata = model_info['training_metadata']
            print(f"  - Imágenes de entrenamiento: {metadata['n_training_images']}")
            print(f"  - Categorías: {metadata['dataset_categories']}")
        
        print("\n📋 === CARGA DE DATOS DE PRUEBA ===")
        # Cargar datos de coordenadas de prueba
        test_coords = predictor.load_test_coordinates()
        print(f"\n📊 Resumen de datos de prueba:")
        print(f"  - Total muestras: {test_coords['metadata']['total_samples']}")
        print(f"  - Distribución: {test_coords['metadata']['category_distribution']}")
        
        print("\n📷 === CARGA DE IMÁGENES DEL DATASET ===")
        # Cargar una muestra de imágenes (solo COVID para prueba)
        dataset_images = predictor.load_dataset_images(categories=["COVID"])
        print(f"\n📊 Resumen de imágenes del dataset:")
        print(f"  - Total imágenes: {dataset_images['metadata']['total_images']}")
        print(f"  - Categorías cargadas: {dataset_images['metadata']['categories_loaded']}")
        
        print("\n📦 === CARGA DE CONFIGURACIONES ===")
        # Cargar configuración de bounding boxes
        bbox_config = predictor.load_bounding_boxes_config()
        print(f"\n📊 Resumen de bounding boxes:")
        print(f"  - Landmarks disponibles: {len(bbox_config['landmarks_available'])}")
        
        # Cargar configuración de templates óptimos
        templates_config = predictor.load_optimal_templates_config()
        print(f"\n📊 Resumen de templates óptimos:")
        print(f"  - Templates disponibles: {len(templates_config['landmarks_available'])}")
        
        print("\n" + "=" * 80)
        print("✅ TODAS LAS FUNCIONES DE CARGA PROBADAS EXITOSAMENTE")
        print("=" * 80)
        print("\n🎯 Sistema completo de carga de datos implementado:")
        print("  ✓ Modelos PCA (L1-L15)")
        print("  ✓ Coordenadas de prueba CSV")
        print("  ✓ Imágenes del dataset original")
        print("  ✓ Configuración de bounding boxes")
        print("  ✓ Configuración de templates óptimos")
        print("\n🚀 Listo para implementar algoritmos de predicción de landmarks!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()