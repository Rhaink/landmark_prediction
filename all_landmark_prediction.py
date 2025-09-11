#!/usr/bin/env python3
"""
Sistema de Predicción de Landmarks Médicos Multi-Landmark (L1-L15)
==================================================================

COPIA EXACTA de landmark_prediction.py con modificaciones mínimas para soporte multi-landmark.
Conserva TODAS las optimizaciones: progressive batch, adaptive sampling, early stopping.

Cambios realizados:
- self.pca_model_l1 → self.loaded_models = {}
- get_l1_config() → get_landmark_config(landmark)  
- predict_landmark_l1() → predict_landmark(landmark="L1")
- landmarks[0] → landmarks[int(landmark[1:])-1]
- Agregado predict_landmarks_batch() para múltiples landmarks

Autor: Sistema de desarrollo con IA especializada
Fecha: 2025-09-10
Basado en: landmark_prediction.py (COPIA EXACTA con cambios mínimos)
"""

import logging
import sys
import time
from typing import Dict, List, Tuple
import json
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

# Importar el módulo de carga de datos
sys.path.append(".")
from landmark_predictor_loader import LandmarkPredictor as DataLoader


class AllLandmarkPredictor:
    """
    Predictor multi-landmark usando template matching con PCA (L1-L15)
    COPIA EXACTA de LandmarkPredictor con soporte multi-landmark
    """

    def __init__(self, models_base_path="output_pca_analysis_all_landmarks"):
        """
        Inicializar el predictor de landmarks

        Args:
            models_base_path (str): Ruta base donde están los modelos entrenados
        """
        # Configurar logging
        self._setup_logging()

        # Inicializar cargador de datos
        self.data_loader = DataLoader(models_base_path)

        # Datos cargados - CAMBIO: diccionario para múltiples modelos
        self.loaded_models = {}  # CAMBIO: self.pca_model_l1 → self.loaded_models
        self.bbox_config = None
        self.template_config = None
        self.test_coordinates = None
        self.dataset_images = None

        # Resultados de predicción
        self.predictions = {}
        self.evaluation_results = {}

        self.logger.info("AllLandmarkPredictor inicializado")

    def load_model_from_objects(
        self, pca_model_dict, bbox_config=None, template_config=None, quiet=False
    ):
        """
        Cargar modelo PCA desde objetos en memoria (para hyperparameter tuning)

        Args:
            pca_model_dict (dict): Diccionario con componentes del modelo PCA
            bbox_config (dict): Configuración de bounding boxes (opcional)
            template_config (dict): Configuración de templates (opcional)
            quiet (bool): Si True, reduce la salida de logging
        """
        if not quiet:
            self.logger.info("Cargando modelo PCA desde objetos en memoria...")

        # NOTA: Esta función mantiene compatibilidad pero ya no es relevante para multi-landmark
        # Almacenar en formato legacy para compatibilidad
        self.loaded_models["L1"] = {  # CAMBIO: pca_model_l1 → loaded_models["L1"]
            "pca_components": pca_model_dict["principal_components"],
            "mean_image": pca_model_dict["mean_image"],
            "explained_variance_ratio": pca_model_dict["explained_variance_ratio"],
            "explained_variance": pca_model_dict["explained_variance"],
            "n_components": pca_model_dict["n_components"],
            "image_dimensions": pca_model_dict["image_dimensions"],
            "n_pixels": pca_model_dict["n_pixels"],
        }

        # Cargar configuraciones si se proporcionan
        if bbox_config is not None:
            self.bbox_config = bbox_config
        if template_config is not None:
            self.template_config = template_config

        if not quiet:
            self.logger.info(
                f"✅ Modelo PCA cargado: {pca_model_dict['n_components']} componentes"
            )

    def evaluate_on_validation_set(
        self,
        validation_coordinates_path=None,
        validation_images_path=None,
        step_size=2,
        quiet=False,
    ):
        """
        Evaluar modelo en conjunto de validación y devolver error promedio

        Args:
            validation_coordinates_path (str): Ruta a coordenadas de validación (opcional si ya cargadas)
            validation_images_path (str): Ruta a imágenes de validación (opcional si ya cargadas)
            step_size (int): Tamaño del paso para búsqueda
            quiet (bool): Si True, reduce la salida de logging

        Returns:
            float: Error euclidiano promedio en píxeles
        """
        if not quiet:
            self.logger.info("Evaluando modelo en conjunto de validación...")

        # Usar datos ya cargados si están disponibles
        if validation_coordinates_path is None and self.test_coordinates is None:
            raise ValueError(
                "Debe proporcionar validation_coordinates_path o cargar coordenadas primero"
            )

        if validation_images_path is None and self.dataset_images is None:
            raise ValueError(
                "Debe proporcionar validation_images_path o cargar imágenes primero"
            )

        # Cargar coordenadas de validación si se proporciona nueva ruta
        if validation_coordinates_path is not None:
            self.test_coordinates = self.data_loader.load_test_coordinates(
                validation_coordinates_path
            )

        # Cargar imágenes de validación si se proporciona nueva ruta
        if validation_images_path is not None:
            self.dataset_images = self.data_loader.load_dataset_images(
                validation_images_path
            )

        # Obtener lista de imágenes de validación
        validation_filenames = list(self.test_coordinates["data"].keys())

        if not quiet:
            self.logger.info(
                f"Evaluando en {len(validation_filenames)} imágenes de validación..."
            )

        # Realizar predicciones - CAMBIO: usar predict_landmark en lugar de predict_landmark_l1
        errors = []
        successful_predictions = 0

        for filename in validation_filenames:
            try:
                result = self.predict_landmark(filename, "L1", step_size)  # CAMBIO: generalizado
                if "euclidean_error" in result:
                    errors.append(result["euclidean_error"])
                    successful_predictions += 1
            except Exception as e:
                if not quiet:
                    self.logger.warning(f"Error prediciendo {filename}: {e}")
                continue

        if not errors:
            raise ValueError(
                "No se pudieron realizar predicciones válidas en el conjunto de validación"
            )

        mean_error = float(np.mean(errors))

        if not quiet:
            self.logger.info(
                f"✅ Evaluación completada: {successful_predictions}/{len(validation_filenames)} exitosas"
            )
            self.logger.info(f"📊 Error euclidiano promedio: {mean_error:.3f} píxeles")

        return mean_error

    def _setup_logging(self):
        """
        Configurar sistema de logging
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("=== Inicio del Sistema de Predicción Multi-Landmark ===")

    def load_required_data(self, target_landmarks=None):
        """
        Cargar datos necesarios para predicción multi-landmark - CAMBIO: soporte múltiples landmarks

        Args:
            target_landmarks (List[str]): Landmarks a cargar (ej: ["L1", "L2"])
                                        Si es None, carga todos L1-L15
        """
        self.logger.info("Cargando datos necesarios para predicción...")

        try:
            # Determinar qué landmarks cargar - CAMBIO: soporte multi-landmark
            if target_landmarks is None:
                target_landmarks = [f"L{i}" for i in range(1, 16)]  # L1-L15
            elif isinstance(target_landmarks, str):
                target_landmarks = [target_landmarks]  # Convertir string a lista

            # Cargar modelos PCA para landmarks especificados - CAMBIO: loop múltiple
            self.logger.info(f"Cargando {len(target_landmarks)} modelos PCA...")
            for landmark in target_landmarks:
                try:
                    model = self.data_loader.load_pca_model(landmark)
                    self.loaded_models[landmark] = model
                    self.logger.info(f"✅ {landmark} cargado")
                except Exception as e:
                    self.logger.error(f"❌ Error cargando {landmark}: {e}")

            # Cargar configuraciones
            self.logger.info("Cargando configuraciones...")
            self.bbox_config = self.data_loader.load_bounding_boxes_config()
            self.template_config = self.data_loader.load_optimal_templates_config()

            # Cargar datos de prueba
            self.logger.info("Cargando coordenadas de prueba...")
            self.test_coordinates = self.data_loader.load_test_coordinates()

            # Cargar imágenes del dataset (todas las categorías)
            self.logger.info("Cargando imágenes del dataset...")
            self.dataset_images = self.data_loader.load_dataset_images()

            self.logger.info("✅ Todos los datos cargados exitosamente")

        except Exception as e:
            error_msg = f"Error cargando datos: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def get_landmark_config(self, landmark: str) -> Tuple[Dict, Dict]:
        """
        Obtener configuración para landmark específico - CAMBIO: generalizado de get_l1_config

        Args:
            landmark (str): Landmark a obtener (ej: "L1", "L2", etc.)

        Returns:
            Tuple[Dict, Dict]: (bbox, template)
        """
        bbox = self.bbox_config["config"]["landmark_bounding_boxes"][landmark]["bbox"]
        template = self.template_config["config"]["optimal_templates_corrected"][landmark]

        return bbox, template

    def generate_template_candidates(
        self, image: np.ndarray, bbox: Dict, template: Dict, step_size: int = 2
    ) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Generar candidatos de template dentro del bounding box - SIN CAMBIOS del original
        """
        # Configuración del bounding box
        bbox_x = int(bbox["x"])
        bbox_y = int(bbox["y"])
        bbox_width = int(bbox["width"])
        bbox_height = int(bbox["height"])

        # Configuración del template
        anchor_x = template["anchor_point"]["x"]
        anchor_y = template["anchor_point"]["y"]
        left_ext = template["template_extensions"]["left"]
        right_ext = template["template_extensions"]["right"]
        up_ext = template["template_extensions"]["up"]
        down_ext = template["template_extensions"]["down"]
        template_width = template["template_dimensions"]["width"]
        template_height = template["template_dimensions"]["height"]

        candidates = []

        # Convertir imagen a escala de grises
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()

        # Iterar posiciones de anclaje dentro del bounding box
        for ay in range(bbox_y, bbox_y + bbox_height, step_size):
            for ax in range(bbox_x, bbox_x + bbox_width, step_size):
                # Calcular límites del template desde el anclaje
                template_left = ax - left_ext
                template_right = ax + right_ext + 1  # +1 para incluir el píxel
                template_top = ay - up_ext
                template_bottom = ay + down_ext + 1  # +1 para incluir el píxel

                # Verificar que el template esté dentro de los límites de la imagen
                if (
                    template_left >= 0
                    and template_right <= 299
                    and template_top >= 0
                    and template_bottom <= 299
                ):
                    # Extraer recorte
                    crop = image_gray[
                        template_top:template_bottom, template_left:template_right
                    ]

                    # Verificar dimensiones correctas
                    if crop.shape == (template_height, template_width):
                        # Normalizar a [0,1]
                        crop_normalized = crop.astype(np.float32) / 255.0
                        candidates.append((crop_normalized, (ax, ay)))

        self.logger.debug(f"Generados {len(candidates)} candidatos de template")
        return candidates

    def calculate_pca_similarity(self, crop: np.ndarray, pca_model: Dict) -> float:
        """
        Calcular similitud PCA para un recorte - SIN CAMBIOS del original
        """
        # Aplanar el recorte
        crop_flat = crop.flatten()

        # Centrar usando la media del modelo
        mean_image = pca_model["mean_image"]
        crop_centered = crop_flat - mean_image

        # Obtener componentes principales
        pca_components = pca_model["pca_components"]

        # Proyectar manualmente al espacio PCA
        crop_projected = np.dot(crop_centered, pca_components.T)

        # Reconstruir desde el espacio PCA
        crop_reconstructed = np.dot(crop_projected, pca_components)

        # Calcular error de reconstrucción (menor = más similar)
        reconstruction_error = np.mean((crop_centered - crop_reconstructed) ** 2)

        return reconstruction_error

    def calculate_similarity_batch_optimized(
        self, crops_batch: np.ndarray, pca_model: Dict
    ) -> np.ndarray:
        """
        Calcular similitudes PCA para múltiples recortes - SIN CAMBIOS del original
        """
        # Obtener datos del modelo
        mean_image = pca_model["mean_image"]
        pca_components = pca_model["pca_components"]  # Shape: (n_components, n_pixels)

        # Centrar todos los recortes de una vez
        crops_centered = crops_batch - mean_image  # Broadcasting automático

        # Proyectar todos los recortes al espacio PCA en una sola operación matricial
        crops_projected = np.dot(crops_centered, pca_components.T)

        # Reconstruir todos desde el espacio PCA en una sola operación matricial
        crops_reconstructed = np.dot(crops_projected, pca_components)

        # Calcular errores de reconstrucción para todos los candidatos
        reconstruction_errors = crops_centered - crops_reconstructed

        # Error cuadrático medio por candidato
        scores = np.mean(reconstruction_errors**2, axis=1)

        return scores
    
    def calculate_similarity_progressive_batch(self, crops_batch: np.ndarray, pca_model: Dict, 
                                             component_stages: List[int] = None, 
                                             adaptive_sampling: bool = True,
                                             medical_mode: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Calcular similitudes PCA usando refinamiento progresivo - SIN CAMBIOS del original
        """
        if component_stages is None:
            # Configuración optimizada basada en análisis: 20 (preview) → 110 (production) → 668 (fallback)
            component_stages = [20, 110, 668]
        
        # Obtener datos del modelo completo
        mean_image = pca_model['mean_image']
        full_components = pca_model['pca_components']  # Shape: (668, n_pixels)
        n_candidates, n_pixels = crops_batch.shape
        
        # Centrar todos los recortes una vez
        crops_centered = crops_batch - mean_image
        
        debug_info = {
            'stages_processed': [],
            'candidates_remaining': [],
            'early_stopping_triggered': False,
            'final_stage_used': None,
            'adaptive_sampling_used': False,
            'original_candidates': n_candidates
        }
        
        # ADAPTIVE CANDIDATE SAMPLING: Procesar solo un subconjunto inicial
        if adaptive_sampling and n_candidates > 1000:
            if medical_mode:
                # Medical mode: More conservative sampling - keep more candidates for safety
                sample_step = 2  # Esto reduce ~14,100 candidatos a ~7,050 (more conservative)
                sampled_indices = np.arange(0, n_candidates, sample_step)
                debug_info['medical_sampling'] = 'Conservative: reduced step size for better coverage'
            else:
                # Standard mode: Aggressive sampling for speed
                sample_step = 4  # Esto reduce ~14,100 candidatos a ~3,525
                sampled_indices = np.arange(0, n_candidates, sample_step)
            
            debug_info['adaptive_sampling_used'] = True
            debug_info['sample_step'] = sample_step
            debug_info['sampled_candidates'] = len(sampled_indices)
        else:
            # Para datasets pequeños o sin adaptive sampling, usar todos
            sampled_indices = np.arange(n_candidates)
        
        # Variables para tracking de candidatos
        active_candidates = sampled_indices.copy()  # Indices de candidatos activos
        final_scores = np.full(n_candidates, np.inf)  # Scores finales (inf = no procesado)
        
        for stage_idx, n_components in enumerate(component_stages):
            if len(active_candidates) == 0:
                break  # No quedan candidatos por procesar
                
            # Limitar componentes al máximo disponible
            n_components = min(n_components, full_components.shape[0])
            
            # Extraer subconjunto de componentes PCA para esta etapa
            stage_components = full_components[:n_components, :]  # Shape: (n_components, n_pixels)
            
            # Procesar solo los candidatos activos
            active_crops_centered = crops_centered[active_candidates]
            
            # Proyección y reconstrucción con componentes limitados
            crops_projected = np.dot(active_crops_centered, stage_components.T)
            crops_reconstructed = np.dot(crops_projected, stage_components)
            
            # Calcular errores de reconstrucción
            reconstruction_errors = active_crops_centered - crops_reconstructed
            stage_scores = np.mean(reconstruction_errors ** 2, axis=1)
            
            # Actualizar scores finales para candidatos procesados
            final_scores[active_candidates] = stage_scores
            
            # Debug info
            debug_info['stages_processed'].append({
                'stage': stage_idx,
                'components': n_components,
                'candidates_processed': len(active_candidates),
                'mean_score': np.mean(stage_scores),
                'min_score': np.min(stage_scores)
            })
            debug_info['candidates_remaining'].append(len(active_candidates))
            debug_info['final_stage_used'] = n_components
            
            # EARLY STOPPING LOGIC - Medical mode uses conservative thresholds
            if stage_idx < len(component_stages) - 1:
                if medical_mode:
                    # MEDICAL MODE: Intelligent conservative thresholds for medical-grade precision
                    if n_components == 20:
                        n_keep = max(3, len(active_candidates) // 10)  # 10% with minimum of 3
                        best_indices = np.argsort(stage_scores)[:n_keep]
                        active_candidates = active_candidates[best_indices]
                        
                        best_score = np.min(stage_scores)
                        if best_score < 0.005:  # Conservative: 0.005
                            debug_info['early_stopping_triggered'] = True
                            debug_info['early_stopping_reason'] = f'Medical mode: good score {best_score:.6f} at stage {n_components}'
                            break
                    
                    elif n_components == 110:
                        best_indices = np.argsort(stage_scores)[:2]  # Top 2
                        active_candidates = active_candidates[best_indices]
                        
                        best_score = stage_scores[best_indices[0]]
                        if best_score < 0.01:  # Balanced: 0.01
                            debug_info['early_stopping_triggered'] = True
                            debug_info['early_stopping_reason'] = f'Medical mode: balanced score {best_score:.6f} at stage {n_components}'
                            break
                    
                    best_score = np.min(stage_scores)
                    if best_score < 0.005:  # Balanced medical threshold
                        debug_info['early_stopping_triggered'] = True
                        debug_info['early_stopping_reason'] = f'Medical mode: conservative precision {best_score:.6f}'
                        break
                else:
                    # STANDARD MODE: Aggressive thresholds for speed
                    if n_components == 20:
                        n_keep = max(1, len(active_candidates) // 20)
                        best_indices = np.argsort(stage_scores)[:n_keep]
                        active_candidates = active_candidates[best_indices]
                        
                        best_score = np.min(stage_scores)
                        if best_score < 0.01:
                            debug_info['early_stopping_triggered'] = True
                            break
                    
                    elif n_components == 110:
                        best_indices = np.argsort(stage_scores)[:1]  # Solo el mejor
                        active_candidates = active_candidates[best_indices]
                        
                        best_score = stage_scores[best_indices[0]]
                        if best_score < 0.05:  # Threshold más relajado
                            debug_info['early_stopping_triggered'] = True
                            break
                    
                    best_score = np.min(stage_scores)
                    if best_score < 0.001:
                        debug_info['early_stopping_triggered'] = True
                        break
        
        # Si usamos adaptive sampling, interpolar scores para candidatos no evaluados
        if debug_info['adaptive_sampling_used']:
            final_scores[final_scores == np.inf] = np.max(final_scores[final_scores != np.inf]) * 2.0
        
        return final_scores, debug_info

    def predict_landmark(self, image_filename: str, landmark: str = "L1", step_size: int = 2, medical_mode: bool = False) -> Dict:
        """
        Predecir posición de landmark específico - CAMBIO: generalizado de predict_landmark_l1

        Args:
            image_filename (str): Nombre del archivo de imagen (sin extensión)
            landmark (str): Landmark a predecir (ej: "L1", "L2", etc.)
            step_size (int): Tamaño del paso para búsqueda
            medical_mode (bool): Modo médico seguro

        Returns:
            Dict: Resultados de la predicción
        """
        start_time = time.time()

        # Verificar que el modelo esté cargado - CAMBIO: verificación dinámica
        if landmark not in self.loaded_models:
            raise ValueError(f"Modelo {landmark} no está cargado. Landmarks disponibles: {list(self.loaded_models.keys())}")

        # Buscar imagen en el dataset cargado (código idéntico al original)
        image = None
        category = None
        search_filename = image_filename

        if not search_filename.endswith(".png"):
            search_filename = f"{image_filename}.png"

        for cat, images in self.dataset_images["data"].items():
            if search_filename in images:
                image = images[search_filename]
                category = cat
                break
            elif cat == "Viral_Pneumonia" and search_filename.startswith("Viral"):
                viral_name = search_filename.replace("Viral", "Viral Pneumonia", 1)
                if viral_name in images:
                    image = images[viral_name]
                    category = cat
                    search_filename = viral_name
                    break

        if image is None:
            raise ValueError(
                f"Imagen {image_filename} (buscada como {search_filename}) no encontrada en el dataset"
            )

        # Obtener configuración del landmark específico - CAMBIO: parametrizado
        bbox, template = self.get_landmark_config(landmark)

        # Generar candidatos
        candidates = self.generate_template_candidates(
            image, bbox, template, step_size
        )

        if not candidates:
            raise RuntimeError(
                f"No se generaron candidatos válidos para {landmark} en {image_filename}"
            )

        # Preparar batch de candidatos para procesamiento vectorizado
        crops_batch = np.array(
            [crop.flatten() for crop, _ in candidates]
        )  # Shape: (n_candidates, n_pixels)
        positions_batch = np.array(
            [position for _, position in candidates]
        )  # Shape: (n_candidates, 2)

        # Usar modelo específico del landmark - CAMBIO: modelo dinámico
        pca_model = self.loaded_models[landmark]

        # MANTENER TODAS LAS OPTIMIZACIONES DEL ORIGINAL
        if medical_mode:
            if step_size == 1:
                scores, debug_info = self.calculate_similarity_progressive_batch(
                    crops_batch, pca_model, adaptive_sampling=True, medical_mode=True
                )
                method_used = 'progressive_pca_medical_safe'
                debug_info['medical_mode'] = True
                debug_info['medical_safety_note'] = 'Conservative thresholds and intelligent sampling'
            else:
                scores = self.calculate_similarity_batch_optimized(crops_batch, pca_model)
                debug_info = {'method': 'batch_optimized_medical', 'reason': 'step_size > 1', 'medical_mode': True}
                method_used = 'batch_optimized_medical'
        elif step_size == 1:
            scores, debug_info = self.calculate_similarity_progressive_batch(
                crops_batch, pca_model, adaptive_sampling=True
            )
            method_used = 'progressive_pca_hybrid_adaptive'
        else:
            scores = self.calculate_similarity_batch_optimized(crops_batch, pca_model)
            debug_info = {'method': 'batch_optimized', 'reason': 'step_size > 1'}
            method_used = 'batch_optimized'

        # Encontrar el mejor candidato
        best_idx = np.argmin(scores)
        best_score = scores[best_idx]
        best_position = tuple(positions_batch[best_idx])
        best_crop = candidates[best_idx][0]

        prediction_time = time.time() - start_time

        # Obtener ground truth - CAMBIO: índice dinámico basado en landmark
        gt_coords = None
        filename_variations = [
            image_filename,
            image_filename + ".png"
            if not image_filename.endswith(".png")
            else image_filename,
            image_filename.replace(".png", "")
            if image_filename.endswith(".png")
            else image_filename + ".png",
            search_filename,
        ]

        for filename_var in filename_variations:
            if filename_var in self.test_coordinates["data"]:
                # CAMBIO CRÍTICO: índice dinámico basado en landmark
                landmark_index = int(landmark[1:]) - 1  # L1→0, L2→1, L3→2, etc.
                gt_coords = self.test_coordinates["data"][filename_var]["landmarks"][landmark_index]
                break

        result = {
            "image_filename": image_filename,
            "landmark": landmark,  # CAMBIO: agregar landmark ID
            "category": category,
            "predicted_position": best_position,
            "ground_truth": gt_coords,
            "best_score": best_score,
            "candidates_evaluated": len(candidates),
            "prediction_time": prediction_time,
            "step_size": step_size,
            "method": method_used,
            "optimization_info": debug_info,
            "medical_mode": medical_mode
        }

        # Calcular error si tenemos ground truth
        error = None
        if gt_coords is not None:
            pred_x, pred_y = best_position
            gt_x, gt_y = gt_coords
            error = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            result["euclidean_error"] = error

        # MANTENER VALIDACIÓN MÉDICA DEL ORIGINAL
        if medical_mode and error is not None:
            medical_quality_check = {
                'error_within_tolerance': error <= 10.0,
                'error_critical': error > 25.0,
                'score_reasonable': best_score < 1.0,
                'passed_quality_check': False
            }
            
            medical_quality_check['passed_quality_check'] = (
                medical_quality_check['error_within_tolerance'] and 
                medical_quality_check['score_reasonable'] and
                not medical_quality_check['error_critical']
            )
            
            result['medical_quality_check'] = medical_quality_check
            
            if not medical_quality_check['passed_quality_check']:
                result['medical_warning'] = f"Quality check failed - Error: {error:.2f}px, Score: {best_score:.6f}"
                if debug_info.get('adaptive_sampling_used', False) or debug_info.get('early_stopping_triggered', False):
                    result['fallback_recommendation'] = "Consider using step_size=1 with medical_mode=True or validate with reference method"
        
        elif not medical_mode and error is not None:
            result['quality_metrics'] = {
                'error_acceptable': error <= 10.0,
                'error_critical': error > 25.0
            }

        return result

    def predict_landmarks_batch(self, image_filenames: List[str] = None, 
                              landmarks: List[str] = None, 
                              step_size: int = 2,
                              medical_mode: bool = False) -> Dict:
        """
        Predicción batch multi-landmark - NUEVA FUNCIONALIDAD

        Args:
            image_filenames (List[str]): Imágenes a procesar (None = todas en test)
            landmarks (List[str]): Landmarks a predecir (None = todos cargados)
            step_size (int): Tamaño del paso para búsqueda
            medical_mode (bool): Modo médico seguro

        Returns:
            Dict: Resultados organizados por imagen y landmark
        """
        # Configuración por defecto
        if image_filenames is None:
            image_filenames = list(self.test_coordinates["data"].keys())
        
        if landmarks is None:
            landmarks = list(self.loaded_models.keys())

        method_desc = f"multi-landmark processing (step_size={step_size})"
        self.logger.info(
            f"Iniciando predicción multi-landmark para {len(image_filenames)} imágenes × {len(landmarks)} landmarks (método: {method_desc})"
        )

        results = {}
        total_predictions = len(image_filenames) * len(landmarks)
        successful_predictions = 0
        total_time = 0

        # Progress bar para todas las predicciones
        desc = f"Prediciendo landmarks (multi-landmark)"
        with tqdm(total=total_predictions, desc=desc) as pbar:
            for filename in image_filenames:
                results[filename] = {}
                
                for landmark in landmarks:
                    try:
                        result = self.predict_landmark(filename, landmark, step_size, medical_mode)
                        results[filename][landmark] = result
                        total_time += result["prediction_time"]
                        successful_predictions += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error prediciendo {filename} {landmark}: {e}")
                        results[filename][landmark] = {
                            "error": str(e), 
                            "image_filename": filename,
                            "landmark": landmark
                        }
                    
                    pbar.update(1)

        # Almacenar resultados
        self.predictions = results

        batch_summary = {
            "total_predictions": total_predictions,
            "successful_predictions": successful_predictions,
            "failed_predictions": total_predictions - successful_predictions,
            "total_time": total_time,
            "average_time_per_prediction": total_time / successful_predictions if successful_predictions > 0 else 0,
            "landmarks_processed": landmarks,
            "images_processed": len(image_filenames),
            "step_size": step_size,
            "method": "multi_landmark_batch_optimized"
        }

        self.logger.info("Predicción multi-landmark completada:")
        self.logger.info(
            f"  - Exitosas: {successful_predictions}/{total_predictions}"
        )
        self.logger.info(f"  - Tiempo total: {total_time:.2f}s")
        self.logger.info(
            f"  - Tiempo promedio: {batch_summary['average_time_per_prediction']:.2f}s/predicción"
        )

        return {"predictions": results, "summary": batch_summary}

    def evaluate_predictions(self) -> Dict:
        """
        Evaluar predicciones multi-landmark - MODIFICADO para soporte multi-landmark
        """
        if not self.predictions:
            raise ValueError(
                "No hay predicciones para evaluar. Ejecute predict_landmarks_batch() primero."
            )

        self.logger.info("Evaluando predicciones multi-landmark contra ground truth...")

        # Organizar errores por landmark
        errors_by_landmark = {}
        errors_by_category = {"COVID": {}, "Normal": {}, "Viral_Pneumonia": {}}
        total_errors = []

        for filename, landmarks_results in self.predictions.items():
            for landmark, result in landmarks_results.items():
                if "euclidean_error" in result:
                    error = result["euclidean_error"]
                    category = result["category"]

                    # Por landmark
                    if landmark not in errors_by_landmark:
                        errors_by_landmark[landmark] = []
                    errors_by_landmark[landmark].append(error)

                    # Por categoría
                    if landmark not in errors_by_category[category]:
                        errors_by_category[category][landmark] = []
                    errors_by_category[category][landmark].append(error)

                    total_errors.append(error)

        if not total_errors:
            self.logger.warning("No se encontraron predicciones válidas para evaluar")
            return {}

        # Métricas generales
        total_errors = np.array(total_errors)
        general_metrics = {
            "total_predictions": len(total_errors),
            "mean_error": float(np.mean(total_errors)),
            "median_error": float(np.median(total_errors)),
            "std_error": float(np.std(total_errors)),
            "min_error": float(np.min(total_errors)),
            "max_error": float(np.max(total_errors)),
            "percentiles": {
                "25th": float(np.percentile(total_errors, 25)),
                "75th": float(np.percentile(total_errors, 75)),
                "90th": float(np.percentile(total_errors, 90)),
                "95th": float(np.percentile(total_errors, 95)),
            },
        }

        # Métricas por landmark
        landmark_metrics = {}
        for landmark, errors in errors_by_landmark.items():
            if errors:
                errors_arr = np.array(errors)
                landmark_metrics[landmark] = {
                    "count": len(errors),
                    "mean_error": float(np.mean(errors_arr)),
                    "median_error": float(np.median(errors_arr)),
                    "std_error": float(np.std(errors_arr)),
                    "success_rate_10px": float(np.sum(errors_arr <= 10) / len(errors_arr))
                }

        # Métricas por categoría
        category_metrics = {}
        for category, landmark_errors in errors_by_category.items():
            if landmark_errors:
                category_metrics[category] = {}
                for landmark, errors in landmark_errors.items():
                    if errors:
                        errors_arr = np.array(errors)
                        category_metrics[category][landmark] = {
                            "count": len(errors),
                            "mean_error": float(np.mean(errors_arr)),
                            "std_error": float(np.std(errors_arr)),
                        }

        # Tasa de éxito general
        success_thresholds = [5, 10, 15, 20]
        success_rates = {}
        for threshold in success_thresholds:
            success_count = np.sum(total_errors <= threshold)
            success_rates[f"success_rate_{threshold}px"] = float(success_count / len(total_errors))

        evaluation_results = {
            "general_metrics": general_metrics,
            "landmark_metrics": landmark_metrics,
            "category_metrics": category_metrics,
            "success_rates": success_rates,
            "landmarks_evaluated": list(errors_by_landmark.keys())
        }

        self.evaluation_results = evaluation_results

        # Log de resultados principales
        self.logger.info("📊 Resultados de evaluación multi-landmark:")
        self.logger.info(f"  - Predicciones evaluadas: {len(total_errors)}")
        self.logger.info(f"  - Error promedio: {general_metrics['mean_error']:.2f} píxeles")
        self.logger.info(f"  - Error mediano: {general_metrics['median_error']:.2f} píxeles")
        self.logger.info(
            f"  - Tasa de éxito ≤10px: {success_rates['success_rate_10px']:.1%}"
        )
        self.logger.info(
            f"  - Tasa de éxito ≤20px: {success_rates['success_rate_20px']:.1%}"
        )

        return evaluation_results

    def save_results(self, output_dir: str = "results", formats: List[str] = None) -> Dict[str, str]:
        """
        Guardar resultados en múltiples formatos para diferentes usos

        Args:
            output_dir (str): Directorio donde guardar los resultados
            formats (List[str]): Formatos a guardar ['json', 'pickle', 'csv', 'npz', 'hdf5']

        Returns:
            Dict[str, str]: Rutas de archivos guardados
        """
        if not self.predictions:
            raise ValueError("No hay resultados para guardar. Ejecute predict_landmarks_batch() primero.")

        if formats is None:
            formats = ['json', 'pickle', 'csv', 'npz']  # Formatos por defecto

        # Crear directorio de resultados
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # Preparar datos para diferentes formatos
        results_data = {
            'predictions': self.predictions,
            'evaluation': self.evaluation_results,
            'metadata': {
                'timestamp': timestamp,
                'total_predictions': len([p for img in self.predictions.values() for p in img.values()]),
                'landmarks_processed': list(self.loaded_models.keys()),
                'images_processed': list(self.predictions.keys()),
                'format_version': '1.0'
            }
        }

        # 1. JSON - Para análisis e intercambio
        if 'json' in formats:
            json_file = output_path / f"landmark_results_{timestamp}.json"
            
            # Convertir numpy arrays a listas para JSON
            json_data = self._prepare_for_json(results_data)
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            saved_files['json'] = str(json_file)
            self.logger.info(f"✅ Resultados guardados en JSON: {json_file}")

        # 2. Pickle - Para Python nativo (más rápido)
        if 'pickle' in formats:
            pickle_file = output_path / f"landmark_results_{timestamp}.pkl"
            
            with open(pickle_file, 'wb') as f:
                pickle.dump(results_data, f)
            
            saved_files['pickle'] = str(pickle_file)
            self.logger.info(f"✅ Resultados guardados en Pickle: {pickle_file}")

        # 3. CSV - Para análisis estadístico
        if 'csv' in formats:
            csv_file = output_path / f"landmark_results_{timestamp}.csv"
            
            # Crear DataFrame plano
            rows = []
            for image_name, landmarks in self.predictions.items():
                for landmark_id, result in landmarks.items():
                    if 'euclidean_error' in result:
                        row = {
                            'image_name': image_name,
                            'landmark': landmark_id,
                            'predicted_x': result['predicted_position'][0],
                            'predicted_y': result['predicted_position'][1],
                            'ground_truth_x': result['ground_truth'][0] if result['ground_truth'] else None,
                            'ground_truth_y': result['ground_truth'][1] if result['ground_truth'] else None,
                            'euclidean_error': result['euclidean_error'],
                            'category': result['category'],
                            'prediction_time': result['prediction_time'],
                            'best_score': result['best_score'],
                            'method': result['method']
                        }
                        rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(csv_file, index=False)
            
            saved_files['csv'] = str(csv_file)
            self.logger.info(f"✅ Resultados guardados en CSV: {csv_file}")

        # 4. NPZ - Para arrays numpy (machine learning)
        if 'npz' in formats:
            npz_file = output_path / f"landmark_results_{timestamp}.npz"
            
            # Extraer coordenadas como arrays numpy
            predictions_array, ground_truth_array, errors_array = self._extract_coordinate_arrays()
            
            np.savez_compressed(
                npz_file,
                predictions=predictions_array,
                ground_truth=ground_truth_array,
                errors=errors_array,
                image_names=np.array(list(self.predictions.keys())),
                landmarks=np.array(list(self.loaded_models.keys())),
                metadata=json.dumps(results_data['metadata'])
            )
            
            saved_files['npz'] = str(npz_file)
            self.logger.info(f"✅ Resultados guardados en NPZ: {npz_file}")

        # 5. HDF5 - Para datasets grandes (opcional)
        if 'hdf5' in formats:
            try:
                import h5py
                hdf5_file = output_path / f"landmark_results_{timestamp}.h5"
                
                with h5py.File(hdf5_file, 'w') as f:
                    # Grupos principales
                    pred_group = f.create_group('predictions')
                    eval_group = f.create_group('evaluation')
                    meta_group = f.create_group('metadata')
                    
                    # Guardar coordenadas
                    predictions_array, ground_truth_array, errors_array = self._extract_coordinate_arrays()
                    pred_group.create_dataset('coordinates', data=predictions_array)
                    pred_group.create_dataset('ground_truth', data=ground_truth_array)
                    pred_group.create_dataset('errors', data=errors_array)
                    
                    # Guardar metadatos
                    meta_group.attrs['timestamp'] = timestamp
                    meta_group.attrs['total_predictions'] = len(predictions_array)
                
                saved_files['hdf5'] = str(hdf5_file)
                self.logger.info(f"✅ Resultados guardados en HDF5: {hdf5_file}")
                
            except ImportError:
                self.logger.warning("HDF5 no disponible. Instale h5py: pip install h5py")

        return saved_files

    def _prepare_for_json(self, data: Dict) -> Dict:
        """Convertir numpy arrays y otros objetos no serializables para JSON"""
        import copy
        
        json_data = copy.deepcopy(data)
        
        # Convertir numpy arrays en las predicciones
        for image_name, landmarks in json_data['predictions'].items():
            for landmark_id, result in landmarks.items():
                if 'predicted_position' in result:
                    result['predicted_position'] = list(result['predicted_position'])
                if 'ground_truth' in result and result['ground_truth'] is not None:
                    result['ground_truth'] = list(result['ground_truth'])
                
                # Limpiar campos numpy/complejos para JSON
                for key in list(result.keys()):
                    value = result[key]
                    if isinstance(value, np.ndarray):
                        result[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        result[key] = float(value)
                    elif isinstance(value, dict) and 'optimization_info' in key:
                        # Simplificar info de optimización para JSON
                        result[key] = {'method': value.get('method', 'unknown')}
        
        return json_data

    def _extract_coordinate_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extraer coordenadas como arrays numpy organizados"""
        
        # Organizar datos
        images = list(self.predictions.keys())
        landmarks = list(self.loaded_models.keys())
        n_images, n_landmarks = len(images), len(landmarks)
        
        # Arrays: (n_images, n_landmarks, 2) para coordenadas x,y
        predictions_array = np.full((n_images, n_landmarks, 2), np.nan)
        ground_truth_array = np.full((n_images, n_landmarks, 2), np.nan)
        errors_array = np.full((n_images, n_landmarks), np.nan)
        
        for i, image_name in enumerate(images):
            for j, landmark in enumerate(landmarks):
                if landmark in self.predictions[image_name]:
                    result = self.predictions[image_name][landmark]
                    
                    if 'predicted_position' in result:
                        predictions_array[i, j] = result['predicted_position']
                    
                    if 'ground_truth' in result and result['ground_truth'] is not None:
                        ground_truth_array[i, j] = result['ground_truth']
                    
                    if 'euclidean_error' in result:
                        errors_array[i, j] = result['euclidean_error']
        
        return predictions_array, ground_truth_array, errors_array

    @classmethod
    def load_results(cls, file_path: str) -> 'AllLandmarkPredictor':
        """
        Cargar resultados previamente guardados

        Args:
            file_path (str): Ruta al archivo de resultados

        Returns:
            AllLandmarkPredictor: Instancia con resultados cargados
        """
        predictor = cls()
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                predictor.predictions = data['predictions']
                predictor.evaluation_results = data.get('evaluation', {})
        
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                predictor.predictions = data['predictions']
                predictor.evaluation_results = data.get('evaluation', {})
        
        elif file_path.suffix == '.npz':
            data = np.load(file_path, allow_pickle=True)
            # Reconstruir estructura desde arrays
            predictions_array = data['predictions']
            ground_truth_array = data['ground_truth'] 
            errors_array = data['errors']
            image_names = data['image_names']
            landmarks = data['landmarks']
            
            # Reconstruir diccionario de predicciones
            predictor.predictions = {}
            for i, image_name in enumerate(image_names):
                predictor.predictions[image_name] = {}
                for j, landmark in enumerate(landmarks):
                    if not np.isnan(predictions_array[i, j, 0]):
                        predictor.predictions[image_name][landmark] = {
                            'predicted_position': tuple(predictions_array[i, j]),
                            'ground_truth': tuple(ground_truth_array[i, j]) if not np.isnan(ground_truth_array[i, j, 0]) else None,
                            'euclidean_error': errors_array[i, j] if not np.isnan(errors_array[i, j]) else None
                        }
        
        else:
            raise ValueError(f"Formato no soportado: {file_path.suffix}")
        
        return predictor

    # MANTENER COMPATIBILIDAD: Alias para compatibilidad con código existente
    def predict_landmark_l1(self, image_filename: str, step_size: int = 2, medical_mode: bool = False) -> Dict:
        """Alias para compatibilidad con código existente"""
        return self.predict_landmark(image_filename, "L1", step_size, medical_mode)

    def predict_batch(self, image_filenames: List[str] = None, step_size: int = 2, use_numba: bool = False) -> Dict:
        """Alias para compatibilidad - usa solo L1"""
        return self.predict_landmarks_batch(image_filenames, ["L1"], step_size)


def main():
    """
    Función principal para ejecutar predicción multi-landmark
    """
    print("=" * 80)
    print("🎯 SISTEMA DE PREDICCIÓN MULTI-LANDMARK (L1-L15)")
    print("=" * 80)
    print()

    try:
        # Crear predictor
        predictor = AllLandmarkPredictor()

        # Cargar datos (por defecto: todos los landmarks L1-L15)
        print("📂 Cargando datos necesarios...")
        predictor.load_required_data()

        print(f"\n🔍 === PREDICCIÓN MULTI-LANDMARK ===")

        # Configuración completa - TODAS las imágenes y landmarks
        test_filenames = list(predictor.test_coordinates["data"].keys())  # Todas las 144 imágenes
        test_landmarks = list(predictor.loaded_models.keys())  # Todos los landmarks L1-L15

        print(f"Procesando {len(test_filenames)} imágenes × {len(test_landmarks)} landmarks")
        print(f"Total predicciones: {len(test_filenames) * len(test_landmarks):,}")

        # Realizar predicción multi-landmark con máxima precisión
        batch_results = predictor.predict_landmarks_batch(
            image_filenames=test_filenames,
            landmarks=test_landmarks,
            step_size=1  # Máxima precisión
        )

        print("\n📊 === EVALUACIÓN DE RESULTADOS ===")

        # Evaluar predicciones
        evaluation = predictor.evaluate_predictions()

        if evaluation:
            print("\n🎯 Resumen de rendimiento:")
            print(
                f"  - Error promedio: {evaluation['general_metrics']['mean_error']:.2f}±{evaluation['general_metrics']['std_error']:.2f} píxeles"
            )
            print(f"  - Error mediano: {evaluation['general_metrics']['median_error']:.2f} píxeles")
            print(
                f"  - Rango de error: {evaluation['general_metrics']['min_error']:.2f} - {evaluation['general_metrics']['max_error']:.2f} píxeles"
            )

            print("\n📈 Tasas de éxito:")
            for threshold, rate in evaluation["success_rates"].items():
                print(f"  - {threshold}: {rate:.1%}")

            print(f"\n📍 Rendimiento por landmark (top 5):")
            sorted_landmarks = sorted(evaluation['landmark_metrics'].items(), key=lambda x: x[1]['mean_error'])
            for landmark, metrics in sorted_landmarks[:5]:
                print(f"  - {landmark}: {metrics['mean_error']:.2f}px (≤10px: {metrics['success_rate_10px']:.1%})")

        else:
            print("\n⚠️  No se pudieron evaluar las predicciones")

        print("\n📁 === GUARDANDO RESULTADOS ===")
        
        # Guardar resultados en múltiples formatos
        saved_files = predictor.save_results(
            output_dir="results",
            formats=['json', 'pickle', 'csv', 'npz']
        )
        
        print("✅ Resultados guardados:")
        for format_type, file_path in saved_files.items():
            print(f"   - {format_type.upper()}: {file_path}")

        print("\n" + "=" * 80)
        print("✅ PREDICCIÓN MULTI-LANDMARK COMPLETADA")
        print("=" * 80)
        print("💡 Sistema completo ejecutado:")
        print(f"   - {len(test_landmarks)} landmarks procesados: {test_landmarks}")
        print(f"   - {len(test_filenames)} imágenes de test analizadas")
        print(f"   - Precisión médica: step_size=1 para máxima exactitud")
        print(f"   - Resultados persistidos en {len(saved_files)} formatos")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()