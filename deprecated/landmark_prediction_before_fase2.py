#!/usr/bin/env python3
"""
Sistema de Predicción de Landmarks Médicos con PCA
=================================================

Este script implementa el algoritmo principal de predicción de landmarks
utilizando template matching con modelos PCA entrenados.

Algoritmo:
1. Cargar modelo PCA, bounding boxes, templates y datos de prueba
2. Para cada imagen: generar candidatos de template dentro del bounding box
3. Proyectar cada candidato al espacio PCA y calcular similitud
4. Seleccionar la mejor predicción y evaluar contra ground truth

Autor: Sistema de desarrollo con IA especializada
Fecha: 2025-08-21
Basado en: landmark_predictor_loader.py
"""

import logging
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Importar el módulo de carga de datos
sys.path.append(".")
from landmark_predictor_loader import LandmarkPredictor as DataLoader


class LandmarkPredictor:
    """
    Predictor principal de landmarks usando template matching con PCA
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

        # Datos cargados
        self.pca_model_l1 = None
        self.bbox_config = None
        self.template_config = None
        self.test_coordinates = None
        self.dataset_images = None

        # Resultados de predicción
        self.predictions = {}
        self.evaluation_results = {}

        self.logger.info("LandmarkPredictor inicializado")

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

        # Almacenar el modelo PCA en el formato esperado por el sistema existente
        self.pca_model_l1 = {
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

        # Realizar predicciones
        errors = []
        successful_predictions = 0

        for filename in validation_filenames:
            try:
                result = self.predict_landmark_l1(filename, step_size)
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
        self.logger.info("=== Inicio del Sistema de Predicción de Landmarks ===")

    def load_required_data(self, target_landmark=None):
        """
        Cargar todos los datos necesarios para predicción

        Args:
            target_landmark (str): Landmark específico a cargar (e.g., "L1", "L2", etc.)
                                 Si es None, intenta cargar L1 por defecto
        """
        self.logger.info("Cargando datos necesarios para predicción...")

        try:
            # Determinar qué landmark cargar
            if target_landmark is None:
                # Comportamiento por defecto: buscar L1
                landmark_to_load = "L1"
                self.logger.info("Cargando modelo PCA L1...")
            else:
                landmark_to_load = target_landmark
                self.logger.info(f"Cargando modelo PCA {landmark_to_load}...")

            # Cargar modelo PCA para el landmark especificado
            self.pca_model_l1 = self.data_loader.load_pca_model(landmark_to_load)

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

    def get_l1_config(self) -> Tuple[Dict, Dict]:
        """
        Obtener configuración específica para L1

        Returns:
            Tuple[Dict, Dict]: (bbox_l1, template_l1)
        """
        bbox_l1 = self.bbox_config["config"]["landmark_bounding_boxes"]["L1"]["bbox"]
        template_l1 = self.template_config["config"]["optimal_templates_corrected"][
            "L1"
        ]

        return bbox_l1, template_l1

    def generate_template_candidates(
        self, image: np.ndarray, bbox: Dict, template: Dict, step_size: int = 2
    ) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Generar candidatos de template dentro del bounding box

        Args:
            image (np.ndarray): Imagen original 299x299x3
            bbox (Dict): Bounding box de L1
            template (Dict): Template óptimo de L1
            step_size (int): Tamaño del paso para muestreo

        Returns:
            List[Tuple[np.ndarray, Tuple[int, int]]]: Lista de (recorte, posición_anclaje)
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
        Calcular similitud PCA para un recorte

        Args:
            crop (np.ndarray): Recorte normalizado
            pca_model (Dict): Modelo PCA cargado

        Returns:
            float: Score de similitud (menor = mejor)
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
        Calcular similitudes PCA para múltiples recortes en una sola operación vectorizada

        Esta es la implementación clave de la optimización Phase 1: Batch Processing.
        En lugar de procesar ~1,600 candidatos individualmente, los procesa todos
        en operaciones matriciales altamente optimizadas usando BLAS.

        Args:
            crops_batch (np.ndarray): Array 2D donde cada fila es un recorte aplanado
                                    Shape: (n_candidates, n_pixels)
            pca_model (Dict): Modelo PCA cargado

        Returns:
            np.ndarray: Array 1D con scores de similitud para cada candidato (menor = mejor)
                       Shape: (n_candidates,)
        """
        # Obtener datos del modelo
        mean_image = pca_model["mean_image"]
        pca_components = pca_model["pca_components"]  # Shape: (n_components, n_pixels)

        # Centrar todos los recortes de una vez
        crops_centered = crops_batch - mean_image  # Broadcasting automático

        # Proyectar todos los recortes al espacio PCA en una sola operación matricial
        # crops_centered: (n_candidates, n_pixels)
        # pca_components.T: (n_pixels, n_components)
        # Resultado: (n_candidates, n_components)
        crops_projected = np.dot(crops_centered, pca_components.T)

        # Reconstruir todos desde el espacio PCA en una sola operación matricial
        # crops_projected: (n_candidates, n_components)
        # pca_components: (n_components, n_pixels)
        # Resultado: (n_candidates, n_pixels)
        crops_reconstructed = np.dot(crops_projected, pca_components)

        # Calcular errores de reconstrucción para todos los candidatos
        reconstruction_errors = crops_centered - crops_reconstructed

        # Error cuadrático medio por candidato
        scores = np.mean(reconstruction_errors**2, axis=1)

        return scores

    def predict_landmark_l1(self, image_filename: str, step_size: int = 2) -> Dict:
        """
        Predecir la posición del landmark L1 en una imagen

        Args:
            image_filename (str): Nombre del archivo de imagen (sin extensión)
            step_size (int): Tamaño del paso para búsqueda

        Returns:
            Dict: Resultados de la predicción
        """
        start_time = time.time()

        # Buscar imagen en el dataset cargado (añadir .png si es necesario)
        image = None
        category = None
        search_filename = image_filename

        # Si no tiene extensión, añadir .png
        if not search_filename.endswith(".png"):
            search_filename = f"{image_filename}.png"

        for cat, images in self.dataset_images["data"].items():
            # Buscar por nombre exacto
            if search_filename in images:
                image = images[search_filename]
                category = cat
                break
            # Para Viral Pneumonia, buscar también con espacios
            elif cat == "Viral_Pneumonia" and search_filename.startswith("Viral"):
                # Convertir "Viral Pneumonia-XXX.png" a "Viral Pneumonia-XXX.png"
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

        # Obtener configuración L1
        bbox_l1, template_l1 = self.get_l1_config()

        # Generar candidatos
        candidates = self.generate_template_candidates(
            image, bbox_l1, template_l1, step_size
        )

        if not candidates:
            raise RuntimeError(
                f"No se generaron candidatos válidos para {image_filename}"
            )

        # FASE 1 OPTIMIZATION: Batch Processing
        # En lugar de evaluar candidatos individualmente (~1,600 iteraciones),
        # procesamos todos en una sola operación matricial vectorizada

        # Preparar batch de candidatos para procesamiento vectorizado
        crops_batch = np.array(
            [crop.flatten() for crop, _ in candidates]
        )  # Shape: (n_candidates, n_pixels)
        positions_batch = np.array(
            [position for _, position in candidates]
        )  # Shape: (n_candidates, 2)

        # Calcular similitudes para todos los candidatos en una sola operación
        scores = self.calculate_similarity_batch_optimized(
            crops_batch, self.pca_model_l1
        )

        # Encontrar el mejor candidato
        best_idx = np.argmin(scores)
        best_score = scores[best_idx]
        best_position = tuple(positions_batch[best_idx])
        best_crop = candidates[best_idx][0]  # Reconstruir el crop original

        prediction_time = time.time() - start_time

        # Obtener ground truth si está disponible
        gt_coords = None

        # Intentar múltiples variaciones del filename para encontrar ground truth
        filename_variations = [
            image_filename,
            image_filename + ".png"
            if not image_filename.endswith(".png")
            else image_filename,
            image_filename.replace(".png", "")
            if image_filename.endswith(".png")
            else image_filename + ".png",
            search_filename,  # filename que se usó para buscar la imagen
        ]

        for filename_var in filename_variations:
            if filename_var in self.test_coordinates["data"]:
                # L1 es el primer landmark (índice 0)
                gt_coords = self.test_coordinates["data"][filename_var]["landmarks"][0]
                break

        result = {
            "image_filename": image_filename,
            "category": category,
            "predicted_position": best_position,
            "ground_truth": gt_coords,
            "best_score": best_score,
            "candidates_evaluated": len(candidates),
            "prediction_time": prediction_time,
            "step_size": step_size,
        }

        # Calcular error si tenemos ground truth
        if gt_coords is not None:
            pred_x, pred_y = best_position
            gt_x, gt_y = gt_coords
            error = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            result["euclidean_error"] = error

        return result

    def predict_batch(
        self,
        image_filenames: List[str] = None,
        step_size: int = 2,
        use_numba: bool = False,
    ) -> Dict:
        """
        Realizar predicción en lote para múltiples imágenes

        Args:
            image_filenames (List[str]): Lista de archivos a procesar (None = todos en test)
            step_size (int): Tamaño del paso para búsqueda
            use_numba (bool): Si True, usar versión optimizada con Numba

        Returns:
            Dict: Resultados de todas las predicciones
        """
        # Si no se especifican archivos, usar todos los del conjunto de prueba
        if image_filenames is None:
            image_filenames = list(self.test_coordinates["data"].keys())

        method_desc = "batch processing optimizado"
        self.logger.info(
            f"Iniciando predicción en lote para {len(image_filenames)} imágenes (método: {method_desc})"
        )

        results = {}
        successful_predictions = 0
        total_time = 0

        desc = "Prediciendo landmarks (batch processing optimizado)"
        for filename in tqdm(image_filenames, desc=desc):
            try:
                # Usar método estándar
                result = self.predict_landmark_l1(filename, step_size)

                results[filename] = result
                total_time += result["prediction_time"]
                successful_predictions += 1

            except Exception as e:
                self.logger.error(f"Error prediciendo {filename}: {e}")
                results[filename] = {"error": str(e), "image_filename": filename}

        # Almacenar resultados
        self.predictions = results

        batch_summary = {
            "total_images": len(image_filenames),
            "successful_predictions": successful_predictions,
            "failed_predictions": len(image_filenames) - successful_predictions,
            "total_time": total_time,
            "average_time_per_image": total_time / successful_predictions
            if successful_predictions > 0
            else 0,
            "step_size": step_size,
            "method": "batch_optimized",
        }

        self.logger.info("Predicción en lote completada:")
        self.logger.info(
            f"  - Exitosas: {successful_predictions}/{len(image_filenames)}"
        )
        self.logger.info(f"  - Tiempo total: {total_time:.2f}s")
        self.logger.info(
            f"  - Tiempo promedio: {batch_summary['average_time_per_image']:.2f}s/imagen"
        )

        return {"predictions": results, "summary": batch_summary}

    def evaluate_predictions(self) -> Dict:
        """
        Evaluar las predicciones contra ground truth

        Returns:
            Dict: Métricas de evaluación
        """
        if not self.predictions:
            raise ValueError(
                "No hay predicciones para evaluar. Ejecute predict_batch() primero."
            )

        self.logger.info("Evaluando predicciones contra ground truth...")

        # Extraer errores válidos
        errors = []
        errors_by_category = {"COVID": [], "Normal": [], "Viral_Pneumonia": []}
        successful_predictions = []

        for filename, result in self.predictions.items():
            if "euclidean_error" in result:
                error = result["euclidean_error"]
                category = result["category"]

                errors.append(error)
                successful_predictions.append(result)

                # Agrupar por categoría
                if category in errors_by_category:
                    errors_by_category[category].append(error)

        if not errors:
            self.logger.warning("No se encontraron predicciones válidas para evaluar")
            return {}

        # Calcular métricas generales
        errors = np.array(errors)

        metrics = {
            "total_predictions": len(errors),
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "std_error": float(np.std(errors)),
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "percentiles": {
                "25th": float(np.percentile(errors, 25)),
                "75th": float(np.percentile(errors, 75)),
                "90th": float(np.percentile(errors, 90)),
                "95th": float(np.percentile(errors, 95)),
            },
        }

        # Métricas por categoría
        category_metrics = {}
        for category, cat_errors in errors_by_category.items():
            if cat_errors:
                cat_errors = np.array(cat_errors)
                category_metrics[category] = {
                    "count": len(cat_errors),
                    "mean_error": float(np.mean(cat_errors)),
                    "median_error": float(np.median(cat_errors)),
                    "std_error": float(np.std(cat_errors)),
                }

        metrics["by_category"] = category_metrics

        # Umbrales de éxito
        success_thresholds = [5, 10, 15, 20]
        success_rates = {}

        for threshold in success_thresholds:
            success_count = np.sum(errors <= threshold)
            success_rates[f"success_rate_{threshold}px"] = success_count / len(errors)

        metrics["success_rates"] = success_rates

        # Almacenar resultados
        self.evaluation_results = metrics

        # Log de resultados principales
        self.logger.info("📊 Resultados de evaluación:")
        self.logger.info(f"  - Predicciones evaluadas: {len(errors)}")
        self.logger.info(f"  - Error promedio: {metrics['mean_error']:.2f} píxeles")
        self.logger.info(f"  - Error mediano: {metrics['median_error']:.2f} píxeles")
        self.logger.info(
            f"  - Tasa de éxito ≤10px: {success_rates['success_rate_10px']:.1%}"
        )
        self.logger.info(
            f"  - Tasa de éxito ≤20px: {success_rates['success_rate_20px']:.1%}"
        )

        return metrics


def main():
    """
    Función principal para ejecutar el algoritmo de predicción
    """
    print("=" * 80)
    print("🎯 SISTEMA DE PREDICCIÓN DE LANDMARKS CON PCA")
    print("=" * 80)
    print()

    try:
        # Crear predictor
        predictor = LandmarkPredictor()

        # Cargar datos necesarios
        print("📂 Cargando datos necesarios...")
        predictor.load_required_data()

        print("\n🔍 === PREDICCIÓN DE LANDMARK L1 ===")

        # Realizar predicción en lote (configuración para pruebas más detalladas)
        test_filenames = list(predictor.test_coordinates["data"].keys())[
            :144
        ]  # 10 imágenes para prueba comparativa

        print(f"Procesando {len(test_filenames)} imágenes de prueba...")
        batch_results = predictor.predict_batch(
            test_filenames, step_size=1
        )  # step_size pequeño para mayor precisión

        print("\n📊 === EVALUACIÓN DE RESULTADOS ===")

        # Evaluar predicciones
        evaluation = predictor.evaluate_predictions()

        if evaluation:
            print("\n🎯 Resumen de rendimiento:")
            print(
                f"  - Error promedio: {evaluation['mean_error']:.2f}±{evaluation['std_error']:.2f} píxeles"
            )
            print(f"  - Error mediano: {evaluation['median_error']:.2f} píxeles")
            print(
                f"  - Rango de error: {evaluation['min_error']:.2f} - {evaluation['max_error']:.2f} píxeles"
            )

            print("\n📈 Tasas de éxito:")
            for threshold, rate in evaluation["success_rates"].items():
                print(f"  - {threshold}: {rate:.1%}")

            print("\n📂 Rendimiento por categoría médica:")
            for category, metrics in evaluation["by_category"].items():
                print(
                    f"  - {category}: {metrics['mean_error']:.2f}±{metrics['std_error']:.2f} px ({metrics['count']} imágenes)"
                )
        else:
            print("\n⚠️  No se pudieron evaluar las predicciones")
            print("   Posibles causas: archivos no encontrados, errores en predicción")

        print("\n" + "=" * 80)
        print("✅ ALGORITMO DE PREDICCIÓN COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print("\n🚀 Sistema de predicción de landmarks L1 funcionando!")
        print("💡 Siguiente paso: Extender a todos los landmarks L1-L15")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
