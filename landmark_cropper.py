#!/usr/bin/env python3
"""
Landmark Cropper: Extractor de recortes de landmarks usando templates óptimos

Este script extrae recortes de landmarks de imágenes médicas usando:
- Coordenadas de coordenadas_maestro.csv
- Bounding boxes de landmark_bounding_boxes_corrected.json  
- Templates óptimos de optimal_templates_fixed.json

Inicialmente procesa landmarks L1 y L2 con validación exhaustiva.
"""

import os
import json
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm


class LandmarkCropper:
    def __init__(self, csv_path: str, bbox_path: str, templates_path: str, 
                 dataset_path: str, output_path: str, target_landmarks: List[str] = None):
        """
        Inicializa el extractor de recortes de landmarks.
        
        Args:
            csv_path: Ruta al archivo CSV con coordenadas
            bbox_path: Ruta al archivo JSON con bounding boxes
            templates_path: Ruta al archivo JSON con templates óptimos
            dataset_path: Directorio con imágenes originales
            output_path: Directorio de salida para recortes
            target_landmarks: Lista de landmarks a procesar (default: ['L1', 'L2'])
        """
        self.csv_path = csv_path
        self.bbox_path = bbox_path
        self.templates_path = templates_path
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.target_landmarks = target_landmarks or [f'L{i}' for i in range(1, 16)]  # L1-L15
        
        # Datos cargados
        self.coordinates_data = None
        self.bbox_data = None
        self.templates_data = None
        
        # Estadísticas de procesamiento
        self.extraction_stats = {
            'total_images': 0,
            'successful_extractions': {},
            'failed_extractions': {},
            'errors': []
        }
        
        # Crear directorios de salida primero
        self._setup_output_directories()
        
        # Configurar logging
        self._setup_logging()
        
        # Log información de directorios
        for landmark in self.target_landmarks:
            landmark_dir = self.output_path / landmark
            self.logger.info(f"Directorio creado para {landmark}: {landmark_dir}")
        
        # Cargar todos los datos
        self._load_all_data()

    def _setup_logging(self):
        """Configura el sistema de logging."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.output_path / 'extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_all_data(self):
        """Carga todos los archivos de datos necesarios."""
        self.logger.info("Cargando datos de coordenadas, bounding boxes y templates...")
        
        # Cargar coordenadas desde CSV
        self.coordinates_data = self._load_coordinates()
        self.logger.info(f"Coordenadas cargadas: {len(self.coordinates_data)} imágenes")
        
        # Cargar bounding boxes
        with open(self.bbox_path, 'r') as f:
            self.bbox_data = json.load(f)['landmark_bounding_boxes']
        self.logger.info(f"Bounding boxes cargados para {len(self.bbox_data)} landmarks")
        
        # Cargar templates óptimos
        with open(self.templates_path, 'r') as f:
            self.templates_data = json.load(f)['optimal_templates_corrected']
        self.logger.info(f"Templates óptimos cargados para {len(self.templates_data)} landmarks")

    def _load_coordinates(self) -> List[Dict]:
        """
        Carga las coordenadas desde el archivo CSV.
        
        Returns:
            Lista de diccionarios con datos de coordenadas por imagen
        """
        coordinates = []
        
        with open(self.csv_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 32:  # índice + 30 coordenadas + filename
                    continue
                
                # Extraer coordenadas (ignorar índice y filename)
                coords_data = [float(x) for x in parts[1:-1]]
                filename = parts[-1]
                
                # Convertir a pares de coordenadas (x,y)
                points = []
                for i in range(0, len(coords_data), 2):
                    points.append((int(coords_data[i]), int(coords_data[i+1])))
                
                coordinates.append({
                    'filename': filename,
                    'points': points  # Lista de 15 puntos (x,y)
                })
        
        return coordinates

    def _setup_output_directories(self):
        """Crea la estructura de directorios de salida."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        for landmark in self.target_landmarks:
            landmark_dir = self.output_path / landmark
            landmark_dir.mkdir(exist_ok=True)

    def apply_clahe(self, gray_image: np.ndarray, clip_limit: float = 2.0, 
                    tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para 
        normalización de contraste en imágenes médicas.
        
        CLAHE mejora el contraste local mientras previene la sobre-amplificación
        del ruido, siendo especialmente efectivo para imágenes de rayos X.
        
        Args:
            gray_image: Imagen en escala de grises (numpy array)
            clip_limit: Límite de contraste (default: 2.0 - óptimo para rayos X)
            tile_size: Tamaño de tiles para ecualización adaptiva (default: (8,8))
                      Balance entre preservación de detalle local y eficiencia
        
        Returns:
            Imagen procesada con CLAHE aplicado
        """
        # Crear objeto CLAHE con parámetros científicamente optimizados
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        
        # Aplicar CLAHE
        clahe_image = clahe.apply(gray_image)
        
        return clahe_image

    def find_image_path(self, filename: str) -> Optional[Path]:
        """
        Encuentra la ruta completa de una imagen en el dataset.
        
        Args:
            filename: Nombre del archivo (con o sin extensión)
            
        Returns:
            Path completo de la imagen o None si no se encuentra
        """
        # Asegurar extensión .png
        if not filename.endswith('.png'):
            filename += '.png'
        
        # Buscar en cada categoría
        categories = ['COVID', 'Normal', 'Viral_Pneumonia']
        
        for category in categories:
            # Manejar nombres con espacios para Viral_Pneumonia
            if category == 'Viral_Pneumonia':
                # Probar con espacios y guiones
                variations = [
                    filename,
                    filename.replace('-', ' '),
                    filename.replace('_', ' ')
                ]
            else:
                variations = [filename]
            
            for variation in variations:
                image_path = self.dataset_path / category / variation
                if image_path.exists():
                    return image_path
        
        return None

    def extract_landmark_crop(self, image: np.ndarray, landmark_coords: Tuple[int, int], 
                            landmark_id: str) -> Optional[np.ndarray]:
        """
        Extrae un recorte de landmark usando el template óptimo correspondiente.
        
        Args:
            image: Imagen original (299x299)
            landmark_coords: Coordenadas (x, y) del landmark
            landmark_id: ID del landmark (ej: 'L1', 'L2')
            
        Returns:
            Recorte de la imagen o None si hay error
        """
        if landmark_id not in self.templates_data:
            self.logger.error(f"Template no encontrado para {landmark_id}")
            return None
        
        template = self.templates_data[landmark_id]
        extensions = template['template_extensions']
        
        x, y = landmark_coords
        
        # Calcular límites del recorte usando las extensiones del template
        crop_left = x - extensions['left']
        crop_right = x + extensions['right']
        crop_top = y - extensions['up']
        crop_bottom = y + extensions['down']
        
        # Validar que el recorte esté dentro de los límites de la imagen (299x299)
        img_height, img_width = image.shape[:2]
        
        if (crop_left < 0 or crop_right >= img_width or 
            crop_top < 0 or crop_bottom >= img_height):
            self.logger.warning(
                f"Recorte fuera de límites para {landmark_id}: "
                f"left={crop_left}, right={crop_right}, top={crop_top}, bottom={crop_bottom}"
            )
            return None
        
        # Extraer recorte
        crop = image[crop_top:crop_bottom+1, crop_left:crop_right+1]
        
        # Validar dimensiones del recorte
        expected_width = template['template_dimensions']['width']
        expected_height = template['template_dimensions']['height']
        
        actual_height, actual_width = crop.shape[:2]
        
        if actual_width != expected_width or actual_height != expected_height:
            self.logger.error(
                f"Dimensiones incorrectas para {landmark_id}: "
                f"esperado {expected_width}x{expected_height}, "
                f"obtenido {actual_width}x{actual_height}"
            )
            return None
        
        return crop

    def process_single_image(self, coord_data: Dict) -> Dict:
        """
        Procesa una sola imagen extrayendo recortes de los landmarks objetivo.
        
        Args:
            coord_data: Diccionario con datos de coordenadas de la imagen
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        filename = coord_data['filename']
        points = coord_data['points']
        
        result = {
            'filename': filename,
            'landmarks_processed': {},
            'errors': []
        }
        
        # Encontrar la imagen
        image_path = self.find_image_path(filename)
        if not image_path:
            error = f"Imagen no encontrada: {filename}"
            result['errors'].append(error)
            return result
        
        # Cargar imagen
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                error = f"Error cargando imagen: {filename}"
                result['errors'].append(error)
                return result
            
            # Validar dimensiones
            if image.shape[:2] != (299, 299):
                error = f"Dimensiones incorrectas: {filename} - {image.shape[:2]}"
                result['errors'].append(error)
                return result
            
            # Convertir a escala de grises para procesamiento de landmarks
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar CLAHE para normalización de contraste
            # Esto mejora la robustez del modelo PCA ante variaciones de iluminación
            image_processed = self.apply_clahe(image_gray)
                
        except Exception as e:
            error = f"Error procesando imagen {filename}: {str(e)}"
            result['errors'].append(error)
            return result
        
        # Procesar cada landmark objetivo
        for landmark_id in self.target_landmarks:
            # Extraer número del landmark (L1 -> 1, L15 -> 15)
            try:
                landmark_number = int(landmark_id[1:])  # Remover 'L' y convertir a int
                landmark_index = landmark_number - 1    # Convertir a índice 0-based
            except (ValueError, IndexError):
                error = f"Formato de landmark inválido: {landmark_id}"
                result['errors'].append(error)
                continue
            
            if landmark_index >= len(points):
                error = f"No hay coordenadas suficientes para {landmark_id} en {filename} (índice {landmark_index}, disponibles {len(points)})"
                result['errors'].append(error)
                continue
            
            landmark_coords = points[landmark_index]  # L1 = points[0], L2 = points[1], L15 = points[14]
            
            # Extraer recorte usando la imagen procesada con CLAHE
            crop = self.extract_landmark_crop(image_processed, landmark_coords, landmark_id)
            
            if crop is not None:
                # Guardar recorte
                output_filename = f"{filename.replace('.png', '')}_{landmark_id}.png"
                output_path = self.output_path / landmark_id / output_filename
                
                try:
                    cv2.imwrite(str(output_path), crop)
                    result['landmarks_processed'][landmark_id] = {
                        'success': True,
                        'output_path': str(output_path),
                        'crop_size': crop.shape[:2],
                        'coordinates': landmark_coords
                    }
                except Exception as e:
                    error = f"Error guardando recorte {landmark_id} para {filename}: {str(e)}"
                    result['errors'].append(error)
                    result['landmarks_processed'][landmark_id] = {'success': False, 'error': error}
            else:
                result['landmarks_processed'][landmark_id] = {
                    'success': False,
                    'error': 'Extracción de recorte falló'
                }
        
        return result

    def process_images(self, max_images: Optional[int] = None) -> Dict:
        """
        Procesa todas las imágenes del dataset de entrenamiento.
        
        Args:
            max_images: Límite máximo de imágenes a procesar (para pruebas)
            
        Returns:
            Diccionario con estadísticas del procesamiento
        """
        self.logger.info(f"Iniciando procesamiento de imágenes para landmarks: {self.target_landmarks}")
        
        # Preparar datos
        images_to_process = self.coordinates_data
        if max_images:
            images_to_process = images_to_process[:max_images]
            self.logger.info(f"Limitando procesamiento a {max_images} imágenes")
        
        self.extraction_stats['total_images'] = len(images_to_process)
        
        # Inicializar contadores
        for landmark in self.target_landmarks:
            self.extraction_stats['successful_extractions'][landmark] = 0
            self.extraction_stats['failed_extractions'][landmark] = 0
        
        # Procesar imágenes con barra de progreso
        successful_images = 0
        
        for coord_data in tqdm(images_to_process, desc="Procesando imágenes"):
            result = self.process_single_image(coord_data)
            
            # Actualizar estadísticas
            has_success = False
            for landmark_id in self.target_landmarks:
                if landmark_id in result['landmarks_processed']:
                    if result['landmarks_processed'][landmark_id]['success']:
                        self.extraction_stats['successful_extractions'][landmark_id] += 1
                        has_success = True
                    else:
                        self.extraction_stats['failed_extractions'][landmark_id] += 1
            
            if has_success:
                successful_images += 1
            
            # Agregar errores globales
            if result['errors']:
                self.extraction_stats['errors'].extend(result['errors'])
        
        # Generar reporte final
        self.extraction_stats['successful_images'] = successful_images
        self._save_extraction_report()
        
        return self.extraction_stats

    def _save_extraction_report(self):
        """Guarda un reporte detallado de la extracción."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'processing_summary': {
                'total_images': self.extraction_stats['total_images'],
                'successful_images': self.extraction_stats['successful_images'],
                'landmarks_processed': self.target_landmarks
            },
            'landmark_statistics': {},
            'template_info': {},
            'total_errors': len(self.extraction_stats['errors']),
            'error_samples': self.extraction_stats['errors'][:10]  # Primeros 10 errores
        }
        
        # Estadísticas por landmark
        for landmark in self.target_landmarks:
            successful = self.extraction_stats['successful_extractions'][landmark]
            failed = self.extraction_stats['failed_extractions'][landmark]
            total = successful + failed
            
            report['landmark_statistics'][landmark] = {
                'successful_extractions': successful,
                'failed_extractions': failed,
                'total_attempts': total,
                'success_rate': successful / total if total > 0 else 0
            }
            
            # Información del template
            if landmark in self.templates_data:
                template = self.templates_data[landmark]
                report['template_info'][landmark] = {
                    'dimensions': template['template_dimensions'],
                    'anchor_point': template['anchor_point'],
                    'extensions': template['template_extensions']
                }
        
        # Guardar reporte
        report_path = self.output_path / 'extraction_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Reporte de extracción guardado en: {report_path}")
        
        # Log de resumen
        self.logger.info("=== RESUMEN DE PROCESAMIENTO ===")
        self.logger.info(f"Total de imágenes procesadas: {report['processing_summary']['total_images']}")
        self.logger.info(f"Imágenes con al menos un éxito: {report['processing_summary']['successful_images']}")
        
        for landmark in self.target_landmarks:
            stats = report['landmark_statistics'][landmark]
            self.logger.info(
                f"{landmark}: {stats['successful_extractions']}/{stats['total_attempts']} "
                f"({stats['success_rate']:.2%} éxito)"
            )


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Extractor de recortes de landmarks usando templates óptimos"
    )
    parser.add_argument(
        '--csv', 
        default='data/coordenadas/coordenadas_train.csv',
        help='Ruta al archivo CSV con coordenadas maestras'
    )
    parser.add_argument(
        '--bbox',
        default='landmark_bounding_boxes_corrected.json',
        help='Ruta al archivo JSON con bounding boxes'
    )
    parser.add_argument(
        '--templates',
        default='optimal_templates_fixed.json',
        help='Ruta al archivo JSON con templates óptimos'
    )
    parser.add_argument(
        '--dataset',
        default='data/dataset',
        help='Directorio con imágenes del dataset'
    )
    parser.add_argument(
        '--output',
        default='output_landmarks',
        help='Directorio de salida para recortes'
    )
    parser.add_argument(
        '--landmarks',
        nargs='+',
        default=[f'L{i}' for i in range(1, 16)],
        help='Landmarks a procesar (default: L1-L15)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        help='Número máximo de imágenes a procesar (para pruebas)'
    )
    
    args = parser.parse_args()
    
    # Crear extractor
    cropper = LandmarkCropper(
        csv_path=args.csv,
        bbox_path=args.bbox,
        templates_path=args.templates,
        dataset_path=args.dataset,
        output_path=args.output,
        target_landmarks=args.landmarks
    )
    
    # Procesar imágenes
    stats = cropper.process_images(max_images=args.max_images)
    
    print(f"\n✅ Procesamiento completado!")
    print(f"📊 Ver detalles en: {args.output}/extraction_report.json")


if __name__ == "__main__":
    main()