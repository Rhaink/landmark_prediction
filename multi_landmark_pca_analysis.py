#!/usr/bin/env python3
"""
Análisis PCA Multi-Landmark para Todos los Landmarks L1-L15
==========================================================

Este script extiende el análisis PCA original para procesar todos los landmarks
L1 a L15 de forma automática y eficiente, manteniendo la calidad científica
del análisis original.

Funcionalidades:
- Procesamiento automático de 15 landmarks (L1-L15)
- Detección automática de dimensiones por landmark
- Gestión eficiente de memoria para 10,035 imágenes totales
- Reportes consolidados y comparativos entre landmarks
- Estructura de salida organizada por landmark
- Barras de progreso y logging detallado

Autor: Sistema de desarrollo con IA especializada
Fecha: 2025-08-19
Basado en: pca_eigenfaces_analysis.py
"""

import os
import sys
import gc
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Importar la clase original
try:
    from pca_eigenfaces_analysis import LandmarkPCAAnalysis
except ImportError:
    print("Error: No se pudo importar LandmarkPCAAnalysis desde pca_eigenfaces_analysis.py")
    print("Asegúrese de que el archivo pca_eigenfaces_analysis.py esté en el mismo directorio.")
    sys.exit(1)

class MultiLandmarkPCAProcessor:
    """
    Procesador principal para análisis PCA de múltiples landmarks
    """
    
    def __init__(self, landmarks_base_path="output_landmarks_complete", 
                 output_base_path="output_pca_analysis_all_landmarks"):
        """
        Inicializar el procesador multi-landmark
        
        Args:
            landmarks_base_path (str): Ruta base donde están los directorios L1-L15
            output_base_path (str): Ruta base donde guardar todos los resultados
        """
        self.landmarks_base_path = Path(landmarks_base_path)
        self.output_base_path = Path(output_base_path)
        
        # Validar que el directorio de landmarks existe
        if not self.landmarks_base_path.exists():
            raise FileNotFoundError(f"Directorio de landmarks no encontrado: {self.landmarks_base_path}")
        
        # Crear directorio de salida
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        self._setup_logging()
        
        # Detectar landmarks disponibles
        self.available_landmarks = self._detect_available_landmarks()
        
        # Almacenar información de procesamiento
        self.processing_results = {}
        self.consolidated_data = {}
        
        self.logger.info(f"MultiLandmarkPCAProcessor inicializado")
        self.logger.info(f"Landmarks detectados: {self.available_landmarks}")
        self.logger.info(f"Directorio de salida: {self.output_base_path}")
    
    def _setup_logging(self):
        """
        Configurar sistema de logging
        """
        log_file = self.output_base_path / "processing_log.txt"
        
        # Configurar logging con formato detallado
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== Inicio del Análisis PCA Multi-Landmark ===")
    
    def _detect_available_landmarks(self) -> List[str]:
        """
        Detectar automáticamente los landmarks disponibles (L1-L15)
        
        Returns:
            List[str]: Lista de landmarks disponibles ordenados
        """
        landmarks = []
        
        for i in range(1, 16):  # L1 a L15
            landmark_dir = self.landmarks_base_path / f"L{i}"
            if landmark_dir.exists() and landmark_dir.is_dir():
                # Verificar que tenga imágenes PNG
                png_files = list(landmark_dir.glob("*.png"))
                if png_files:
                    landmarks.append(f"L{i}")
                    self.logger.info(f"Landmark L{i} detectado: {len(png_files)} imágenes")
                else:
                    self.logger.warning(f"Landmark L{i}: directorio existe pero sin imágenes PNG")
            else:
                self.logger.warning(f"Landmark L{i}: directorio no encontrado")
        
        return landmarks
    
    def detect_landmark_dimensions(self, landmark: str) -> Tuple[int, int]:
        """
        Detectar automáticamente las dimensiones de las imágenes de un landmark
        
        Args:
            landmark (str): Nombre del landmark (ej: "L1")
            
        Returns:
            Tuple[int, int]: (width, height) de las imágenes
        """
        landmark_path = self.landmarks_base_path / landmark
        
        # Buscar la primera imagen PNG disponible
        png_files = list(landmark_path.glob("*.png"))
        if not png_files:
            raise FileNotFoundError(f"No se encontraron imágenes PNG en {landmark_path}")
        
        # Cargar la primera imagen para detectar dimensiones
        sample_img_path = png_files[0]
        img = cv2.imread(str(sample_img_path))
        
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {sample_img_path}")
        
        height, width = img.shape[:2]
        
        self.logger.info(f"{landmark}: Dimensiones detectadas {width}x{height} píxeles")
        return width, height
    
    def process_single_landmark(self, landmark: str) -> Dict:
        """
        Procesar un solo landmark usando LandmarkPCAAnalysis
        
        Args:
            landmark (str): Nombre del landmark (ej: "L1")
            
        Returns:
            Dict: Información del procesamiento
        """
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Procesando {landmark}")
        self.logger.info(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            # Detectar dimensiones automáticamente
            width, height = self.detect_landmark_dimensions(landmark)
            
            # Configurar rutas
            input_path = self.landmarks_base_path / landmark
            output_path = self.output_base_path / landmark
            
            # Crear analizador con dimensiones específicas
            analyzer = LandmarkPCAAnalysis(
                input_path=str(input_path),
                output_path=str(output_path),
                image_size=(width, height)
            )
            
            # Ejecutar análisis completo (igual que en el script original)
            self.logger.info(f"{landmark}: Cargando y preprocesando imágenes...")
            analyzer.load_and_preprocess_images()
            
            self.logger.info(f"{landmark}: Computando análisis PCA...")
            analyzer.compute_pca()
            
            self.logger.info(f"{landmark}: Generando visualizaciones...")
            analyzer.save_mean_image()
            analyzer.save_top_eigenfaces()
            analyzer.save_cumulative_variance_plot()
            analyzer.save_reconstructions()
            analyzer.save_2d_projection()
            
            self.logger.info(f"{landmark}: Guardando modelo entrenado...")
            model_file = analyzer.save_trained_model()
            
            self.logger.info(f"{landmark}: Generando reportes...")
            report = analyzer.generate_analysis_report()
            
            # Obtener estadísticas del modelo
            model_summary = analyzer.get_model_summary()
            
            # Calcular tiempo de procesamiento
            processing_time = time.time() - start_time
            
            # Información del procesamiento (convertir tipos numpy a Python nativos)
            labels, counts = np.unique(analyzer.image_labels, return_counts=True)
            categories = {str(label): int(count) for label, count in zip(labels, counts)}
            
            processing_info = {
                'landmark': landmark,
                'status': 'completed',
                'processing_time_seconds': float(processing_time),
                'dimensions': f"{width}x{height}",
                'total_images': int(len(analyzer.image_filenames)),
                'categories': categories,
                'pca_components': int(analyzer.pca_model.n_components_),
                'variance_top5': [float(x) for x in analyzer.pca_model.explained_variance_ratio_[:5]],
                'components_for_90_percent': int(np.argmax(np.cumsum(analyzer.pca_model.explained_variance_ratio_) >= 0.9)) + 1,
                'components_for_95_percent': int(np.argmax(np.cumsum(analyzer.pca_model.explained_variance_ratio_) >= 0.95)) + 1,
                'model_file': str(model_file),
                'output_directory': str(output_path)
            }
            
            # Liberar memoria explícitamente
            del analyzer
            gc.collect()
            
            self.logger.info(f"{landmark}: ✅ COMPLETADO en {processing_time:.1f}s")
            return processing_info
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_info = {
                'landmark': landmark,
                'status': 'failed',
                'error': str(e),
                'processing_time_seconds': float(processing_time)
            }
            
            self.logger.error(f"{landmark}: ❌ ERROR después de {processing_time:.1f}s: {e}")
            return error_info
    
    def process_all_landmarks(self, landmarks: Optional[List[str]] = None) -> Dict:
        """
        Procesar todos los landmarks disponibles
        
        Args:
            landmarks (Optional[List[str]]): Lista específica de landmarks a procesar.
                                           Si es None, procesa todos los disponibles.
        
        Returns:
            Dict: Resumen completo del procesamiento
        """
        if landmarks is None:
            landmarks = self.available_landmarks
        
        self.logger.info(f"\n🚀 INICIANDO PROCESAMIENTO DE {len(landmarks)} LANDMARKS")
        self.logger.info(f"Landmarks a procesar: {landmarks}")
        
        total_start_time = time.time()
        
        # Procesar cada landmark secuencialmente
        for i, landmark in enumerate(tqdm(landmarks, desc="Procesando landmarks"), 1):
            self.logger.info(f"\n[{i}/{len(landmarks)}] Procesando {landmark}")
            
            result = self.process_single_landmark(landmark)
            self.processing_results[landmark] = result
            
            # Mostrar progreso
            if result['status'] == 'completed':
                self.logger.info(f"✅ {landmark} completado ({i}/{len(landmarks)})")
            else:
                self.logger.error(f"❌ {landmark} falló ({i}/{len(landmarks)})")
        
        total_processing_time = time.time() - total_start_time
        
        # Generar estadísticas finales
        completed = [r for r in self.processing_results.values() if r['status'] == 'completed']
        failed = [r for r in self.processing_results.values() if r['status'] == 'failed']
        
        summary = {
            'total_landmarks_attempted': int(len(landmarks)),
            'completed_successfully': int(len(completed)),
            'failed': int(len(failed)),
            'total_processing_time_seconds': float(total_processing_time),
            'total_processing_time_formatted': f"{total_processing_time/60:.1f} minutos",
            'completed_landmarks': [r['landmark'] for r in completed],
            'failed_landmarks': [r['landmark'] for r in failed],
            'processing_details': self.processing_results
        }
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🏁 PROCESAMIENTO COMPLETADO")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"✅ Exitosos: {len(completed)}/{len(landmarks)} landmarks")
        self.logger.info(f"❌ Fallidos: {len(failed)}/{len(landmarks)} landmarks")
        self.logger.info(f"⏱️  Tiempo total: {total_processing_time/60:.1f} minutos")
        
        if failed:
            self.logger.warning(f"Landmarks fallidos: {[r['landmark'] for r in failed]}")
        
        return summary
    
    def generate_consolidated_reports(self):
        """
        Generar reportes consolidados comparando todos los landmarks procesados
        """
        self.logger.info("\n📊 Generando reportes consolidados...")
        
        # Crear directorio para reportes consolidados
        consolidated_dir = self.output_base_path / "consolidated_reports"
        consolidated_dir.mkdir(exist_ok=True)
        
        # Filtrar solo los landmarks completados exitosamente
        completed_results = {k: v for k, v in self.processing_results.items() 
                           if v['status'] == 'completed'}
        
        if not completed_results:
            self.logger.warning("No hay landmarks completados para generar reportes consolidados")
            return
        
        # Generar reporte de comparación JSON
        self._generate_comparison_summary(consolidated_dir, completed_results)
        
        # Generar gráfico de comparación de varianza
        self._generate_variance_comparison_plot(consolidated_dir, completed_results)
        
        # Generar grid de eigenfaces principales
        self._generate_landmarks_eigenfaces_grid(consolidated_dir, completed_results)
        
        self.logger.info(f"📊 Reportes consolidados guardados en: {consolidated_dir}")
    
    def _generate_comparison_summary(self, output_dir: Path, completed_results: Dict):
        """
        Generar resumen de comparación en formato JSON
        """
        comparison_data = {
            'processing_summary': {
                'total_landmarks_processed': int(len(completed_results)),
                'landmarks': list(completed_results.keys()),
                'total_processing_time_minutes': float(sum(r['processing_time_seconds'] for r in completed_results.values()) / 60)
            },
            'dimensions_analysis': {},
            'variance_analysis': {},
            'dataset_analysis': {}
        }
        
        # Análisis de dimensiones
        for landmark, result in completed_results.items():
            comparison_data['dimensions_analysis'][landmark] = {
                'dimensions': result['dimensions'],
                'total_images': result['total_images'],
                'categories': result['categories']
            }
        
        # Análisis de varianza
        for landmark, result in completed_results.items():
            comparison_data['variance_analysis'][landmark] = {
                'pca_components': result['pca_components'],
                'variance_top5': result['variance_top5'],
                'components_for_90_percent': result['components_for_90_percent'],
                'components_for_95_percent': result['components_for_95_percent']
            }
        
        # Estadísticas generales del dataset
        total_images = sum(r['total_images'] for r in completed_results.values())
        all_categories = {}
        for result in completed_results.values():
            for category, count in result['categories'].items():
                all_categories[category] = all_categories.get(category, 0) + count
        
        comparison_data['dataset_analysis'] = {
            'total_images_all_landmarks': int(total_images),
            'average_images_per_landmark': float(total_images / len(completed_results)),
            'categories_distribution': all_categories
        }
        
        # Guardar reporte
        summary_file = output_dir / "comparison_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        self.logger.info(f"Resumen de comparación guardado: {summary_file}")
    
    def _generate_variance_comparison_plot(self, output_dir: Path, completed_results: Dict):
        """
        Generar gráfico comparativo de varianza explicada entre landmarks
        """
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Varianza de los primeros 5 componentes
        plt.subplot(2, 2, 1)
        landmarks = list(completed_results.keys())
        variance_data = []
        
        for landmark in landmarks:
            variance_data.append(completed_results[landmark]['variance_top5'])
        
        variance_df = pd.DataFrame(variance_data, 
                                 index=landmarks, 
                                 columns=[f'PC{i+1}' for i in range(5)])
        
        variance_df.plot(kind='bar', ax=plt.gca())
        plt.title('Varianza Explicada - Primeros 5 Componentes por Landmark')
        plt.ylabel('Varianza Explicada (%)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Subplot 2: Componentes necesarios para 90% y 95% de varianza
        plt.subplot(2, 2, 2)
        components_90 = [completed_results[landmark]['components_for_90_percent'] for landmark in landmarks]
        components_95 = [completed_results[landmark]['components_for_95_percent'] for landmark in landmarks]
        
        x = np.arange(len(landmarks))
        width = 0.35
        
        plt.bar(x - width/2, components_90, width, label='90% Varianza', alpha=0.8)
        plt.bar(x + width/2, components_95, width, label='95% Varianza', alpha=0.8)
        
        plt.xlabel('Landmarks')
        plt.ylabel('Número de Componentes')
        plt.title('Componentes Necesarios para 90% y 95% de Varianza')
        plt.xticks(x, landmarks, rotation=45)
        plt.legend()
        
        # Subplot 3: Distribución de dimensiones
        plt.subplot(2, 2, 3)
        dimensions = [completed_results[landmark]['dimensions'] for landmark in landmarks]
        
        # Extraer width y height
        widths = []
        heights = []
        for dim in dimensions:
            w, h = map(int, dim.split('x'))
            widths.append(w)
            heights.append(h)
        
        plt.scatter(widths, heights, s=100, alpha=0.7)
        for i, landmark in enumerate(landmarks):
            plt.annotate(landmark, (widths[i], heights[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.xlabel('Ancho (píxeles)')
        plt.ylabel('Alto (píxeles)')
        plt.title('Distribución de Dimensiones por Landmark')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Número de componentes PCA por landmark
        plt.subplot(2, 2, 4)
        pca_components = [completed_results[landmark]['pca_components'] for landmark in landmarks]
        
        plt.bar(landmarks, pca_components, alpha=0.7)
        plt.xlabel('Landmarks')
        plt.ylabel('Número de Componentes PCA')
        plt.title('Componentes PCA Generados por Landmark')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_file = output_dir / "variance_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Gráfico de comparación de varianza guardado: {plot_file}")
    
    def _generate_landmarks_eigenfaces_grid(self, output_dir: Path, completed_results: Dict):
        """
        Generar grid con la primera eigenface de cada landmark procesado
        """
        try:
            landmarks = sorted(completed_results.keys(), key=lambda x: int(x[1:]))  # Ordenar L1, L2, ..., L15
            
            # Cargar primera eigenface de cada landmark
            eigenfaces_data = []
            valid_landmarks = []
            
            for landmark in landmarks:
                eigenface_path = self.output_base_path / landmark / "individual_eigenfaces" / "eigenface_1.png"
                
                if eigenface_path.exists():
                    eigenface = cv2.imread(str(eigenface_path), cv2.IMREAD_GRAYSCALE)
                    if eigenface is not None:
                        eigenfaces_data.append(eigenface)
                        valid_landmarks.append(landmark)
                        self.logger.debug(f"Eigenface cargada para {landmark}: {eigenface.shape}")
                    else:
                        self.logger.warning(f"No se pudo cargar eigenface para {landmark}")
                else:
                    self.logger.warning(f"Eigenface no encontrada para {landmark}: {eigenface_path}")
            
            if not eigenfaces_data:
                self.logger.warning("No se encontraron eigenfaces para generar el grid")
                return
            
            # Crear grid de eigenfaces
            n_landmarks = len(eigenfaces_data)
            cols = 5  # 5 columnas
            rows = (n_landmarks + cols - 1) // cols
            
            # Crear figura
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (eigenface, landmark) in enumerate(zip(eigenfaces_data, valid_landmarks)):
                row = i // cols
                col = i % cols
                
                ax = axes[row, col]
                ax.imshow(eigenface, cmap='gray', interpolation='nearest')
                ax.set_title(f'{landmark}', fontsize=10, fontweight='bold')
                ax.axis('off')
            
            # Ocultar axes vacíos
            for i in range(n_landmarks, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')
            
            plt.suptitle('Primera Eigenface de Cada Landmark', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Guardar grid
            grid_file = output_dir / "landmarks_eigenfaces_grid.png"
            plt.savefig(grid_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Grid de eigenfaces guardado: {grid_file}")
            
        except Exception as e:
            self.logger.error(f"Error generando grid de eigenfaces: {e}")
    
    def save_final_summary(self, processing_summary: Dict):
        """
        Guardar resumen final del procesamiento completo
        """
        summary_file = self.output_base_path / "final_processing_summary.json"
        
        # Agregar metadata adicional
        final_summary = {
            'processing_metadata': {
                'start_time': str(datetime.now()),
                'landmarks_base_path': str(self.landmarks_base_path),
                'output_base_path': str(self.output_base_path),
                'available_landmarks': self.available_landmarks
            },
            'processing_results': processing_summary
        }
        
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        self.logger.info(f"📄 Resumen final guardado: {summary_file}")


def main():
    """
    Función principal para ejecutar el análisis PCA multi-landmark
    """
    print("="*80)
    print("🧬 ANÁLISIS PCA MULTI-LANDMARK PARA LANDMARKS L1-L15")
    print("="*80)
    print()
    
    try:
        # Crear procesador principal
        processor = MultiLandmarkPCAProcessor()
        
        # Procesar todos los landmarks disponibles
        processing_summary = processor.process_all_landmarks()
        
        # Generar reportes consolidados
        processor.generate_consolidated_reports()
        
        # Guardar resumen final
        processor.save_final_summary(processing_summary)
        
        # Mostrar estadísticas finales
        print("\n" + "="*80)
        print("🎉 ANÁLISIS MULTI-LANDMARK COMPLETADO")
        print("="*80)
        print(f"✅ Landmarks procesados exitosamente: {processing_summary['completed_successfully']}")
        print(f"❌ Landmarks fallidos: {processing_summary['failed']}")
        print(f"⏱️  Tiempo total de procesamiento: {processing_summary['total_processing_time_formatted']}")
        print(f"📁 Resultados guardados en: {processor.output_base_path}")
        
        if processing_summary['failed'] > 0:
            print(f"⚠️  Landmarks fallidos: {processing_summary['failed_landmarks']}")
        
        print("\n🔍 Estructura de resultados generada:")
        print("📁 output_pca_analysis_all_landmarks/")
        print("   ├── L1/, L2/, ..., L15/ (análisis completos)")
        print("   ├── consolidated_reports/ (comparaciones)")
        print("   ├── final_processing_summary.json")
        print("   └── processing_log.txt")
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        raise


if __name__ == "__main__":
    main()