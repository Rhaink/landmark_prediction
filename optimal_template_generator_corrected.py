#!/usr/bin/env python3
"""
VERSIÓN CORREGIDA del generador de templates óptimos.

Esta versión corrige el error conceptual del algoritmo original:
- Los templates se definen como EXTENSIONES desde el punto de anclaje
- No como rectángulos absolutos con posición fija
- Garantiza que el template funcione sin importar dónde se mueva el anclaje dentro del bbox
"""

import json
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime
import time


class OptimalTemplateGeneratorCorrected:
    def __init__(self, landmark_bbox_path, output_path):
        self.landmark_bbox_path = Path(landmark_bbox_path)
        self.output_path = Path(output_path)
        
        # Dimensiones de imagen fijas
        self.image_width = 299
        self.image_height = 299
        
        # Datos cargados
        self.landmark_bboxes = None
        self.optimal_templates = {}
        
    def load_landmark_bboxes(self):
        """Cargar datos de landmark bounding boxes."""
        print(f"Cargando landmark bounding boxes desde: {self.landmark_bbox_path}")
        
        if not self.landmark_bbox_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.landmark_bbox_path}")
        
        with open(self.landmark_bbox_path, 'r') as f:
            data = json.load(f)
        
        self.landmark_bboxes = data['landmark_bounding_boxes']
        
        print(f"Datos cargados:")
        print(f"  - Landmark bounding boxes: {len(self.landmark_bboxes)}")
        print(f"  - Imagen: {self.image_width}x{self.image_height} píxeles")
        
        return self.landmark_bboxes
    
    def calculate_optimal_template_extensions(self, bbox):
        """
        Calcular las extensiones óptimas del template desde cualquier punto de anclaje.
        
        ALGORITMO CORREGIDO:
        El template se define por sus extensiones desde el punto de anclaje:
        - left_extension: píxeles hacia la izquierda del anclaje
        - right_extension: píxeles hacia la derecha del anclaje  
        - up_extension: píxeles hacia arriba del anclaje
        - down_extension: píxeles hacia abajo del anclaje
        
        Para que funcione con cualquier anclaje en el bbox, las extensiones deben ser
        tales que el template nunca se salga sin importar dónde se coloque el anclaje.
        
        Args:
            bbox: Bounding box donde el anclaje puede moverse
            
        Returns:
            dict: Extensiones del template y información asociada
        """
        # Límites del bounding box (redondeados)
        bbox_left = int(bbox['x'])
        bbox_right = int(bbox['x'] + bbox['width'] - 1)
        bbox_top = int(bbox['y'])
        bbox_bottom = int(bbox['y'] + bbox['height'] - 1)
        
        # Asegurar que están dentro de la imagen
        bbox_left = max(0, bbox_left)
        bbox_right = min(self.image_width - 1, bbox_right)
        bbox_top = max(0, bbox_top)
        bbox_bottom = min(self.image_height - 1, bbox_bottom)
        
        # ALGORITMO CORREGIDO:
        # Para que el template funcione en cualquier posición del anclaje dentro del bbox:
        
        # Extensión máxima hacia la izquierda:
        # Cuando el anclaje está en la posición MÁS A LA IZQUIERDA del bbox (bbox_left),
        # el template puede extenderse hasta el borde izquierdo de la imagen (0)
        max_left_extension = bbox_left - 0  # = bbox_left
        
        # Extensión máxima hacia la derecha:
        # Cuando el anclaje está en la posición MÁS A LA DERECHA del bbox (bbox_right),
        # el template puede extenderse hasta el borde derecho de la imagen (width-1)
        max_right_extension = (self.image_width - 1) - bbox_right
        
        # Extensión máxima hacia arriba:
        # Cuando el anclaje está en la posición MÁS ARRIBA del bbox (bbox_top),
        # el template puede extenderse hasta el borde superior de la imagen (0)
        max_up_extension = bbox_top - 0  # = bbox_top
        
        # Extensión máxima hacia abajo:
        # Cuando el anclaje está en la posición MÁS ABAJO del bbox (bbox_bottom),
        # el template puede extenderse hasta el borde inferior de la imagen (height-1)
        max_down_extension = (self.image_height - 1) - bbox_bottom
        
        # Dimensiones totales del template
        total_width = max_left_extension + max_right_extension + 1  # +1 por el píxel del anclaje
        total_height = max_up_extension + max_down_extension + 1   # +1 por el píxel del anclaje
        total_area = total_width * total_height
        
        # Punto de anclaje óptimo (centro del bounding box)
        optimal_anchor_x = (bbox_left + bbox_right) // 2
        optimal_anchor_y = (bbox_top + bbox_bottom) // 2
        
        return {
            'extensions': {
                'left': max_left_extension,
                'right': max_right_extension,
                'up': max_up_extension,
                'down': max_down_extension
            },
            'template_dimensions': {
                'width': total_width,
                'height': total_height,
                'area': total_area
            },
            'optimal_anchor': {
                'x': optimal_anchor_x,
                'y': optimal_anchor_y
            },
            'bbox_info': {
                'left': bbox_left,
                'right': bbox_right,
                'top': bbox_top,
                'bottom': bbox_bottom,
                'width': bbox_right - bbox_left + 1,
                'height': bbox_bottom - bbox_top + 1
            }
        }
    
    def process_landmark(self, landmark_name, landmark_data):
        """
        Procesar un landmark individual para encontrar su template y anclaje óptimo.
        
        Args:
            landmark_name: Nombre del landmark (ej: "L1", "L2")
            landmark_data: Datos del landmark con bbox y estadísticas
            
        Returns:
            dict: Resultado del procesamiento del landmark
        """
        print(f"  Procesando {landmark_name}...")
        
        bbox = landmark_data['bbox']
        
        # Mostrar información del bounding box
        print(f"    Bounding box: {bbox['width']:.0f}x{bbox['height']:.0f} px "
              f"en ({bbox['x']:.0f},{bbox['y']:.0f})")
        
        start_time = time.time()
        
        # Calcular template óptimo con extensiones
        template_result = self.calculate_optimal_template_extensions(bbox)
        
        processing_time = time.time() - start_time
        
        # Construir resultado
        result = {
            'anchor_point': template_result['optimal_anchor'],
            'template_extensions': template_result['extensions'],
            'template_dimensions': template_result['template_dimensions'],
            'metrics': {
                'max_area_found': template_result['template_dimensions']['area'],
                'positions_evaluated': 1,
                'bbox_coverage': {
                    'anchor_relative_x': 0.5,  # Centro del bbox
                    'anchor_relative_y': 0.5   # Centro del bbox
                }
            },
            'processing': {
                'method': 'corrected_extensions',
                'processing_time_seconds': processing_time,
                'original_bbox': bbox
            },
            'bbox_analysis': template_result['bbox_info']
        }
        
        # Mostrar resultado
        anchor = result['anchor_point']
        dims = result['template_dimensions']
        exts = result['template_extensions']
        
        print(f"    Resultado: Anclaje en ({anchor['x']},{anchor['y']}) -> "
              f"Template {dims['width']}x{dims['height']} px "
              f"(área: {dims['area']:,} px²)")
        print(f"    Extensiones: L={exts['left']}, R={exts['right']}, "
              f"U={exts['up']}, D={exts['down']}")
        print(f"    Eficiencia: {(dims['area']/(self.image_width*self.image_height))*100:.1f}%")
        print(f"    Tiempo: {processing_time:.3f}s")
        
        return result
    
    def process_all_landmarks(self):
        """Procesar todos los landmarks para encontrar templates y anclajes óptimos."""
        print(f"\nProcesando {len(self.landmark_bboxes)} landmarks con algoritmo corregido...")
        
        self.optimal_templates = {}
        success_count = 0
        total_time = 0
        
        for landmark_name, landmark_data in self.landmark_bboxes.items():
            result = self.process_landmark(landmark_name, landmark_data)
            
            if result is not None:
                self.optimal_templates[landmark_name] = result
                success_count += 1
                total_time += result['processing']['processing_time_seconds']
            else:
                print(f"    FALLO: No se pudo procesar {landmark_name}")
        
        print(f"\nProcesamiento completado:")
        print(f"  - Landmarks exitosos: {success_count}/{len(self.landmark_bboxes)}")
        print(f"  - Tiempo total: {total_time:.3f}s")
        print(f"  - Tiempo promedio por landmark: {total_time/success_count:.3f}s")
        
        return self.optimal_templates
    
    def save_results(self):
        """Guardar resultados en archivo JSON."""
        print(f"\nGuardando resultados en: {self.output_path}")
        
        # Calcular estadísticas globales
        template_areas = [data['template_dimensions']['area'] for data in self.optimal_templates.values()]
        template_widths = [data['template_dimensions']['width'] for data in self.optimal_templates.values()]
        template_heights = [data['template_dimensions']['height'] for data in self.optimal_templates.values()]
        
        # Estructura de salida
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source_file': str(self.landmark_bbox_path),
                'algorithm': 'optimal_template_generator_corrected',
                'search_method': 'corrected_extensions',
                'image_dimensions': f'{self.image_width}x{self.image_height}',
                'total_landmarks_processed': len(self.optimal_templates),
                'algorithm_description': 'Templates definidos como extensiones desde el punto de anclaje',
                'processing_summary': {
                    'avg_template_area': float(np.mean(template_areas)),
                    'max_template_area': float(np.max(template_areas)),
                    'min_template_area': float(np.min(template_areas))
                }
            },
            'global_statistics': {
                'template_areas': {
                    'min': float(np.min(template_areas)),
                    'max': float(np.max(template_areas)),
                    'mean': float(np.mean(template_areas)),
                    'std': float(np.std(template_areas))
                },
                'template_dimensions': {
                    'width': {
                        'min': float(np.min(template_widths)),
                        'max': float(np.max(template_widths)),
                        'mean': float(np.mean(template_widths))
                    },
                    'height': {
                        'min': float(np.min(template_heights)),
                        'max': float(np.max(template_heights)),
                        'mean': float(np.mean(template_heights))
                    }
                }
            },
            'optimal_templates_corrected': self.optimal_templates
        }
        
        # Guardar archivo
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Archivo guardado exitosamente: {self.output_path}")
        
        return output_data
    
    def run(self):
        """Ejecutar el proceso completo de generación de templates óptimos corregidos."""
        print("=== Generador de Templates Óptimos (VERSIÓN CORREGIDA) ===")
        print(f"Entrada: {self.landmark_bbox_path}")
        print(f"Salida: {self.output_path}")
        print(f"Dimensiones de imagen: {self.image_width}x{self.image_height}")
        print("Algoritmo: Templates como extensiones desde el punto de anclaje")
        
        try:
            # 1. Cargar datos de landmark bounding boxes
            self.load_landmark_bboxes()
            
            # 2. Procesar todos los landmarks
            self.process_all_landmarks()
            
            # 3. Guardar resultados
            output_data = self.save_results()
            
            # 4. Mostrar resumen final
            print(f"\n=== Resumen Final ===")
            print(f"Templates óptimos generados: {len(self.optimal_templates)}")
            
            if len(self.optimal_templates) > 0:
                avg_area = output_data['global_statistics']['template_areas']['mean']
                max_area = output_data['global_statistics']['template_areas']['max']
                print(f"Área promedio de templates: {avg_area:,.0f} px²")
                print(f"Área máxima encontrada: {max_area:,.0f} px²")
                print(f"Eficiencia promedio: {(avg_area/(self.image_width*self.image_height))*100:.1f}% de la imagen")
            
            print(f"\n✅ Proceso completado exitosamente")
            print(f"\n🔧 DIFERENCIAS CON LA VERSIÓN ANTERIOR:")
            print(f"   - Templates definidos como EXTENSIONES desde el anclaje")
            print(f"   - Garantizado funcionamiento en cualquier posición del anclaje")
            print(f"   - Algoritmo conceptualmente correcto")
            
            return True
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Generar templates óptimos CORREGIDOS con extensiones desde puntos de anclaje"
    )
    parser.add_argument("--input", default="landmark_bounding_boxes_corrected.json",
                       help="Archivo con landmark bounding boxes (default: landmark_bounding_boxes_corrected.json)")
    parser.add_argument("--output", default="optimal_templates_fixed.json",
                       help="Archivo de salida (default: optimal_templates_fixed.json)")
    
    args = parser.parse_args()
    
    # Crear generador corregido
    generator = OptimalTemplateGeneratorCorrected(
        landmark_bbox_path=args.input,
        output_path=args.output
    )
    
    # Ejecutar
    success = generator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()