#!/usr/bin/env python3
"""
Generador de Bounding Boxes Corregidos

Este script regenera los bounding boxes usando rangos reales (min/max)
en lugar de rangos estadísticos (mean ± 2×std), solucionando el problema
de outliers que causaba fallos en la extracción de recortes.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List


class CorrectedBoundingBoxGenerator:
    def __init__(self, coordinates_path: str, output_path: str):
        """
        Inicializar generador de bounding boxes corregidos.
        
        Args:
            coordinates_path: Ruta al CSV con coordenadas
            output_path: Ruta de salida para JSON corregido
        """
        self.coordinates_path = Path(coordinates_path)
        self.output_path = Path(output_path)
        
        # Datos
        self.coordinates_data = None
        self.landmark_coords = {}
        self.corrected_bboxes = {}

    def load_coordinates(self) -> List[Dict]:
        """Cargar coordenadas desde CSV."""
        print(f"Cargando coordenadas desde: {self.coordinates_path}")
        
        coordinates = []
        
        with open(self.coordinates_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split(',')
                    if len(parts) != 32:
                        continue
                    
                    # Extraer coordenadas
                    coords_data = [float(x) for x in parts[1:-1]]
                    filename = parts[-1]
                    
                    # Convertir a pares de coordenadas
                    points = []
                    for i in range(0, len(coords_data), 2):
                        points.append((coords_data[i], coords_data[i+1]))
                    
                    coordinates.append({
                        'filename': filename,
                        'points': points
                    })
                    
                except Exception as e:
                    print(f"ERROR procesando línea {line_num}: {e}")
                    continue
        
        print(f"Coordenadas cargadas: {len(coordinates)} imágenes")
        return coordinates

    def organize_coordinates_by_landmark(self):
        """Organizar coordenadas por landmark."""
        print("Organizando coordenadas por landmark...")
        
        # Inicializar para todos los landmarks
        for i in range(15):
            landmark_id = f"L{i+1}"
            self.landmark_coords[landmark_id] = {
                'x_values': [],
                'y_values': []
            }
        
        # Recopilar coordenadas
        for coord_data in self.coordinates_data:
            points = coord_data['points']
            
            for i, (x, y) in enumerate(points):
                landmark_id = f"L{i+1}"
                self.landmark_coords[landmark_id]['x_values'].append(x)
                self.landmark_coords[landmark_id]['y_values'].append(y)
        
        # Calcular estadísticas
        for landmark_id in self.landmark_coords:
            coords = self.landmark_coords[landmark_id]
            
            x_values = coords['x_values']
            y_values = coords['y_values']
            
            coords['x_stats'] = {
                'min': float(min(x_values)),
                'max': float(max(x_values)),
                'mean': float(np.mean(x_values)),
                'std': float(np.std(x_values)),
                'median': float(np.median(x_values)),
                'q25': float(np.percentile(x_values, 25)),
                'q75': float(np.percentile(x_values, 75)),
                'count': len(x_values)
            }
            
            coords['y_stats'] = {
                'min': float(min(y_values)),
                'max': float(max(y_values)),
                'mean': float(np.mean(y_values)),
                'std': float(np.std(y_values)),
                'median': float(np.median(y_values)),
                'q25': float(np.percentile(y_values, 25)),
                'q75': float(np.percentile(y_values, 75)),
                'count': len(y_values)
            }
            
            print(f"{landmark_id}: X[{coords['x_stats']['min']:.0f}-{coords['x_stats']['max']:.0f}], "
                  f"Y[{coords['y_stats']['min']:.0f}-{coords['y_stats']['max']:.0f}]")

    def generate_real_range_bboxes(self):
        """Generar bounding boxes usando rangos reales completos."""
        print("\nGenerando bounding boxes con rangos reales...")
        
        for landmark_id in self.landmark_coords:
            coords = self.landmark_coords[landmark_id]
            
            # Usar rangos reales exactos
            real_x_min = coords['x_stats']['min']
            real_x_max = coords['x_stats']['max']
            real_y_min = coords['y_stats']['min']
            real_y_max = coords['y_stats']['max']
            
            # Agregar margen mínimo de seguridad (1 píxel) y asegurar límites de imagen
            safe_x_min = max(0, real_x_min - 1)
            safe_x_max = min(298, real_x_max + 1)  # Coordenadas van 0-298
            safe_y_min = max(0, real_y_min - 1)
            safe_y_max = min(298, real_y_max + 1)
            
            # Calcular bbox
            bbox_width = safe_x_max - safe_x_min
            bbox_height = safe_y_max - safe_y_min
            center_x = (safe_x_min + safe_x_max) / 2
            center_y = (safe_y_min + safe_y_max) / 2
            area = bbox_width * bbox_height
            
            corrected_bbox = {
                'bbox': {
                    'x': safe_x_min,
                    'y': safe_y_min,
                    'width': bbox_width,
                    'height': bbox_height,
                    'center_x': center_x,
                    'center_y': center_y,
                    'area': area
                },
                'statistics': coords,
                'method_used': 'real_range',
                'safety_margin': 1.0
            }
            
            self.corrected_bboxes[landmark_id] = corrected_bbox
            
            print(f"{landmark_id}: [{safe_x_min:.0f}-{safe_x_max:.0f}] × [{safe_y_min:.0f}-{safe_y_max:.0f}] "
                  f"({bbox_width:.0f}×{bbox_height:.0f}, área: {area:.0f})")

    def validate_coverage(self):
        """Validar que los bounding boxes cubren 100% de coordenadas."""
        print("\nValidando cobertura de bounding boxes...")
        
        all_valid = True
        
        for landmark_id in self.corrected_bboxes:
            bbox = self.corrected_bboxes[landmark_id]['bbox']
            coords = self.landmark_coords[landmark_id]
            
            bbox_x_min = bbox['x']
            bbox_x_max = bbox['x'] + bbox['width']
            bbox_y_min = bbox['y']
            bbox_y_max = bbox['y'] + bbox['height']
            
            # Verificar todas las coordenadas
            outliers = 0
            for x, y in zip(coords['x_values'], coords['y_values']):
                if x < bbox_x_min or x > bbox_x_max or y < bbox_y_min or y > bbox_y_max:
                    outliers += 1
            
            coverage = ((len(coords['x_values']) - outliers) / len(coords['x_values'])) * 100
            
            if outliers == 0:
                print(f"✅ {landmark_id}: 100% cobertura ({len(coords['x_values'])} coordenadas)")
            else:
                print(f"❌ {landmark_id}: {coverage:.1f}% cobertura ({outliers} outliers)")
                all_valid = False
        
        if all_valid:
            print("✅ Todos los bounding boxes cubren 100% de las coordenadas")
        else:
            print("❌ Algunos bounding boxes no cubren todas las coordenadas")
        
        return all_valid

    def save_corrected_bboxes(self):
        """Guardar bounding boxes corregidos."""
        # Preparar metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'source_file': str(self.coordinates_path),
            'algorithm': 'real_range_corrected',
            'total_images_analyzed': len(self.coordinates_data),
            'landmarks_per_image': 15,
            'total_landmarks_analyzed': len(self.coordinates_data) * 15,
            'image_dimensions': '299x299',
            'bbox_method': 'real_range',
            'safety_margin': 1.0,
            'coverage': '100%'
        }
        
        # Calcular estadísticas globales
        all_areas = [bbox_data['bbox']['area'] for bbox_data in self.corrected_bboxes.values()]
        all_widths = [bbox_data['bbox']['width'] for bbox_data in self.corrected_bboxes.values()]
        all_heights = [bbox_data['bbox']['height'] for bbox_data in self.corrected_bboxes.values()]
        
        global_statistics = {
            'bbox_areas': {
                'min': float(min(all_areas)),
                'max': float(max(all_areas)),
                'mean': float(np.mean(all_areas)),
                'std': float(np.std(all_areas))
            },
            'bbox_dimensions': {
                'width': {
                    'min': float(min(all_widths)),
                    'max': float(max(all_widths)),
                    'mean': float(np.mean(all_widths))
                },
                'height': {
                    'min': float(min(all_heights)),
                    'max': float(max(all_heights)),
                    'mean': float(np.mean(all_heights))
                }
            }
        }
        
        # Estructura final
        output_data = {
            'metadata': metadata,
            'global_statistics': global_statistics,
            'landmark_bounding_boxes': self.corrected_bboxes
        }
        
        # Guardar archivo
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Bounding boxes corregidos guardados en: {self.output_path}")
        
        # Resumen
        print(f"\n📊 RESUMEN:")
        print(f"  Total de landmarks procesados: {len(self.corrected_bboxes)}")
        print(f"  Método: rangos reales (min/max) + margen de seguridad")
        print(f"  Cobertura garantizada: 100%")
        print(f"  Área promedio de bboxes: {global_statistics['bbox_areas']['mean']:.0f} px²")

    def generate_corrected_bboxes(self):
        """Generar bounding boxes corregidos completos."""
        print("🔧 GENERANDO BOUNDING BOXES CORREGIDOS")
        print("="*50)
        
        # Cargar y procesar datos
        self.coordinates_data = self.load_coordinates()
        self.organize_coordinates_by_landmark()
        
        # Generar bboxes corregidos
        self.generate_real_range_bboxes()
        
        # Validar
        is_valid = self.validate_coverage()
        
        if is_valid:
            # Guardar
            self.save_corrected_bboxes()
            print("\n✅ BOUNDING BOXES CORREGIDOS GENERADOS EXITOSAMENTE")
            return True
        else:
            print("\n❌ ERROR: Algunos bounding boxes no cubren todas las coordenadas")
            return False


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Generar bounding boxes corregidos")
    parser.add_argument('--coordinates', default='data/coordenadas/coordenadas_maestro.csv',
                       help='Ruta al archivo CSV con coordenadas (usar dataset completo)')
    parser.add_argument('--output', default='landmark_bounding_boxes_corrected.json',
                       help='Ruta de salida para bounding boxes corregidos')
    
    args = parser.parse_args()
    
    # Generar bounding boxes corregidos
    generator = CorrectedBoundingBoxGenerator(args.coordinates, args.output)
    success = generator.generate_corrected_bboxes()
    
    if success:
        print("\n🎉 PROCESO COMPLETADO")
        print("Los bounding boxes corregidos están listos para generar templates.")
    else:
        print("\n❌ PROCESO FALLÓ")
        exit(1)


if __name__ == "__main__":
    main()