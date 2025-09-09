#!/usr/bin/env python3
"""
Validador Unificado de Templates

Combina validación teórica y práctica en un solo script simple:
- Validación matemática del algoritmo (exhaustiva)
- Validación con coordenadas reales del dataset
- Reportes consolidados y claros
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Tuple
import time


class UnifiedTemplateValidator:
    def __init__(self, coordinates_path: str, templates_path: str, bbox_path: str):
        """
        Inicializar validador unificado.
        
        Args:
            coordinates_path: Ruta al CSV con coordenadas
            templates_path: Ruta al JSON con templates
            bbox_path: Ruta al JSON con bounding boxes
        """
        self.coordinates_path = Path(coordinates_path)
        self.templates_path = Path(templates_path)
        self.bbox_path = Path(bbox_path)
        
        # Dimensiones de imagen
        self.image_width = 299
        self.image_height = 299
        
        # Datos
        self.coordinates_data = None
        self.templates_data = None
        self.bbox_data = None
        
        # Resultados
        self.theoretical_results = {}
        self.practical_results = {}

    def load_all_data(self):
        """Cargar todos los datos necesarios."""
        print("🔄 Cargando datos...")
        
        # Templates
        with open(self.templates_path, 'r') as f:
            self.templates_data = json.load(f)
        
        # Bounding boxes
        with open(self.bbox_path, 'r') as f:
            self.bbox_data = json.load(f)
        
        # Coordenadas
        self.coordinates_data = []
        with open(self.coordinates_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split(',')
                    if len(parts) != 32:
                        continue
                    
                    # Extraer coordenadas (saltar índice y tomar filename)
                    coords = [float(x) for x in parts[1:-1]]  # Saltar índice y filename
                    filename = parts[-1]
                    
                    # Organizar en pares (x,y) para landmarks L1-L15
                    points = {}
                    for i in range(15):
                        x = coords[i*2]
                        y = coords[i*2 + 1]
                        points[f'L{i+1}'] = (int(x), int(y))
                    
                    self.coordinates_data.append({
                        'filename': filename,
                        'landmarks': points
                    })
                
                except (ValueError, IndexError) as e:
                    print(f"⚠️  Error en línea {line_num}: {e}")
        
        print(f"✅ Datos cargados:")
        print(f"   - Templates: {len(self.templates_data['optimal_templates_corrected'])}")
        print(f"   - Bounding boxes: {len(self.bbox_data['landmark_bounding_boxes'])}")
        print(f"   - Coordenadas: {len(self.coordinates_data)} imágenes")

    def validate_theoretical_algorithm(self, landmarks_to_test: List[str] = None):
        """
        Validación teórica: probar todas las posiciones del anclaje.
        Equivalente a validate_corrected_templates.py
        """
        if landmarks_to_test is None:
            landmarks_to_test = list(self.templates_data['optimal_templates_corrected'].keys())
        
        print(f"\n🧮 VALIDACIÓN TEÓRICA (Algoritmo Matemático)")
        print(f"Probando todas las posiciones del anclaje para {len(landmarks_to_test)} landmarks...")
        
        passed_count = 0
        failed_count = 0
        
        for landmark_name in sorted(landmarks_to_test):
            print(f"\n  📐 Validando {landmark_name}...")
            
            # Obtener datos
            template_data = self.templates_data['optimal_templates_corrected'][landmark_name]
            bbox_data = self.bbox_data['landmark_bounding_boxes'][landmark_name]['bbox']
            
            extensions = template_data['template_extensions']
            dimensions = template_data['template_dimensions']
            
            # Límites del bounding box
            bbox_left = max(0, int(bbox_data['x']))
            bbox_right = min(self.image_width - 1, int(bbox_data['x'] + bbox_data['width'] - 1))
            bbox_top = max(0, int(bbox_data['y']))
            bbox_bottom = min(self.image_height - 1, int(bbox_data['y'] + bbox_data['height'] - 1))
            
            # Probar todas las posiciones del anclaje
            total_positions = (bbox_right - bbox_left + 1) * (bbox_bottom - bbox_top + 1)
            valid_positions = 0
            
            print(f"    Template: {dimensions['width']}x{dimensions['height']} px")
            print(f"    Probando {total_positions:,} posiciones del anclaje...")
            
            for anchor_y in range(bbox_top, bbox_bottom + 1):
                for anchor_x in range(bbox_left, bbox_right + 1):
                    # Calcular límites del template
                    template_left = anchor_x - extensions['left']
                    template_right = anchor_x + extensions['right']
                    template_top = anchor_y - extensions['up']
                    template_bottom = anchor_y + extensions['down']
                    
                    # Verificar límites
                    if (template_left >= 0 and template_right <= self.image_width - 1 and
                        template_top >= 0 and template_bottom <= self.image_height - 1):
                        valid_positions += 1
            
            # Calcular resultado
            success_rate = (valid_positions / total_positions) * 100
            passed = success_rate == 100.0
            
            result = {
                'landmark_name': landmark_name,
                'total_positions': total_positions,
                'valid_positions': valid_positions,
                'success_rate': success_rate,
                'passed': passed,
                'template_info': {
                    'dimensions': dimensions,
                    'extensions': extensions
                }
            }
            
            self.theoretical_results[landmark_name] = result
            
            if passed:
                passed_count += 1
                print(f"    ✅ PASÓ - {success_rate:.1f}% posiciones válidas")
            else:
                failed_count += 1
                print(f"    ❌ FALLÓ - {success_rate:.1f}% posiciones válidas")
        
        return passed_count, failed_count

    def validate_practical_coordinates(self, landmarks_to_test: List[str] = ['L1', 'L2']):
        """
        Validación práctica: probar con coordenadas reales del dataset.
        Equivalente a validate_fixed_templates.py
        """
        print(f"\n🎯 VALIDACIÓN PRÁCTICA (Coordenadas Reales)")
        print(f"Probando coordenadas del dataset para landmarks: {landmarks_to_test}")
        
        # Estadísticas generales
        total_tested = 0
        total_valid = 0
        landmark_stats = {landmark: {'valid': 0, 'invalid': 0, 'errors': []} 
                         for landmark in landmarks_to_test}
        
        for img_data in self.coordinates_data:
            for landmark_id in landmarks_to_test:
                if landmark_id not in img_data['landmarks']:
                    continue
                
                x, y = img_data['landmarks'][landmark_id]
                template = self.templates_data['optimal_templates_corrected'][landmark_id]
                extensions = template['template_extensions']
                
                # Calcular límites del recorte
                crop_left = x - extensions['left']
                crop_right = x + extensions['right']
                crop_top = y - extensions['up']
                crop_bottom = y + extensions['down']
                
                # Validar límites
                valid = (crop_left >= 0 and crop_right < self.image_width and 
                        crop_top >= 0 and crop_bottom < self.image_height)
                
                total_tested += 1
                if valid:
                    total_valid += 1
                    landmark_stats[landmark_id]['valid'] += 1
                else:
                    landmark_stats[landmark_id]['invalid'] += 1
                    # Registrar error (solo primeros 3)
                    if len(landmark_stats[landmark_id]['errors']) < 3:
                        violation = []
                        if crop_left < 0: violation.append(f"left={crop_left}")
                        if crop_right >= self.image_width: violation.append(f"right={crop_right}")
                        if crop_top < 0: violation.append(f"top={crop_top}")
                        if crop_bottom >= self.image_height: violation.append(f"bottom={crop_bottom}")
                        
                        landmark_stats[landmark_id]['errors'].append({
                            'filename': img_data['filename'],
                            'coordinates': (x, y),
                            'violations': violation
                        })
        
        # Mostrar resultados por landmark
        for landmark_id in landmarks_to_test:
            stats = landmark_stats[landmark_id]
            total = stats['valid'] + stats['invalid']
            success_rate = (stats['valid'] / total * 100) if total > 0 else 0
            
            result = {
                'landmark_name': landmark_id,
                'total_coordinates': total,
                'valid_coordinates': stats['valid'],
                'invalid_coordinates': stats['invalid'],
                'success_rate': success_rate,
                'sample_errors': stats['errors']
            }
            
            self.practical_results[landmark_id] = result
            
            print(f"\n  📊 {landmark_id}:")
            print(f"    Coordenadas probadas: {total}")
            print(f"    Válidas: {stats['valid']} ({success_rate:.1f}%)")
            print(f"    Inválidas: {stats['invalid']}")
            
            if stats['errors']:
                print(f"    Primeros errores:")
                for error in stats['errors'][:2]:
                    print(f"      {error['filename']}: {error['coordinates']} -> {error['violations']}")
        
        overall_success_rate = (total_valid / total_tested * 100) if total_tested > 0 else 0
        print(f"\n  📈 Resumen general:")
        print(f"    Total de pruebas: {total_tested}")
        print(f"    Éxito general: {overall_success_rate:.1f}%")
        
        return overall_success_rate >= 99.0  # 99% o más se considera éxito

    def generate_unified_report(self):
        """Generar reporte consolidado."""
        print(f"\n" + "="*60)
        print(f"📋 REPORTE UNIFICADO DE VALIDACIÓN")
        print(f"="*60)
        
        # Resumen teórico
        theoretical_total = len(self.theoretical_results)
        theoretical_passed = sum(1 for r in self.theoretical_results.values() if r['passed'])
        
        print(f"\n🧮 VALIDACIÓN TEÓRICA:")
        print(f"   Landmarks probados: {theoretical_total}")
        print(f"   Algoritmo correcto: {theoretical_passed}/{theoretical_total}")
        print(f"   Tasa de éxito: {(theoretical_passed/theoretical_total)*100:.1f}%")
        
        # Resumen práctico
        practical_total = len(self.practical_results)
        practical_success = sum(1 for r in self.practical_results.values() 
                               if r['success_rate'] >= 99.0)
        
        print(f"\n🎯 VALIDACIÓN PRÁCTICA:")
        print(f"   Landmarks probados: {practical_total}")
        print(f"   Funcionamiento correcto: {practical_success}/{practical_total}")
        
        # Estado general
        all_theoretical_passed = theoretical_passed == theoretical_total
        all_practical_passed = practical_success == practical_total
        
        print(f"\n🎉 ESTADO GENERAL:")
        if all_theoretical_passed and all_practical_passed:
            print(f"   ✅ VALIDACIÓN COMPLETA EXITOSA")
            print(f"   ✅ Templates funcionan correctamente")
            print(f"   ✅ Listos para usar en producción")
            return True
        else:
            print(f"   ❌ VALIDACIÓN FALLIDA")
            if not all_theoretical_passed:
                print(f"   ❌ Errores en algoritmo teórico")
            if not all_practical_passed:
                print(f"   ❌ Errores con coordenadas reales")
            return False

    def run_full_validation(self, landmarks_to_test: List[str] = None):
        """Ejecutar validación completa."""
        print("🧪 INICIANDO VALIDACIÓN UNIFICADA DE TEMPLATES")
        print("="*55)
        
        start_time = time.time()
        
        try:
            # Cargar datos
            self.load_all_data()
            
            # Validación teórica (todos los landmarks)
            theoretical_passed, theoretical_failed = self.validate_theoretical_algorithm(landmarks_to_test)
            
            # Validación práctica (landmarks especificados)
            if landmarks_to_test is None:
                practical_landmarks = ['L1', 'L2']  # Por defecto
            else:
                practical_landmarks = landmarks_to_test
            
            practical_success = self.validate_practical_coordinates(practical_landmarks)
            
            # Generar reporte
            overall_success = self.generate_unified_report()
            
            elapsed_time = time.time() - start_time
            print(f"\n⏱️  Tiempo total de validación: {elapsed_time:.2f}s")
            
            return overall_success
            
        except Exception as e:
            print(f"\n❌ ERROR durante la validación: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Validador unificado de templates (teórico + práctico)"
    )
    parser.add_argument('--coordinates', default='data/coordenadas/coordenadas_maestro.csv',
                       help='Ruta al archivo CSV con coordenadas')
    parser.add_argument('--templates', default='optimal_templates_fixed.json',
                       help='Ruta al archivo JSON con templates')
    parser.add_argument('--bbox', default='landmark_bounding_boxes_corrected.json',
                       help='Ruta al archivo JSON con bounding boxes')
    parser.add_argument('--landmarks', nargs='+', 
                       help='Landmarks específicos a validar (default: todos para teórico, L1 L2 para práctico)')
    parser.add_argument('--practical-only', action='store_true',
                       help='Ejecutar solo validación práctica (más rápido)')
    
    args = parser.parse_args()
    
    # Crear validador
    validator = UnifiedTemplateValidator(
        coordinates_path=args.coordinates,
        templates_path=args.templates,
        bbox_path=args.bbox
    )
    
    # Ejecutar validación
    if args.practical_only:
        print("🚀 Modo rápido: Solo validación práctica")
        validator.load_all_data()
        success = validator.validate_practical_coordinates(args.landmarks or ['L1', 'L2'])
        print(f"\n{'✅ ÉXITO' if success else '❌ FALLO'}: Validación práctica")
    else:
        success = validator.run_full_validation(args.landmarks)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()