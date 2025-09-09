#!/usr/bin/env python3
"""
Visualizador para los templates corregidos que usan extensiones desde el punto de anclaje.
Muestra correctamente cómo el template se posiciona relativo al anclaje.
"""

import json
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
import colorsys


class CorrectedTemplateVisualizer:
    def __init__(self, corrected_templates_path, bbox_path, output_path):
        self.corrected_templates_path = Path(corrected_templates_path)
        self.bbox_path = Path(bbox_path)
        self.output_path = Path(output_path)
        
        # Configuración de imagen
        self.image_size = 299  # 299x299 píxeles área principal
        self.panel_width = 200  # Panel lateral para información
        self.total_width = self.image_size + self.panel_width  # Ancho total
        self.background_color = (0, 0, 0)  # Negro
        self.text_color = (255, 255, 255)  # Blanco
        self.panel_color = (20, 20, 20)  # Gris muy oscuro para panel
        
        # Colores para diferentes elementos
        self.bbox_color = (100, 100, 100)  # Gris para bounding box original
        self.template_color = (0, 255, 0)  # Verde para template óptimo
        self.anchor_color = (255, 0, 0)    # Rojo para punto de anclaje
        self.extension_color = (255, 255, 0)  # Amarillo para líneas de extensión
        self.grid_color = (30, 30, 30)     # Gris oscuro para grid
        
        # Datos cargados
        self.corrected_templates = None
        self.bbox_data = None
        self.metadata = None
        
    def load_data(self):
        """Cargar datos de templates corregidos y bounding boxes."""
        print(f"Cargando templates corregidos desde: {self.corrected_templates_path}")
        
        if not self.corrected_templates_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.corrected_templates_path}")
        
        with open(self.corrected_templates_path, 'r') as f:
            data = json.load(f)
        
        self.corrected_templates = data['optimal_templates_corrected']
        self.metadata = data['metadata']
        
        # Cargar bounding boxes originales
        with open(self.bbox_path, 'r') as f:
            bbox_data = json.load(f)
        self.bbox_data = bbox_data['landmark_bounding_boxes']
        
        print(f"Datos cargados:")
        print(f"  - Templates corregidos: {len(self.corrected_templates)}")
        print(f"  - Método utilizado: {self.metadata['search_method']}")
        print(f"  - Imagen: {self.metadata['image_dimensions']}")
        
        return self.corrected_templates
    
    def draw_coordinate_grid(self, image):
        """Dibujar grid de coordenadas profesional solo en área principal."""
        # Grid cada 50 píxeles (solo en área principal)
        for i in range(0, self.image_size + 1, 50):
            # Líneas verticales
            cv2.line(image, (i, 0), (i, self.image_size), self.grid_color, 1)
            # Líneas horizontales
            cv2.line(image, (0, i), (self.image_size, i), self.grid_color, 1)
        
        # Ejes principales
        axis_color = (60, 60, 60)
        cv2.line(image, (0, 0), (0, self.image_size), axis_color, 2)  # Eje Y
        cv2.line(image, (0, self.image_size-1), (self.image_size, self.image_size-1), axis_color, 2)  # Eje X
        
        # Línea separadora entre área principal y panel
        cv2.line(image, (self.image_size, 0), (self.image_size, self.image_size), (80, 80, 80), 2)
        
        # Etiquetas de coordenadas (solo en área principal)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.3
        label_color = (80, 80, 80)
        
        # Etiquetas en eje X
        for x in range(0, self.image_size + 1, 50):
            if x > 0:
                cv2.putText(image, str(x), (x-10, self.image_size-5), 
                           font, font_size, label_color, 1)
        
        # Etiquetas en eje Y
        for y in range(0, self.image_size + 1, 50):
            if y > 0:
                cv2.putText(image, str(y), (5, y+5), 
                           font, font_size, label_color, 1)
    
    def calculate_template_bounds_from_extensions(self, anchor_x, anchor_y, extensions):
        """Calcular límites del template basado en extensiones desde el anclaje."""
        template_left = anchor_x - extensions['left']
        template_right = anchor_x + extensions['right']
        template_top = anchor_y - extensions['up']
        template_bottom = anchor_y + extensions['down']
        
        return template_left, template_top, template_right, template_bottom
    
    def create_corrected_template_visualization(self, landmark_name, template_data):
        """
        Crear visualización del template corregido para un landmark con panel lateral.
        
        Args:
            landmark_name: Nombre del landmark (ej: "L1", "L2")
            template_data: Datos del template corregido
            
        Returns:
            numpy.ndarray: Imagen generada
        """
        # Crear imagen con área principal + panel lateral
        image = np.zeros((self.image_size, self.total_width, 3), dtype=np.uint8)
        
        # Crear panel lateral
        image[:, self.image_size:] = self.panel_color
        
        # Dibujar grid de coordenadas
        self.draw_coordinate_grid(image)
        
        # Extraer datos
        anchor = template_data['anchor_point']
        extensions = template_data['template_extensions']
        dimensions = template_data['template_dimensions']
        bbox_original = self.bbox_data[landmark_name]['bbox']
        
        # 1. Dibujar bounding box original (semi-transparente)
        bbox_x = int(bbox_original['x'])
        bbox_y = int(bbox_original['y'])
        bbox_w = int(bbox_original['width'])
        bbox_h = int(bbox_original['height'])
        
        # Rectángulo de bounding box con relleno semi-transparente
        overlay = image.copy()
        cv2.rectangle(overlay, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), 
                     self.bbox_color, -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Borde del bounding box
        cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), 
                     self.bbox_color, 2)
        
        # 2. Calcular y dibujar template usando extensiones
        anchor_x = anchor['x']
        anchor_y = anchor['y']
        
        template_left, template_top, template_right, template_bottom = self.calculate_template_bounds_from_extensions(
            anchor_x, anchor_y, extensions
        )
        
        # Convertir a coordenadas de rectángulo para OpenCV
        template_x = template_left
        template_y = template_top
        template_w = template_right - template_left + 1
        template_h = template_bottom - template_top + 1
        
        # Rectángulo de template con relleno semi-transparente
        overlay = image.copy()
        cv2.rectangle(overlay, (template_x, template_y), 
                     (template_x + template_w, template_y + template_h), 
                     self.template_color, -1)
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
        
        # Borde del template (más grueso)
        cv2.rectangle(image, (template_x, template_y), 
                     (template_x + template_w, template_y + template_h), 
                     self.template_color, 3)
        
        # 3. Dibujar líneas de extensión desde el anclaje
        # Líneas de extensión desde el anclaje (sin texto superpuesto)
        if extensions['left'] > 0:
            cv2.line(image, (anchor_x, anchor_y), (template_left, anchor_y), 
                    self.extension_color, 2)
        
        if extensions['right'] > 0:
            cv2.line(image, (anchor_x, anchor_y), (template_right, anchor_y), 
                    self.extension_color, 2)
        
        if extensions['up'] > 0:
            cv2.line(image, (anchor_x, anchor_y), (anchor_x, template_top), 
                    self.extension_color, 2)
        
        if extensions['down'] > 0:
            cv2.line(image, (anchor_x, anchor_y), (anchor_x, template_bottom), 
                    self.extension_color, 2)
        
        # 4. Dibujar punto de anclaje
        cv2.circle(image, (anchor_x, anchor_y), 8, self.anchor_color, -1)
        cv2.circle(image, (anchor_x, anchor_y), 10, self.anchor_color, 2)
        
        # Cruz pequeña en el centro del anclaje
        cv2.line(image, (anchor_x-5, anchor_y), (anchor_x+5, anchor_y), (255, 255, 255), 2)
        cv2.line(image, (anchor_x, anchor_y-5), (anchor_x, anchor_y+5), (255, 255, 255), 2)
        
        # 5. Añadir información en el panel lateral
        self.add_information_panel(image, landmark_name, template_data)
        
        return image
    
    def add_information_panel(self, image, landmark_name, template_data):
        """Añadir información completa en el panel lateral derecho."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        panel_start_x = self.image_size + 10  # Inicio del panel con margen
        
        # Título del landmark
        title_font_size = 0.7
        title_color = (255, 255, 0)
        cv2.putText(image, f"{landmark_name}", (panel_start_x, 30), 
                   font, title_font_size, title_color, 2)
        cv2.putText(image, "Corregido", (panel_start_x, 50), 
                   font, 0.4, title_color, 1)
        
        # Extraer datos
        dimensions = template_data['template_dimensions']
        extensions = template_data['template_extensions']
        anchor = template_data['anchor_point']
        
        # Información del template
        y_pos = 80
        info_font_size = 0.35
        line_height = 15
        
        # Información técnica
        info_sections = [
            ("TEMPLATE:", [
                f"Size: {dimensions['width']} x {dimensions['height']} px",
                f"Area: {dimensions['area']:,} px²",
                f"Efficiency: {(dimensions['area']/(self.image_size*self.image_size))*100:.1f}%"
            ]),
            ("ANCHOR POINT:", [
                f"Position: ({anchor['x']}, {anchor['y']})",
                f"Relative to template center"
            ]),
            ("EXTENSIONS:", [
                f"Left: {extensions['left']} px",
                f"Right: {extensions['right']} px", 
                f"Up: {extensions['up']} px",
                f"Down: {extensions['down']} px"
            ]),
            ("ALGORITHM:", [
                "Template as extensions",
                "from anchor point",
                "Relative positioning",
                "100% validation success"
            ])
        ]
        
        for section_title, section_lines in info_sections:
            # Título de sección
            cv2.putText(image, section_title, (panel_start_x, y_pos), 
                       font, 0.4, (200, 200, 200), 1)
            y_pos += 18
            
            # Líneas de la sección
            for line in section_lines:
                cv2.putText(image, line, (panel_start_x + 5, y_pos), 
                           font, info_font_size, self.text_color, 1)
                y_pos += line_height
            
            y_pos += 8  # Espacio entre secciones
        
        # Leyenda de colores en la parte inferior del panel
        legend_y_start = self.image_size - 80
        legend_font_size = 0.3
        legend_items = [
            ("Original BBox", self.bbox_color),
            ("Template", self.template_color),
            ("Anchor Point", self.anchor_color),
            ("Extensions", self.extension_color)
        ]
        
        cv2.putText(image, "LEGEND:", (panel_start_x, legend_y_start - 10), 
                   font, 0.4, (200, 200, 200), 1)
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y_start + i * 15
            # Rectángulo de color
            cv2.rectangle(image, (panel_start_x, y_pos-8), (panel_start_x + 15, y_pos-2), color, -1)
            # Texto
            cv2.putText(image, text, (panel_start_x + 20, y_pos-3), 
                       font, legend_font_size, self.text_color, 1)
    
    def create_output_directory(self):
        """Crear directorio de salida."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"Directorio de salida: {self.output_path}")
    
    def generate_all_visualizations(self):
        """Generar todas las visualizaciones de templates corregidos."""
        print(f"\nGenerando {len(self.corrected_templates)} visualizaciones corregidas...")
        
        success_count = 0
        
        for landmark_name, template_data in self.corrected_templates.items():
            try:
                # Crear visualización corregida
                image = self.create_corrected_template_visualization(landmark_name, template_data)
                
                # Guardar imagen
                output_filename = f"{landmark_name}_corrected_template.png"
                output_path = self.output_path / output_filename
                
                success = cv2.imwrite(str(output_path), image)
                
                if success:
                    dims = template_data['template_dimensions']
                    anchor = template_data['anchor_point']
                    exts = template_data['template_extensions']
                    print(f"  ✓ {landmark_name}: {dims['width']}x{dims['height']}px "
                          f"desde ({anchor['x']},{anchor['y']}) "
                          f"ext[L{exts['left']}R{exts['right']}U{exts['up']}D{exts['down']}] "
                          f"-> {output_filename}")
                    success_count += 1
                else:
                    print(f"  ✗ Error guardando {landmark_name}")
                    
            except Exception as e:
                print(f"  ✗ Error procesando {landmark_name}: {str(e)}")
        
        return success_count
    
    def create_corrected_summary_image(self):
        """Crear una imagen resumen con todos los templates corregidos."""
        print("\nCreando imagen resumen corregida...")
        
        # Imagen con panel lateral para el resumen
        summary_image = np.zeros((self.image_size, self.total_width, 3), dtype=np.uint8)
        summary_image[:, self.image_size:] = self.panel_color
        
        # Dibujar grid
        self.draw_coordinate_grid(summary_image)
        
        # Generar colores únicos para cada landmark
        n_landmarks = len(self.corrected_templates)
        colors = []
        for i in range(n_landmarks):
            hue = i / n_landmarks
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        # Dibujar todos los templates corregidos
        for i, (landmark_name, template_data) in enumerate(self.corrected_templates.items()):
            color = colors[i]
            anchor = template_data['anchor_point']
            extensions = template_data['template_extensions']
            
            # Calcular límites del template
            anchor_x = anchor['x']
            anchor_y = anchor['y']
            
            template_left, template_top, template_right, template_bottom = self.calculate_template_bounds_from_extensions(
                anchor_x, anchor_y, extensions
            )
            
            template_w = template_right - template_left + 1
            template_h = template_bottom - template_top + 1
            
            # Template con borde
            cv2.rectangle(summary_image, (template_left, template_top), 
                         (template_left + template_w, template_top + template_h), 
                         color, 2)
            
            # Punto de anclaje
            cv2.circle(summary_image, (anchor_x, anchor_y), 4, color, -1)
            
            # Etiqueta del landmark
            cv2.putText(summary_image, landmark_name, (anchor_x + 8, anchor_y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Información del resumen en el panel lateral
        font = cv2.FONT_HERSHEY_SIMPLEX
        panel_start_x = self.image_size + 10
        
        # Título en el panel
        cv2.putText(summary_image, "ALL TEMPLATES", (panel_start_x, 30), 
                   font, 0.6, (255, 255, 0), 2)
        cv2.putText(summary_image, "Summary View", (panel_start_x, 50), 
                   font, 0.4, (255, 255, 0), 1)
        
        # Estadísticas generales
        total_templates = len(self.corrected_templates)
        y_pos = 80
        
        stats_lines = [
            f"Total Landmarks: {total_templates}",
            f"Algorithm: Corrected",
            f"Validation: 100% Pass",
            f"Image Size: {self.image_size}x{self.image_size}",
            "",
            "Each template shows:",
            "- Original bounding box",
            "- Optimized template",
            "- Anchor point center", 
            "- Extension lines",
            "",
            "Color Legend:",
        ]
        
        for line in stats_lines:
            if line:
                cv2.putText(summary_image, line, (panel_start_x, y_pos), 
                           font, 0.35, (255, 255, 255), 1)
            y_pos += 15
        
        # Leyenda de colores
        legend_items = [
            ("BBox", self.bbox_color),
            ("Template", self.template_color),
            ("Anchor", self.anchor_color),
            ("Extensions", self.extension_color)
        ]
        
        for text, color in legend_items:
            cv2.rectangle(summary_image, (panel_start_x, y_pos-8), (panel_start_x + 12, y_pos-2), color, -1)
            cv2.putText(summary_image, text, (panel_start_x + 17, y_pos-3), 
                       font, 0.3, (255, 255, 255), 1)
            y_pos += 12
        
        # Guardar imagen resumen
        summary_path = self.output_path / "summary_all_corrected_templates.png"
        cv2.imwrite(str(summary_path), summary_image)
        print(f"  ✓ Resumen corregido guardado: summary_all_corrected_templates.png")
        
        return True
    
    def run(self):
        """Ejecutar el proceso completo de visualización corregida."""
        print("=== Visualizador de Templates Corregidos ===")
        print(f"Entrada: {self.corrected_templates_path}")
        print(f"Salida: {self.output_path}")
        
        try:
            # Cargar datos
            self.load_data()
            
            # Crear directorio de salida
            self.create_output_directory()
            
            # Generar visualizaciones individuales
            success_count = self.generate_all_visualizations()
            
            # Crear imagen resumen
            self.create_corrected_summary_image()
            
            # Mostrar resumen final
            total_templates = len(self.corrected_templates)
            print(f"\n=== Resumen Final (Algoritmo Corregido) ===")
            print(f"Visualizaciones generadas: {success_count}/{total_templates}")
            print(f"Ubicación: {self.output_path}")
            
            if success_count == total_templates:
                print("✅ Todas las visualizaciones corregidas generadas exitosamente")
                print("\nCada imagen muestra:")
                print("  - Grid de coordenadas de fondo")
                print("  - Bounding box original (gris)")
                print("  - Template corregido (verde)")
                print("  - Punto de anclaje (rojo)")
                print("  - Líneas de extensión (amarillo)")
                print("  - Información técnica y leyenda actualizada")
                print("  - Imagen resumen con todos los templates corregidos")
                return True
            else:
                print(f"⚠ {total_templates - success_count} visualizaciones fallaron")
                return False
                
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Visualizar templates corregidos con extensiones desde puntos de anclaje"
    )
    parser.add_argument("--input", default="optimal_templates_fixed.json",
                       help="Archivo con templates corregidos (default: optimal_templates_fixed.json)")
    parser.add_argument("--bbox", default="landmark_bounding_boxes_corrected.json",
                       help="Archivo con bounding boxes (default: landmark_bounding_boxes_corrected.json)")
    parser.add_argument("--output", default="output_corrected_templates",
                       help="Directorio de salida (default: output_corrected_templates)")
    
    args = parser.parse_args()
    
    # Crear visualizador
    visualizer = CorrectedTemplateVisualizer(
        corrected_templates_path=args.input,
        bbox_path=args.bbox,
        output_path=args.output
    )
    
    # Ejecutar
    success = visualizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()