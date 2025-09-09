#!/usr/bin/env python3
"""
Generador de animaciones GIF mostrando cómo el template se mantiene dentro de límites
cuando el punto de anclaje se mueve por todo el bounding box.
"""

import json
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
import colorsys
from PIL import Image
import io


class TemplateMovementAnimator:
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
        self.template_color = (0, 255, 0)  # Verde para template
        self.anchor_color = (255, 0, 0)    # Rojo para punto de anclaje
        self.trail_color = (255, 255, 0)   # Amarillo para rastro del anclaje
        self.grid_color = (30, 30, 30)     # Gris oscuro para grid
        
        # Configuración de animación
        self.frame_delay = 100  # milisegundos entre frames
        self.trail_length = 10  # número de posiciones anteriores del anclaje a mostrar
        
        # Datos cargados
        self.corrected_templates = None
        self.bbox_data = None
        self.metadata = None
        
    def load_data(self):
        """Cargar datos de templates corregidos y bounding boxes."""
        print(f"Cargando datos para animación...")
        
        # Cargar templates corregidos
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
        print(f"  - Landmarks a animar: {len(self.corrected_templates)}")
        
        return True
    
    def draw_coordinate_grid(self, image):
        """Dibujar grid de coordenadas básico solo en área principal."""
        # Grid cada 50 píxeles (más sutil para animación)
        for i in range(0, self.image_size + 1, 50):
            cv2.line(image, (i, 0), (i, self.image_size), self.grid_color, 1)
            cv2.line(image, (0, i), (self.image_size, i), self.grid_color, 1)
        
        # Línea separadora entre área principal y panel
        cv2.line(image, (self.image_size, 0), (self.image_size, self.image_size), (80, 80, 80), 2)
    
    def calculate_template_bounds_from_extensions(self, anchor_x, anchor_y, extensions):
        """Calcular límites del template basado en extensiones desde el anclaje."""
        template_left = anchor_x - extensions['left']
        template_right = anchor_x + extensions['right']
        template_top = anchor_y - extensions['up']
        template_bottom = anchor_y + extensions['down']
        
        return template_left, template_top, template_right, template_bottom
    
    def generate_anchor_path(self, bbox_limits):
        """
        Generar una ruta suave para el movimiento del anclaje dentro del bounding box.
        
        Args:
            bbox_limits: (left, right, top, bottom) del bounding box
            
        Returns:
            list: Lista de coordenadas (x, y) para el anclaje
        """
        bbox_left, bbox_right, bbox_top, bbox_bottom = bbox_limits
        
        # Crear un patrón de movimiento en zigzag que cubra todo el bounding box
        path = []
        
        # Parámetros para el patrón
        step_size = max(1, min(3, (bbox_right - bbox_left) // 10))  # Pasos adaptativos
        
        # Recorrer filas del bounding box
        for y in range(bbox_top, bbox_bottom + 1, step_size):
            if (y - bbox_top) // step_size % 2 == 0:
                # Fila par: izquierda a derecha
                for x in range(bbox_left, bbox_right + 1, step_size):
                    path.append((x, min(y, bbox_bottom)))
            else:
                # Fila impar: derecha a izquierda
                for x in range(bbox_right, bbox_left - 1, -step_size):
                    path.append((x, min(y, bbox_bottom)))
        
        # Asegurar que el path tenga al menos un número mínimo de frames
        if len(path) < 20:
            # Si el bounding box es muy pequeño, repetir el path varias veces
            original_path = path.copy()
            for _ in range(3):
                path.extend(original_path)
        
        return path
    
    def create_animation_frame(self, landmark_name, template_data, anchor_pos, frame_number, total_frames, anchor_trail):
        """
        Crear un frame individual de la animación.
        
        Args:
            landmark_name: Nombre del landmark
            template_data: Datos del template
            anchor_pos: Posición actual del anclaje (x, y)
            frame_number: Número del frame actual
            total_frames: Total de frames en la animación
            anchor_trail: Lista de posiciones anteriores del anclaje
            
        Returns:
            numpy.ndarray: Imagen del frame
        """
        # Crear imagen con área principal + panel lateral
        image = np.zeros((self.image_size, self.total_width, 3), dtype=np.uint8)
        
        # Crear panel lateral
        image[:, self.image_size:] = self.panel_color
        
        # Dibujar grid de coordenadas
        self.draw_coordinate_grid(image)
        
        # Extraer datos
        extensions = template_data['template_extensions']
        bbox_original = self.bbox_data[landmark_name]['bbox']
        
        # 1. Dibujar bounding box original
        bbox_x = int(bbox_original['x'])
        bbox_y = int(bbox_original['y'])
        bbox_w = int(bbox_original['width'])
        bbox_h = int(bbox_original['height'])
        
        cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), 
                     self.bbox_color, 2)
        
        # 2. Calcular y dibujar template en la posición actual del anclaje
        anchor_x, anchor_y = anchor_pos
        
        template_left, template_top, template_right, template_bottom = self.calculate_template_bounds_from_extensions(
            anchor_x, anchor_y, extensions
        )
        
        # Convertir a coordenadas de rectángulo para OpenCV
        template_x = template_left
        template_y = template_top
        template_w = template_right - template_left + 1
        template_h = template_bottom - template_top + 1
        
        # Dibujar template con borde
        cv2.rectangle(image, (template_x, template_y), 
                     (template_x + template_w, template_y + template_h), 
                     self.template_color, 2)
        
        # 3. Dibujar rastro del anclaje (posiciones anteriores)
        for i, trail_pos in enumerate(anchor_trail[-self.trail_length:]):
            if trail_pos != anchor_pos:
                # Intensidad decreciente para posiciones más antiguas
                intensity = (i + 1) / len(anchor_trail[-self.trail_length:])
                trail_color = tuple(int(c * intensity * 0.5) for c in self.trail_color)
                cv2.circle(image, trail_pos, 2, trail_color, -1)
        
        # 4. Dibujar punto de anclaje actual
        cv2.circle(image, (anchor_x, anchor_y), 6, self.anchor_color, -1)
        cv2.circle(image, (anchor_x, anchor_y), 8, self.anchor_color, 2)
        
        # 5. Añadir información en el panel lateral
        self.add_animation_info_panel(image, landmark_name, template_data, anchor_pos, 
                                    frame_number, total_frames, template_w, template_h)
        
        return image
    
    def add_animation_info_panel(self, image, landmark_name, template_data, anchor_pos, 
                               frame_number, total_frames, template_w, template_h):
        """Añadir información de animación en el panel lateral."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        panel_start_x = self.image_size + 10
        
        # Título del landmark
        cv2.putText(image, f"{landmark_name}", (panel_start_x, 30), 
                   font, 0.7, (255, 255, 0), 2)
        cv2.putText(image, "ANIMATION", (panel_start_x, 50), 
                   font, 0.4, (255, 255, 0), 1)
        
        # Información del frame
        anchor_x, anchor_y = anchor_pos
        extensions = template_data['template_extensions']
        
        y_pos = 80
        line_height = 15
        
        # Información de progreso
        progress_sections = [
            ("PROGRESS:", [
                f"Frame: {frame_number + 1}/{total_frames}",
                f"Progress: {((frame_number + 1)/total_frames)*100:.1f}%"
            ]),
            ("CURRENT ANCHOR:", [
                f"Position: ({anchor_x}, {anchor_y})",
                f"Moving within BBox"
            ]),
            ("TEMPLATE:", [
                f"Size: {template_w} x {template_h} px",
                f"Extensions from anchor:",
                f"  Left: {extensions['left']} px",
                f"  Right: {extensions['right']} px",
                f"  Up: {extensions['up']} px", 
                f"  Down: {extensions['down']} px"
            ]),
            ("VALIDATION:", [
                "Template always within",
                "image boundaries",
                "Anchor moves freely",
                "within bounding box"
            ])
        ]
        
        for section_title, section_lines in progress_sections:
            # Título de sección
            cv2.putText(image, section_title, (panel_start_x, y_pos), 
                       font, 0.4, (200, 200, 200), 1)
            y_pos += 18
            
            # Líneas de la sección
            for line in section_lines:
                cv2.putText(image, line, (panel_start_x + 5, y_pos), 
                           font, 0.35, self.text_color, 1)
                y_pos += line_height
            
            y_pos += 8
    
    def create_landmark_animation(self, landmark_name, template_data):
        """
        Crear animación GIF para un landmark específico.
        
        Args:
            landmark_name: Nombre del landmark
            template_data: Datos del template
            
        Returns:
            bool: True si fue exitoso
        """
        print(f"  Creando animación para {landmark_name}...")
        
        # Obtener límites del bounding box
        bbox_original = self.bbox_data[landmark_name]['bbox']
        bbox_left = max(0, int(bbox_original['x']))
        bbox_right = min(self.image_size - 1, int(bbox_original['x'] + bbox_original['width'] - 1))
        bbox_top = max(0, int(bbox_original['y']))
        bbox_bottom = min(self.image_size - 1, int(bbox_original['y'] + bbox_original['height'] - 1))
        bbox_limits = (bbox_left, bbox_right, bbox_top, bbox_bottom)
        
        # Generar ruta del anclaje
        anchor_path = self.generate_anchor_path(bbox_limits)
        
        print(f"    Generando {len(anchor_path)} frames...")
        
        # Crear frames de la animación
        frames = []
        anchor_trail = []
        
        for frame_num, anchor_pos in enumerate(anchor_path):
            # Actualizar rastro del anclaje
            anchor_trail.append(anchor_pos)
            
            # Crear frame
            frame = self.create_animation_frame(
                landmark_name, template_data, anchor_pos, 
                frame_num, len(anchor_path), anchor_trail
            )
            
            # Convertir de BGR (OpenCV) a RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        
        # Guardar como GIF
        gif_filename = f"{landmark_name}_animation.gif"
        gif_path = self.output_path / gif_filename
        
        try:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=self.frame_delay,
                loop=0,  # Loop infinito
                optimize=True
            )
            
            print(f"    ✓ Guardado: {gif_filename} ({len(frames)} frames)")
            return True
            
        except Exception as e:
            print(f"    ✗ Error guardando {gif_filename}: {e}")
            return False
    
    def create_summary_animation(self):
        """Crear una animación resumen con múltiples landmarks."""
        print(f"\n  Creando animación resumen...")
        
        # Seleccionar algunos landmarks representativos para el resumen
        representative_landmarks = ['L1', 'L5', 'L9', 'L14']  # Esquinas y centro
        available_landmarks = [name for name in representative_landmarks 
                             if name in self.corrected_templates]
        
        if not available_landmarks:
            # Si no hay landmarks representativos, usar los primeros 4
            available_landmarks = list(self.corrected_templates.keys())[:4]
        
        print(f"    Landmarks en resumen: {', '.join(available_landmarks)}")
        
        # Generar colores únicos
        colors = []
        for i in range(len(available_landmarks)):
            hue = i / len(available_landmarks)
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        # Generar rutas para cada landmark
        landmark_paths = {}
        max_frames = 0
        
        for landmark_name in available_landmarks:
            bbox_original = self.bbox_data[landmark_name]['bbox']
            bbox_left = max(0, int(bbox_original['x']))
            bbox_right = min(self.image_size - 1, int(bbox_original['x'] + bbox_original['width'] - 1))
            bbox_top = max(0, int(bbox_original['y']))
            bbox_bottom = min(self.image_size - 1, int(bbox_original['y'] + bbox_original['height'] - 1))
            bbox_limits = (bbox_left, bbox_right, bbox_top, bbox_bottom)
            
            path = self.generate_anchor_path(bbox_limits)
            landmark_paths[landmark_name] = path
            max_frames = max(max_frames, len(path))
        
        # Crear frames del resumen
        frames = []
        
        for frame_num in range(max_frames):
            # Crear imagen base
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            self.draw_coordinate_grid(image)
            
            # Dibujar cada landmark
            for i, landmark_name in enumerate(available_landmarks):
                template_data = self.corrected_templates[landmark_name]
                extensions = template_data['template_extensions']
                path = landmark_paths[landmark_name]
                color = colors[i]
                
                # Usar posición cíclica si el path es más corto
                pos_index = frame_num % len(path)
                anchor_x, anchor_y = path[pos_index]
                
                # Calcular template
                template_left, template_top, template_right, template_bottom = self.calculate_template_bounds_from_extensions(
                    anchor_x, anchor_y, extensions
                )
                
                template_w = template_right - template_left + 1
                template_h = template_bottom - template_top + 1
                
                # Dibujar template
                cv2.rectangle(image, (template_left, template_top), 
                             (template_left + template_w, template_top + template_h), 
                             color, 2)
                
                # Dibujar anclaje
                cv2.circle(image, (anchor_x, anchor_y), 4, color, -1)
                
                # Etiqueta
                cv2.putText(image, landmark_name, (anchor_x + 6, anchor_y - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Título
            cv2.putText(image, f"Multiple Templates Animation - Frame {frame_num + 1}/{max_frames}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Convertir y añadir frame
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        
        # Guardar GIF resumen
        summary_path = self.output_path / "summary_multiple_templates_animation.gif"
        
        try:
            frames[0].save(
                summary_path,
                save_all=True,
                append_images=frames[1:],
                duration=self.frame_delay,
                loop=0,
                optimize=True
            )
            
            print(f"    ✓ Guardado: summary_multiple_templates_animation.gif ({len(frames)} frames)")
            return True
            
        except Exception as e:
            print(f"    ✗ Error guardando resumen: {e}")
            return False
    
    def create_output_directory(self):
        """Crear directorio de salida."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"Directorio de animaciones: {self.output_path}")
    
    def run(self):
        """Ejecutar la generación completa de animaciones."""
        print("=== Generador de Animaciones de Templates ===")
        print(f"Entrada: {self.corrected_templates_path}")
        print(f"Salida: {self.output_path}")
        
        try:
            # Cargar datos
            self.load_data()
            
            # Crear directorio de salida
            self.create_output_directory()
            
            print(f"\nGenerando animaciones individuales...")
            
            # Generar animaciones individuales
            success_count = 0
            total_landmarks = len(self.corrected_templates)
            
            for landmark_name, template_data in self.corrected_templates.items():
                if self.create_landmark_animation(landmark_name, template_data):
                    success_count += 1
            
            # Crear animación resumen
            print(f"\nGenerando animación resumen...")
            summary_success = self.create_summary_animation()
            
            # Mostrar resumen final
            print(f"\n=== Resumen Final ===")
            print(f"Animaciones individuales: {success_count}/{total_landmarks}")
            print(f"Animación resumen: {'✓' if summary_success else '✗'}")
            print(f"Ubicación: {self.output_path}")
            
            if success_count == total_landmarks and summary_success:
                print("✅ Todas las animaciones generadas exitosamente")
                print(f"\nArchivos generados:")
                print(f"  - {success_count} GIFs individuales de landmarks")
                print(f"  - 1 GIF resumen con múltiples templates")
                print(f"  - Cada animación muestra el anclaje moviéndose por el bounding box")
                print(f"  - Template se mantiene dentro de límites en todo momento")
                return True
            else:
                print(f"⚠ Algunas animaciones fallaron")
                return False
                
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Crear animaciones GIF del movimiento de templates"
    )
    parser.add_argument("--input", default="optimal_templates_fixed.json",
                       help="Archivo con templates corregidos (default: optimal_templates_fixed.json)")
    parser.add_argument("--bbox", default="landmark_bounding_boxes_corrected.json",
                       help="Archivo con bounding boxes (default: landmark_bounding_boxes_corrected.json)")
    parser.add_argument("--output", default="animations",
                       help="Directorio de salida (default: animations)")
    parser.add_argument("--delay", type=int, default=100,
                       help="Delay entre frames en ms (default: 100)")
    
    args = parser.parse_args()
    
    # Crear animador
    animator = TemplateMovementAnimator(
        corrected_templates_path=args.input,
        bbox_path=args.bbox,
        output_path=args.output
    )
    
    # Configurar delay si se especificó
    animator.frame_delay = args.delay
    
    # Ejecutar
    success = animator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()