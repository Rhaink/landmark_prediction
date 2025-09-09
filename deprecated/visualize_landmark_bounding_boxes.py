#!/usr/bin/env python3
"""
Script para generar visualizaciones sintéticas de las 15 bounding boxes individuales.
Genera 15 imágenes con fondo negro, cada una mostrando la región típica 
de un landmark específico basada en el análisis estadístico del dataset.
"""

import json
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
import colorsys


class LandmarkBoundingBoxVisualizer:
    def __init__(self, landmark_bbox_path, output_path):
        self.landmark_bbox_path = Path(landmark_bbox_path)
        self.output_path = Path(output_path)
        
        # Configuración de imagen simple
        self.image_size = 299  # 299x299 píxeles
        self.background_color = (0, 0, 0)  # Negro
        self.text_color = (255, 255, 255)  # Blanco
        
        # Generar 15 colores distintivos para las bounding boxes
        self.landmark_colors = self.generate_distinct_colors(15)
        
        # Datos cargados
        self.landmark_bboxes = None
        
    def generate_distinct_colors(self, n_colors):
        """Generar n colores distintivos usando el espacio HSV."""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            saturation = 0.9
            value = 0.8
            
            # Convertir HSV a RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convertir a formato BGR para OpenCV (0-255)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        return colors
    
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
        print(f"  - Algoritmo utilizado: {data['metadata']['algorithm']}")
        print(f"  - Método de bbox: {data['metadata']['bbox_method']}")
        
        return self.landmark_bboxes
    
    def draw_cartesian_plane(self, image):
        """
        Dibujar un plano cartesiano profesional con grid y ejes.
        
        Args:
            image: Imagen numpy donde dibujar el plano
        """
        # Colores profesionales
        grid_color = (30, 30, 30)      # Gris muy oscuro para grid tenue
        axis_color = (80, 80, 80)      # Gris medio para ejes principales
        text_color = (120, 120, 120)   # Gris claro para etiquetas
        
        # Dibujar grid cada 50 píxeles
        for i in range(0, self.image_size + 1, 50):
            # Líneas verticales (para eje X)
            cv2.line(image, (i, 0), (i, self.image_size), grid_color, 1)
            # Líneas horizontales (para eje Y) 
            cv2.line(image, (0, i), (self.image_size, i), grid_color, 1)
        
        # Dibujar ejes principales más gruesos
        # Eje Y (vertical) en x=0
        cv2.line(image, (0, 0), (0, self.image_size), axis_color, 2)
        # Eje X (horizontal) en y=299 (parte inferior)
        cv2.line(image, (0, self.image_size-1), (self.image_size, self.image_size-1), axis_color, 2)
        
        # Configuración de fuente para etiquetas
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.3
        font_thickness = 1
        
        # Etiquetas en eje X (parte inferior)
        for x in range(0, self.image_size + 1, 50):
            # Posición del texto
            text = str(x)
            text_size = cv2.getTextSize(text, font, font_size, font_thickness)[0]
            text_x = x - text_size[0] // 2
            text_y = self.image_size - 5
            
            # Solo dibujar si no se sale de la imagen
            if text_x >= 0 and text_x + text_size[0] <= self.image_size:
                cv2.putText(image, text, (text_x, text_y), 
                           font, font_size, text_color, font_thickness)
        
        # Etiquetas en eje Y (lado izquierdo)
        for y in range(0, self.image_size + 1, 50):
            # Invertir Y para que 0 esté arriba y 299 abajo
            display_y = y
            text = str(display_y)
            text_size = cv2.getTextSize(text, font, font_size, font_thickness)[0]
            text_x = 5
            text_y = y + text_size[1] // 2
            
            # Solo dibujar si no se sale de la imagen
            if text_y >= text_size[1] and text_y <= self.image_size:
                cv2.putText(image, text, (text_x, text_y), 
                           font, font_size, text_color, font_thickness)
        
        # Etiquetas de los ejes
        # Etiqueta "X" en esquina inferior derecha
        cv2.putText(image, "X", (self.image_size - 15, self.image_size - 8), 
                   font, 0.4, text_color, 1)
        # Etiqueta "Y" en esquina superior izquierda
        cv2.putText(image, "Y", (8, 12), 
                   font, 0.4, text_color, 1)
    
    def create_landmark_visualization(self, landmark_name, landmark_data, color):
        """
        Crear visualización limpia y simple para un landmark específico.
        
        Args:
            landmark_name: Nombre del landmark (ej: "L1", "L2", etc.)
            landmark_data: Datos de bounding box y estadísticas
            color: Color BGR para dibujar la bounding box
            
        Returns:
            numpy.ndarray: Imagen generada
        """
        # Crear imagen con fondo negro
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Dibujar plano cartesiano como capa de fondo
        self.draw_cartesian_plane(image)
        
        # Obtener datos de bounding box
        bbox = landmark_data['bbox']
        stats = landmark_data['statistics']
        
        # Coordenadas de la bounding box
        x = int(bbox['x'])
        y = int(bbox['y'])
        w = int(bbox['width'])
        h = int(bbox['height'])
        
        # Dibujar rectángulo de bounding box relleno (semi-transparente)
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        
        # Dibujar borde de la bounding box (más grueso para mejor visibilidad)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)
        
        # Añadir pequeña equis para marcar el centro
        center_x = int(bbox['center_x'])
        center_y = int(bbox['center_y'])
        # Dibujar equis pequeña (8px total: 4px desde centro en cada dirección)
        cv2.line(image, (center_x-4, center_y-4), (center_x+4, center_y+4), (255, 255, 255), 2)
        cv2.line(image, (center_x-4, center_y+4), (center_x+4, center_y-4), (255, 255, 255), 2)
        
        # Etiqueta simplificada en la esquina superior izquierda
        label_font = cv2.FONT_HERSHEY_DUPLEX
        label_size = 0.8
        label_thickness = 2
        
        # Posicionar etiqueta pequeña en esquina superior izquierda (evitando conflicto con eje Y)
        label_text = landmark_name  # Solo "L1", "L2", etc.
        cv2.putText(image, label_text, (25, 25), 
                   label_font, label_size, color, label_thickness)
        
        # Información mínima en la esquina superior derecha (fuera de área principal)
        info_font = cv2.FONT_HERSHEY_SIMPLEX
        info_size = 0.4
        info_thickness = 1
        
        # Solo mostrar información esencial y compacta
        info_lines = [
            f"{w}x{h}px",
            f"{int(bbox['area'])}px2"
        ]
        
        # Posicionar en esquina superior derecha
        for i, line in enumerate(info_lines):
            text_y = 15 + i * 15
            (text_w, text_h), _ = cv2.getTextSize(line, info_font, info_size, info_thickness)
            text_x = self.image_size - text_w - 10
            cv2.putText(image, line, (text_x, text_y), 
                       info_font, info_size, self.text_color, info_thickness)
        
        return image
    
    def create_output_directory(self):
        """Crear directorio de salida."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"Directorio de salida: {self.output_path}")
    
    def generate_all_visualizations(self):
        """Generar todas las 15 visualizaciones de landmarks."""
        print(f"\nGenerando 15 visualizaciones de landmark bounding boxes...")
        
        success_count = 0
        
        for i, (landmark_name, landmark_data) in enumerate(self.landmark_bboxes.items()):
            try:
                # Obtener color para este landmark
                color = self.landmark_colors[i]
                
                # Crear visualización
                image = self.create_landmark_visualization(landmark_name, landmark_data, color)
                
                # Guardar imagen
                output_filename = f"{landmark_name}.png"
                output_path = self.output_path / output_filename
                
                success = cv2.imwrite(str(output_path), image)
                
                if success:
                    print(f"  ✓ {landmark_name}: {output_filename}")
                    success_count += 1
                else:
                    print(f"  ✗ Error guardando {landmark_name}")
                    
            except Exception as e:
                print(f"  ✗ Error procesando {landmark_name}: {str(e)}")
        
        return success_count
    
    def run(self):
        """Ejecutar el proceso completo de generación de visualizaciones."""
        print("=== Generador de Visualizaciones de Landmark Bounding Boxes ===")
        print(f"Entrada: {self.landmark_bbox_path}")
        print(f"Salida: {self.output_path}")
        print(f"Tamaño de imagen: {self.image_size}x{self.image_size} px")
        
        try:
            # Cargar datos
            self.load_landmark_bboxes()
            
            # Crear directorio de salida
            self.create_output_directory()
            
            # Mostrar colores asignados
            print(f"\nColores asignados por landmark:")
            for i, (landmark_name, _) in enumerate(self.landmark_bboxes.items()):
                color = self.landmark_colors[i]
                print(f"  {landmark_name}: RGB{color}")
            
            # Generar visualizaciones
            success_count = self.generate_all_visualizations()
            
            # Mostrar resumen final
            total_landmarks = len(self.landmark_bboxes)
            print(f"\n=== Resumen Final ===")
            print(f"Visualizaciones generadas: {success_count}/{total_landmarks}")
            print(f"Ubicación: {self.output_path}")
            
            if success_count == total_landmarks:
                print("✓ Todas las visualizaciones generadas exitosamente")
                print("\nCada imagen muestra:")
                print("  - Fondo negro (299x299 px)")
                print("  - Rectángulo de bounding box en color distintivo")
                print("  - Pequeña equis marcando el centro")
                print("  - Etiqueta del landmark y dimensiones")
                return True
            else:
                print(f"⚠ {total_landmarks - success_count} visualizaciones fallaron")
                return False
                
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Generar 15 visualizaciones sintéticas de landmark bounding boxes"
    )
    parser.add_argument("--input", default="landmark_bounding_boxes.json",
                       help="Archivo con landmark bounding boxes (default: landmark_bounding_boxes.json)")
    parser.add_argument("--output", default="output_landmark_bbox_visualized",
                       help="Directorio de salida (default: output_landmark_bbox_visualized)")
    
    args = parser.parse_args()
    
    # Crear visualizador
    visualizer = LandmarkBoundingBoxVisualizer(
        landmark_bbox_path=args.input,
        output_path=args.output
    )
    
    # Ejecutar
    success = visualizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()