#!/usr/bin/env python3
"""
Script para visualizar coordenadas de landmarks en imágenes médicas de 299x299 píxeles.
Procesamiento paralelo y eficiente de todo el dataset.
"""

import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import argparse
import sys

class CoordinateVisualizer:
    def __init__(self, csv_path, dataset_path, output_path, num_processes=None):
        self.csv_path = Path(csv_path)
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.num_processes = num_processes or mp.cpu_count()
        
        # Color verde para todos los landmarks
        self.landmark_color = (0, 255, 0)  # Verde
        
    def load_coordinates(self):
        """Cargar y parsear el archivo CSV de coordenadas."""
        print(f"Cargando coordenadas desde: {self.csv_path}")
        
        # Leer CSV sin header
        df = pd.read_csv(self.csv_path, header=None)
        
        coordinates_data = []
        for _, row in df.iterrows():
            values = row.values
            
            # La última columna es el nombre del archivo
            filename = values[-1]
            
            # Todas las columnas anteriores son coordenadas (x,y pares)
            coords = values[1:-1]  # Excluir el índice (primera columna) y filename (última)
            
            # Convertir a pares x,y
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x, y = int(coords[i]), int(coords[i + 1])
                    points.append((x, y))
            
            coordinates_data.append({
                'filename': filename,
                'points': points
            })
        
        print(f"Cargadas {len(coordinates_data)} entradas de coordenadas")
        return coordinates_data
    
    def find_image_path(self, filename):
        """Encontrar la ruta completa de una imagen en el dataset."""
        # Agregar extensión .png si no la tiene
        if not filename.endswith('.png'):
            filename = filename + '.png'
        
        # Buscar en las tres carpetas
        for category in ['COVID', 'Normal', 'Viral_Pneumonia']:
            # Manejar el caso especial de Viral_Pneumonia
            if category == 'Viral_Pneumonia':
                # Buscar tanto con espacio como con guión bajo
                image_path = self.dataset_path / category / filename
                if image_path.exists():
                    return image_path, category
                
                # Intentar con espacio en lugar de guión bajo
                if filename.startswith('Viral_Pneumonia'):
                    alt_filename = filename.replace('Viral_Pneumonia', 'Viral Pneumonia')
                    image_path = self.dataset_path / category / alt_filename
                    if image_path.exists():
                        return image_path, category
                elif filename.startswith('Viral Pneumonia'):
                    alt_filename = filename.replace('Viral Pneumonia', 'Viral_Pneumonia')
                    image_path = self.dataset_path / category / alt_filename
                    if image_path.exists():
                        return image_path, category
            else:
                image_path = self.dataset_path / category / filename
                if image_path.exists():
                    return image_path, category
        
        return None, None
    
    def visualize_single_image(self, coord_data):
        """Visualizar coordenadas en una sola imagen."""
        filename = coord_data['filename']
        points = coord_data['points']
        
        # Encontrar la imagen
        image_path, category = self.find_image_path(filename)
        if image_path is None:
            return False, f"Imagen no encontrada: {filename}"
        
        try:
            # Cargar imagen
            image = cv2.imread(str(image_path))
            if image is None:
                return False, f"Error al cargar imagen: {filename}"
            
            # Verificar dimensiones
            height, width = image.shape[:2]
            if height != 299 or width != 299:
                return False, f"Dimensiones incorrectas {width}x{height} para {filename}"
            
            # Dibujar puntos con etiquetas numéricas
            for i, (x, y) in enumerate(points):
                # Validar coordenadas
                if 0 <= x < width and 0 <= y < height:
                    # Dibujar círculo verde
                    cv2.circle(image, (x, y), 3, self.landmark_color, -1)
                    cv2.circle(image, (x, y), 5, self.landmark_color, 1)
                    
                    # Agregar etiqueta numérica L1, L2, etc.
                    label = f"L{i+1}"
                    # Posicionar texto ligeramente arriba y a la derecha del punto
                    text_x = x + 8
                    text_y = y - 8
                    
                    # Asegurar que el texto esté dentro de los límites de la imagen
                    if text_x + 20 > width:
                        text_x = x - 25
                    if text_y - 10 < 0:
                        text_y = y + 15
                    
                    cv2.putText(image, label, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.landmark_color, 1)
            
            # Crear directorio de salida si no existe
            output_dir = self.output_path / category
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar imagen procesada
            output_path = output_dir / filename
            
            # Asegurar que el archivo tenga extensión .png
            if not str(output_path).endswith('.png'):
                output_path = output_path.with_suffix('.png')
            
            success = cv2.imwrite(str(output_path), image)
            if not success:
                return False, f"Error al guardar imagen: {filename} -> {output_path}"
            
            return True, f"Procesado: {filename}"
            
        except Exception as e:
            return False, f"Error procesando {filename}: {str(e)}"
    
    def create_output_structure(self):
        """Crear estructura de directorios de salida."""
        print(f"Creando estructura de salida en: {self.output_path}")
        
        categories = ['COVID', 'Normal', 'Viral_Pneumonia']
        for category in categories:
            (self.output_path / category).mkdir(parents=True, exist_ok=True)
    
    def process_images(self, coordinates_data):
        """Procesar todas las imágenes usando multiprocesamiento."""
        print(f"Procesando {len(coordinates_data)} imágenes con {self.num_processes} procesos")
        
        # Crear pool de procesos
        with mp.Pool(processes=self.num_processes) as pool:
            # Procesar con barra de progreso
            results = list(tqdm(
                pool.imap(self.visualize_single_image, coordinates_data),
                total=len(coordinates_data),
                desc="Procesando imágenes"
            ))
        
        # Analizar resultados
        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        
        print(f"\nResultados:")
        print(f"✓ Exitosos: {successful}")
        print(f"✗ Fallidos: {failed}")
        
        # Mostrar errores
        if failed > 0:
            print("\nErrores:")
            for success, message in results:
                if not success:
                    print(f"  - {message}")
    
    def run(self):
        """Ejecutar el proceso completo de visualización."""
        print("=== Visualizador de Coordenadas ===")
        print(f"Dataset: {self.dataset_path}")
        print(f"CSV: {self.csv_path}")
        print(f"Output: {self.output_path}")
        print(f"Procesos: {self.num_processes}")
        
        # Validar archivos de entrada
        if not self.csv_path.exists():
            print(f"ERROR: Archivo CSV no encontrado: {self.csv_path}")
            return False
        
        if not self.dataset_path.exists():
            print(f"ERROR: Directorio de dataset no encontrado: {self.dataset_path}")
            return False
        
        # Cargar coordenadas
        coordinates_data = self.load_coordinates()
        
        # Crear estructura de salida
        self.create_output_structure()
        
        # Procesar imágenes
        self.process_images(coordinates_data)
        
        print("\n=== Procesamiento Completado ===")
        return True

def main():
    parser = argparse.ArgumentParser(description="Visualizar coordenadas en imágenes médicas")
    parser.add_argument("--csv", default="data/coordenadas/coordenadas_maestro.csv", 
                       help="Ruta al archivo CSV de coordenadas")
    parser.add_argument("--dataset", default="data/dataset", 
                       help="Ruta al directorio del dataset")
    parser.add_argument("--output", default="output_visualized", 
                       help="Directorio de salida para imágenes procesadas")
    parser.add_argument("--processes", type=int, default=None,
                       help="Número de procesos paralelos (por defecto: CPU count)")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout en segundos (por defecto: 3600s)")
    
    args = parser.parse_args()
    
    # Configurar timeout muy alto
    import signal
    signal.alarm(args.timeout)
    
    # Crear visualizador
    visualizer = CoordinateVisualizer(
        csv_path=args.csv,
        dataset_path=args.dataset,
        output_path=args.output,
        num_processes=args.processes
    )
    
    # Ejecutar
    success = visualizer.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()