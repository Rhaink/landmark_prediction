#!/usr/bin/env python3
"""
Visualizador de Resultados de Landmarks
======================================

Script para crear visualizaciones y análisis de los resultados guardados
del sistema de predicción multi-landmark.

Funcionalidades:
- Visualizar landmarks predichos vs ground truth en imágenes
- Generar mapas de calor de errores
- Crear gráficos estadísticos por landmark
- Análisis comparativo entre categorías médicas
- Exportar visualizaciones para publicaciones

Autor: Sistema de desarrollo con IA especializada
Fecha: 2025-09-10
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import cv2

# Importar predictor para cargar resultados
import sys
sys.path.append(".")
from all_landmark_prediction import AllLandmarkPredictor


class LandmarkVisualizer:
    """
    Clase para visualización y análisis de resultados de landmarks
    """
    
    def __init__(self, results_file: str = None):
        """
        Inicializar visualizador
        
        Args:
            results_file (str): Ruta al archivo de resultados guardados
        """
        self.predictor = None
        self.results_df = None
        
        if results_file:
            self.load_results(results_file)
    
    def load_results(self, results_file: str):
        """Cargar resultados desde archivo"""
        print(f"📂 Cargando resultados desde: {results_file}")
        
        file_path = Path(results_file)
        
        if file_path.suffix == '.csv':
            # Cargar directamente CSV
            self.results_df = pd.read_csv(results_file)
            print(f"✅ CSV cargado: {len(self.results_df)} predicciones")
            
        else:
            # Cargar usando el predictor
            self.predictor = AllLandmarkPredictor.load_results(results_file)
            self.results_df = self._convert_to_dataframe()
            print(f"✅ Resultados cargados: {len(self.results_df)} predicciones")
    
    def _convert_to_dataframe(self) -> pd.DataFrame:
        """Convertir predicciones a DataFrame para análisis"""
        rows = []
        
        for image_name, landmarks in self.predictor.predictions.items():
            for landmark_id, result in landmarks.items():
                if 'euclidean_error' in result:
                    row = {
                        'image_name': image_name,
                        'landmark': landmark_id,
                        'predicted_x': result['predicted_position'][0],
                        'predicted_y': result['predicted_position'][1],
                        'ground_truth_x': result['ground_truth'][0] if result['ground_truth'] else None,
                        'ground_truth_y': result['ground_truth'][1] if result['ground_truth'] else None,
                        'euclidean_error': result['euclidean_error'],
                        'category': result['category'],
                        'prediction_time': result.get('prediction_time', 0),
                        'best_score': result.get('best_score', 0),
                        'method': result.get('method', 'unknown')
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_landmarks_on_image(self, image_name: str, save_path: str = None, 
                               show_errors: bool = True, landmark_subset: List[str] = None):
        """
        Visualizar landmarks predichos vs ground truth en una imagen específica
        
        Args:
            image_name (str): Nombre de la imagen
            save_path (str): Ruta para guardar la visualización
            show_errors (bool): Mostrar líneas de error
            landmark_subset (List[str]): Subset de landmarks a mostrar
        """
        # Cargar imagen original
        image_path = self._find_image_path(image_name)
        if not image_path:
            print(f"❌ No se encontró imagen: {image_name}")
            return
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Filtrar datos para esta imagen
        image_data = self.results_df[self.results_df['image_name'] == image_name]
        
        if landmark_subset:
            image_data = image_data[image_data['landmark'].isin(landmark_subset)]
        
        # Crear visualización
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        
        # Colores para diferentes landmarks
        colors = plt.cm.tab20(np.linspace(0, 1, len(image_data)))
        
        for idx, (_, row) in enumerate(image_data.iterrows()):
            landmark = row['landmark']
            pred_x, pred_y = row['predicted_x'], row['predicted_y']
            gt_x, gt_y = row['ground_truth_x'], row['ground_truth_y']
            error = row['euclidean_error']
            
            color = colors[idx]
            
            # Landmark predicho (círculo)
            pred_circle = Circle((pred_x, pred_y), radius=3, 
                               color=color, fill=True, alpha=0.8, label=f'{landmark} Pred')
            ax.add_patch(pred_circle)
            
            # Ground truth (cruz)
            if not pd.isna(gt_x) and not pd.isna(gt_y):
                ax.plot(gt_x, gt_y, '+', color=color, markersize=8, markeredgewidth=2)
                
                # Línea de error
                if show_errors:
                    ax.plot([pred_x, gt_x], [pred_y, gt_y], '--', 
                           color=color, alpha=0.6, linewidth=1)
            
            # Etiqueta con error
            ax.annotate(f'{landmark}\n{error:.1f}px', 
                       (pred_x, pred_y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        ax.set_title(f'Landmarks: {image_name}\nCategoría: {image_data.iloc[0]["category"]}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)  # Invertir Y para imagen
        
        # Leyenda
        ax.legend(['Ground Truth (+)', 'Predicción (○)', 'Error (---)'], 
                 loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualización guardada: {save_path}")
        
        plt.show()
    
    def _find_image_path(self, image_name: str) -> Optional[Path]:
        """Encontrar ruta de imagen en el dataset"""
        dataset_dir = Path("data/dataset")
        
        # Buscar en subdirectorios
        for subdir in ["COVID", "Normal", "Viral_Pneumonia"]:
            potential_path = dataset_dir / subdir / f"{image_name}.png"
            if potential_path.exists():
                return potential_path
                
            # Buscar con variaciones de nombre
            potential_path = dataset_dir / subdir / image_name
            if potential_path.exists():
                return potential_path
        
        return None
    
    def plot_error_heatmap(self, save_path: str = None):
        """Crear mapa de calor de errores por landmark y categoría"""
        
        # Pivot table para heatmap
        error_pivot = self.results_df.pivot_table(
            values='euclidean_error', 
            index='landmark', 
            columns='category', 
            aggfunc='mean'
        )
        
        # Crear visualización
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        sns.heatmap(error_pivot, annot=True, fmt='.1f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Error Promedio (píxeles)'}, ax=ax)
        
        ax.set_title('Mapa de Calor: Error por Landmark y Categoría Médica', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Categoría Médica', fontsize=12)
        ax.set_ylabel('Landmark', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Mapa de calor guardado: {save_path}")
        
        plt.show()
    
    def plot_landmark_comparison(self, save_path: str = None):
        """Gráfico comparativo de precisión por landmark"""
        
        # Estadísticas por landmark
        landmark_stats = self.results_df.groupby('landmark')['euclidean_error'].agg([
            'mean', 'median', 'std', 'count'
        ]).reset_index()
        
        # Calcular tasa de éxito ≤10px
        success_rate = self.results_df.groupby('landmark').apply(
            lambda x: (x['euclidean_error'] <= 10).sum() / len(x) * 100
        ).reset_index(name='success_rate_10px')
        
        landmark_stats = landmark_stats.merge(success_rate, on='landmark')
        landmark_stats = landmark_stats.sort_values('mean')
        
        # Crear visualización
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Gráfico 1: Error promedio con barras de error
        bars = ax1.bar(landmark_stats['landmark'], landmark_stats['mean'], 
                      yerr=landmark_stats['std'], capsize=5, alpha=0.7,
                      color=plt.cm.viridis(np.linspace(0, 1, len(landmark_stats))))
        
        ax1.set_title('Error Promedio por Landmark', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Landmark', fontsize=12)
        ax1.set_ylabel('Error Euclidiano (píxeles)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, mean_val in zip(bars, landmark_stats['mean']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico 2: Tasa de éxito ≤10px
        bars2 = ax2.bar(landmark_stats['landmark'], landmark_stats['success_rate_10px'], 
                       alpha=0.7, color=plt.cm.plasma(np.linspace(0, 1, len(landmark_stats))))
        
        ax2.set_title('Tasa de Éxito (≤10px) por Landmark', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Landmark', fontsize=12)
        ax2.set_ylabel('Tasa de Éxito (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Añadir valores en las barras
        for bar, success_val in zip(bars2, landmark_stats['success_rate_10px']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{success_val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comparación de landmarks guardada: {save_path}")
        
        plt.show()
        
        return landmark_stats
    
    def plot_error_distribution(self, save_path: str = None):
        """Distribución de errores con histogramas y boxplots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histograma general
        axes[0,0].hist(self.results_df['euclidean_error'], bins=50, alpha=0.7, 
                      color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribución General de Errores', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Error Euclidiano (píxeles)')
        axes[0,0].set_ylabel('Frecuencia')
        axes[0,0].grid(True, alpha=0.3)
        
        # Boxplot por landmark
        landmark_order = self.results_df.groupby('landmark')['euclidean_error'].median().sort_values().index
        sns.boxplot(data=self.results_df, x='landmark', y='euclidean_error', 
                   order=landmark_order, ax=axes[0,1])
        axes[0,1].set_title('Distribución por Landmark', fontsize=12, fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Boxplot por categoría médica
        sns.boxplot(data=self.results_df, x='category', y='euclidean_error', ax=axes[1,0])
        axes[1,0].set_title('Distribución por Categoría Médica', fontsize=12, fontweight='bold')
        
        # Violin plot combinado
        sns.violinplot(data=self.results_df, x='category', y='euclidean_error', 
                      hue='landmark', ax=axes[1,1], legend=False)
        axes[1,1].set_title('Distribución Detallada', fontsize=12, fontweight='bold')
        axes[1,1].legend().remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Distribución de errores guardada: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, save_path: str = "landmark_analysis_report.txt"):
        """Generar reporte resumen de análisis"""
        
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE ANÁLISIS DE LANDMARKS")
        report.append("=" * 80)
        report.append(f"Fecha de generación: {pd.Timestamp.now()}")
        report.append(f"Total de predicciones: {len(self.results_df):,}")
        report.append("")
        
        # Estadísticas generales
        overall_stats = self.results_df['euclidean_error'].describe()
        report.append("📊 ESTADÍSTICAS GENERALES")
        report.append("-" * 30)
        report.append(f"Error promedio: {overall_stats['mean']:.2f} ± {overall_stats['std']:.2f} px")
        report.append(f"Error mediano: {overall_stats['50%']:.2f} px")
        report.append(f"Error mínimo: {overall_stats['min']:.2f} px")
        report.append(f"Error máximo: {overall_stats['max']:.2f} px")
        report.append("")
        
        # Tasas de éxito
        total_predictions = len(self.results_df)
        success_5px = (self.results_df['euclidean_error'] <= 5).sum()
        success_10px = (self.results_df['euclidean_error'] <= 10).sum()
        success_20px = (self.results_df['euclidean_error'] <= 20).sum()
        
        report.append("🎯 TASAS DE ÉXITO")
        report.append("-" * 20)
        report.append(f"≤ 5px:  {success_5px:4d}/{total_predictions} ({success_5px/total_predictions*100:.1f}%)")
        report.append(f"≤ 10px: {success_10px:4d}/{total_predictions} ({success_10px/total_predictions*100:.1f}%)")
        report.append(f"≤ 20px: {success_20px:4d}/{total_predictions} ({success_20px/total_predictions*100:.1f}%)")
        report.append("")
        
        # Top/Bottom landmarks
        landmark_stats = self.results_df.groupby('landmark')['euclidean_error'].mean().sort_values()
        
        report.append("🏆 TOP 5 LANDMARKS (Menor Error)")
        report.append("-" * 35)
        for i, (landmark, error) in enumerate(landmark_stats.head().items(), 1):
            report.append(f"{i}. {landmark}: {error:.2f} px")
        report.append("")
        
        report.append("⚠️  BOTTOM 5 LANDMARKS (Mayor Error)")
        report.append("-" * 37)
        for i, (landmark, error) in enumerate(landmark_stats.tail().items(), 1):
            report.append(f"{i}. {landmark}: {error:.2f} px")
        report.append("")
        
        # Por categoría médica
        category_stats = self.results_df.groupby('category')['euclidean_error'].mean().sort_values()
        report.append("🏥 RENDIMIENTO POR CATEGORÍA MÉDICA")
        report.append("-" * 40)
        for category, error in category_stats.items():
            count = (self.results_df['category'] == category).sum()
            report.append(f"{category}: {error:.2f} px (n={count})")
        report.append("")
        
        report.append("=" * 80)
        
        # Guardar reporte
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Reporte guardado: {save_path}")
        
        # Mostrar también en consola
        print('\n'.join(report))


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Visualizador de Resultados de Landmarks')
    
    parser.add_argument('--results', '-r', type=str, required=True,
                       help='Ruta al archivo de resultados (JSON, pickle, CSV)')
    parser.add_argument('--output-dir', '-o', type=str, default='visualizations',
                       help='Directorio para guardar visualizaciones')
    parser.add_argument('--image', '-i', type=str, 
                       help='Nombre de imagen específica para visualizar')
    parser.add_argument('--landmarks', '-l', nargs='+',
                       help='Subset de landmarks a visualizar')
    parser.add_argument('--format', '-f', choices=['png', 'pdf', 'svg'], 
                       default='png', help='Formato de imágenes')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Inicializar visualizador
    print("🎨 Iniciando visualizador de landmarks...")
    visualizer = LandmarkVisualizer(args.results)
    
    # Generar visualizaciones
    print("\n📊 Generando análisis estadísticos...")
    
    # 1. Mapa de calor
    visualizer.plot_error_heatmap(
        save_path=output_dir / f"error_heatmap.{args.format}"
    )
    
    # 2. Comparación de landmarks
    landmark_stats = visualizer.plot_landmark_comparison(
        save_path=output_dir / f"landmark_comparison.{args.format}"
    )
    
    # 3. Distribución de errores
    visualizer.plot_error_distribution(
        save_path=output_dir / f"error_distribution.{args.format}"
    )
    
    # 4. Imagen específica (si se proporciona)
    if args.image:
        print(f"\n🖼️  Visualizando imagen: {args.image}")
        visualizer.plot_landmarks_on_image(
            image_name=args.image,
            save_path=output_dir / f"landmarks_{args.image}.{args.format}",
            landmark_subset=args.landmarks
        )
    
    # 5. Reporte resumen
    print("\n📝 Generando reporte resumen...")
    visualizer.generate_summary_report(
        save_path=output_dir / "analysis_report.txt"
    )
    
    print(f"\n✅ Visualizaciones completadas en: {output_dir}")
    print(f"   - Archivos generados: {len(list(output_dir.glob('*')))} archivos")


if __name__ == "__main__":
    main()