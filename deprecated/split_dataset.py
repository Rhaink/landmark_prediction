#!/usr/bin/env python3
"""
Dataset Splitter para Landmark Prediction
Divide coordenadas_maestro.csv en conjuntos de entrenamiento, validación y prueba
con distribución estratificada 70%, 15%, 15%
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse


class DatasetSplitter:
    def __init__(self, csv_path, output_dir, random_state=42):
        """
        Inicializar el divisor de dataset
        
        Args:
            csv_path: Ruta al archivo CSV maestro
            output_dir: Directorio de salida para archivos divididos
            random_state: Semilla para reproducibilidad
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.data = None
        self.split_info = {}
        
    def load_data(self):
        """Cargar y validar datos del CSV"""
        print(f"Cargando datos de {self.csv_path}")
        
        # Cargar CSV sin header
        self.data = pd.read_csv(self.csv_path, header=None)
        
        # Validar estructura esperada
        expected_cols = 32  # índice + 30 coordenadas + filename
        if self.data.shape[1] != expected_cols:
            raise ValueError(f"CSV debe tener {expected_cols} columnas, encontradas {self.data.shape[1]}")
        
        # Extraer clases de los nombres de archivo (última columna)
        filenames = self.data.iloc[:, -1]
        classes = filenames.str.extract(r'(COVID|Normal|Viral)', expand=False)
        
        # Validar que todas las filas tienen clase válida
        invalid_classes = classes.isnull().sum()
        if invalid_classes > 0:
            raise ValueError(f"Encontradas {invalid_classes} filas sin clase válida")
        
        self.data['class'] = classes
        
        print(f"Datos cargados exitosamente: {len(self.data)} muestras")
        self.print_class_distribution(self.data, "Dataset original")
        
    def print_class_distribution(self, data, title):
        """Imprimir distribución de clases"""
        print(f"\n{title}:")
        class_counts = data['class'].value_counts().sort_index()
        total = len(data)
        
        for class_name, count in class_counts.items():
            percentage = (count / total) * 100
            print(f"  {class_name}: {count} muestras ({percentage:.1f}%)")
        print(f"  Total: {total} muestras")
        
    def split_data(self):
        """Dividir datos en train/val/test con estratificación"""
        print("\nDividiendo dataset...")
        
        # Separar features y labels
        X = self.data.drop('class', axis=1)
        y = self.data['class']
        
        # Primera división: 70% train, 30% temp (que será 15% val + 15% test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=0.3, 
            stratify=y, 
            random_state=self.random_state
        )
        
        # Segunda división: dividir temp en val (50%) y test (50%)
        # Esto da 15% cada uno del total original
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        # Recrear DataFrames con las clases
        self.train_data = X_train.copy()
        self.train_data['class'] = y_train
        
        self.val_data = X_val.copy()
        self.val_data['class'] = y_val
        
        self.test_data = X_test.copy()
        self.test_data['class'] = y_test
        
        # Mostrar distribuciones finales
        self.print_class_distribution(self.train_data, "Entrenamiento (70%)")
        self.print_class_distribution(self.val_data, "Validación (15%)")
        self.print_class_distribution(self.test_data, "Prueba (15%)")
        
    def validate_split(self):
        """Validar que la división es correcta"""
        print("\nValidando división...")
        
        # Verificar totales
        total_original = len(self.data)
        total_split = len(self.train_data) + len(self.val_data) + len(self.test_data)
        
        if total_original != total_split:
            raise ValueError(f"Total no coincide: original={total_original}, dividido={total_split}")
        
        # Verificar no hay duplicados entre conjuntos
        train_files = set(self.train_data.iloc[:, -2])  # columna filename (antes de 'class')
        val_files = set(self.val_data.iloc[:, -2])
        test_files = set(self.test_data.iloc[:, -2])
        
        overlaps = []
        if train_files & val_files:
            overlaps.append("train-val")
        if train_files & test_files:
            overlaps.append("train-test")
        if val_files & test_files:
            overlaps.append("val-test")
            
        if overlaps:
            raise ValueError(f"Encontrados archivos duplicados entre conjuntos: {overlaps}")
        
        print("✓ Validación exitosa: no hay duplicados y totales coinciden")
        
    def save_splits(self):
        """Guardar archivos CSV divididos"""
        print(f"\nGuardando archivos en {self.output_dir}")
        
        # Crear directorio de salida si no existe
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Remover columna 'class' temporal antes de guardar
        train_to_save = self.train_data.drop('class', axis=1)
        val_to_save = self.val_data.drop('class', axis=1)
        test_to_save = self.test_data.drop('class', axis=1)
        
        # Guardar CSVs sin header (manteniendo formato original)
        train_path = self.output_dir / "coordenadas_train.csv"
        val_path = self.output_dir / "coordenadas_val.csv"
        test_path = self.output_dir / "coordenadas_test.csv"
        
        train_to_save.to_csv(train_path, header=False, index=False)
        val_to_save.to_csv(val_path, header=False, index=False)
        test_to_save.to_csv(test_path, header=False, index=False)
        
        print(f"✓ Guardado: {train_path} ({len(train_to_save)} muestras)")
        print(f"✓ Guardado: {val_path} ({len(val_to_save)} muestras)")
        print(f"✓ Guardado: {test_path} ({len(test_to_save)} muestras)")
        
        return train_path, val_path, test_path
        
    def create_metadata(self, train_path, val_path, test_path):
        """Crear archivo de metadata con información de la división"""
        metadata = {
            "split_info": {
                "random_state": self.random_state,
                "split_ratios": {
                    "train": 0.7,
                    "validation": 0.15,
                    "test": 0.15
                },
                "created_at": datetime.now().isoformat(),
                "original_file": str(self.csv_path),
                "total_samples": len(self.data)
            },
            "files": {
                "train": {
                    "path": str(train_path),
                    "samples": len(self.train_data),
                    "class_distribution": self.train_data['class'].value_counts().to_dict()
                },
                "validation": {
                    "path": str(val_path),
                    "samples": len(self.val_data),
                    "class_distribution": self.val_data['class'].value_counts().to_dict()
                },
                "test": {
                    "path": str(test_path),
                    "samples": len(self.test_data),
                    "class_distribution": self.test_data['class'].value_counts().to_dict()
                }
            },
            "format": {
                "columns": 32,
                "structure": "index,x1,y1,...,x15,y15,filename",
                "landmarks": 15,
                "image_dimensions": "299x299"
            }
        }
        
        metadata_path = self.output_dir / "dataset_split_info.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        print(f"✓ Metadata guardada: {metadata_path}")
        return metadata_path
        
    def run_split(self):
        """Ejecutar todo el proceso de división"""
        try:
            self.load_data()
            self.split_data()
            self.validate_split()
            train_path, val_path, test_path = self.save_splits()
            metadata_path = self.create_metadata(train_path, val_path, test_path)
            
            print(f"\n🎉 División completada exitosamente!")
            print(f"📁 Archivos generados en: {self.output_dir}")
            print(f"🔑 Semilla utilizada: {self.random_state}")
            
            return {
                'train': train_path,
                'val': val_path,
                'test': test_path,
                'metadata': metadata_path
            }
            
        except Exception as e:
            print(f"❌ Error durante la división: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Dividir dataset de landmarks médicos')
    parser.add_argument(
        '--csv', 
        default='data/coordenadas/coordenadas_maestro.csv',
        help='Ruta al CSV maestro (default: data/coordenadas/coordenadas_maestro.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/coordenadas',
        help='Directorio de salida (default: data/coordenadas)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla aleatoria para reproducibilidad (default: 42)'
    )
    
    args = parser.parse_args()
    
    splitter = DatasetSplitter(args.csv, args.output, args.seed)
    results = splitter.run_split()
    
    return results


if __name__ == "__main__":
    main()