import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def identity_heatmap(identity_matrix, 
                          title,
                          cmap='viridis',
                          figsize=(8, 8)):
    """
    Grafica la matriz de identidad usando un mapa de calor de Seaborn.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(identity_matrix, cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=12)
    plt.xlabel("Secuencia Index")
    plt.ylabel("Secuencia Index")
    plt.show()

def cumulative_variance_plot( cumulative_variance, figsize=(8, 5)):

    # Generar el gráfico
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='--', color='blue')

    plt.title('Varianza Acumulada Explicada por Componentes Principales', fontsize=16)
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Porcentaje de Varianza Acumulada')
    plt.grid(True)

    # Añadir líneas de referencia para facilitar la decisión
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% de Varianza')
    plt.axhline(y=0.90, color='green', linestyle='--', label='90% de Varianza')
    plt.legend()

    # Mostrar el gráfico en la salida de la celda
    plt.show()

def plot_pca_2d(pca_df, pc_x='PC1', pc_y='PC2', color_by=None, label='Values', figsize=(10, 8), 
                title='Visualización de 2 Componentes Principales (PCA)',
                cmap='viridis'):
    """
    Genera y muestra un gráfico de dispersión 2D de los Componentes Principales.

    Args:
        pca_df (pd.DataFrame): DataFrame que contiene los componentes principales (ej. PC1, PC2).
        pc_x (str): Nombre de la columna para el eje X (por defecto, 'PC1').
        pc_y (str): Nombre de la columna para el eje Y (por defecto, 'PC2').
        color_by (str, optional): Nombre de la columna para colorear los puntos. 
                                  Si es None, todos los puntos serán del mismo color.
        title (str): Título del gráfico.
        cmap (str): Mapa de colores a utilizar si se especifica `color_by`.
    """
    plt.figure(figsize=figsize)

    # Si se proporciona una columna para colorear, crea un gráfico de dispersión con color
    if color_by is not None:
        # Asegurarse de que la columna `color_by` exista en el DataFrame
        scatter = plt.scatter(
                    x=pca_df[pc_x],
                    y=pca_df[pc_y],
                    c=color_by,
                    cmap=cmap
                )
        plt.colorbar(scatter, label=label)
    else:
        plt.scatter(x=pca_df[pc_x], y=pca_df[pc_y], c='blue' )
    
    plt.xlabel(f'{pc_x}')
    plt.ylabel(f'{pc_y}')
    plt.title(title)
    plt.grid(True)
    plt.show()