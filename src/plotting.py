import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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

def plot_pca_3d(
    pca_df,
    pc_x='PC1',
    pc_y='PC2',
    pc_z='PC3',
    color_by=None,
    label='Values',
    figsize=(10, 8),
    title='Visualización de 3 Componentes Principales (PCA)',
    cmap='viridis',
    xlim=None,
    ylim=None,
    zlim=None,
    ax=None 
):
    """
    Genera y muestra (o agrega a) un gráfico de dispersión 3D de los Componentes Principales.

    Args:
        pca_df (pd.DataFrame): DataFrame con columnas de componentes principales.
        pc_x, pc_y, pc_z (str): Nombres de las columnas para los ejes X, Y, Z.
        color_by (str, optional): Columna para colorear los puntos.
        label (str): Etiqueta de la barra de color.
        figsize (tuple): Tamaño del gráfico (solo se usa si no se pasa 'ax').
        title (str): Título del gráfico.
        cmap (str): Mapa de colores.
        xlim, ylim, zlim (tuple, optional): Límites de los ejes (min, max).
        ax (Axes3D, optional): Eje 3D existente donde dibujar el gráfico.
    """
    # Crear figura y eje solo si no se proporcionan
    own_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        own_fig = True
    else:
        fig = ax.get_figure()

    # Extraer coordenadas
    x = pca_df[pc_x]
    y = pca_df[pc_y]
    z = pca_df[pc_z]

    # Colorear por variable si se especifica
    if color_by is not None:
        scatter = ax.scatter(x, y, z, c=pca_df[color_by], cmap=cmap)
        fig.colorbar(scatter, ax=ax, label=label)
    else:
        ax.scatter(x, y, z, c='blue')

    # Etiquetas y título
    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    ax.set_zlabel(pc_z)
    ax.set_title(title)

    # Escalas fijas si se proporcionan
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if zlim: ax.set_zlim(zlim)

    # Mostrar solo si la figura pertenece a esta función
    if own_fig:
        plt.tight_layout()
        plt.show()


def clasificar_y_graficar_cuantiles(
    df_input, 
    columna_valor, 
    columna_id=None, 
    cuantiles=(0.33, 0.66), 
    categorias=("Baja", "Media", "Alta"),
    top_n=15,
    palette='Set2',  # Puede ser dict o string de paleta estándar
    titulo_distribucion=None,  # Título del histograma
    titulo_ordenado=None,      # Título del gráfico ordenado
    incluir_grafico_ordenado=True,  # Mostrar o no el scatter ordenado
    resumen=True,
    mostrar_top=True
):
    """
    Clasifica una columna numérica basada en cuantiles, genera gráficos 
    (opcionalmente uno o dos) y muestra resumen estadístico.

    Parámetros nuevos
    -----------------
    titulo_distribucion : str o None
        Título del histograma. Si None, se genera uno automáticamente.
    titulo_ordenado : str o None
        Título del gráfico ordenado. Si None, se genera automáticamente.
    incluir_grafico_ordenado : bool
        Si True, se muestra el scatter de valores ordenados.
    """

    assert columna_valor in df_input.columns, f"El DataFrame debe contener la columna '{columna_valor}'"
    assert len(categorias) == 3, "Actualmente esta función está configurada para 3 categorías."

    df = df_input.copy()

    # Convertir paleta si es string a un dict mapeado a las categorías
    if isinstance(palette, str):
        colores = sns.color_palette(palette, n_colors=3)
        palette = {cat: col for cat, col in zip(categorias, colores)}

    # Calcular cuantiles
    q_low, q_high = df[columna_valor].quantile(list(cuantiles))

    # Crear función de clasificación
    def clasificar(valor):
        if valor < q_low:
            return categorias[0]
        elif valor < q_high:
            return categorias[1]
        else:
            return categorias[2]

    # Asignar categoría
    df['Potencia'] = df[columna_valor].apply(clasificar)

    # Definir títulos automáticos si no se proporcionan
    if titulo_distribucion is None:
        titulo_distribucion = f'Distribución de {columna_valor} por Categoría'
    if titulo_ordenado is None:
        titulo_ordenado = f'{columna_valor} Ordenado con Clasificación'

    # ----- GRAFICOS -----
    n_cols = 2 if incluir_grafico_ordenado else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8*n_cols, 5))

    # Si hay solo un gráfico, seaborn necesita axis como single obj
    if n_cols == 1:
        axes = [axes]

    # Histograma
    sns.histplot(data=df, x=columna_valor, hue='Potencia', palette=palette, bins=25, alpha=0.8, kde=True, ax=axes[0])
    axes[0].axvline(q_low, color='black', linestyle='--', linewidth=1, label=f'Q{int(cuantiles[0]*100)} = {q_low:.2f}')
    axes[0].axvline(q_high, color='black', linestyle='--', linewidth=1, label=f'Q{int(cuantiles[1]*100)} = {q_high:.2f}')
    axes[0].set_title(titulo_distribucion)
    axes[0].set_xlabel(columna_valor)
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()

    # Scatter ordenado
    if incluir_grafico_ordenado:
        ordenado = df.sort_values(columna_valor).reset_index(drop=True)
        ordenado['Orden'] = range(1, len(ordenado) + 1)
        sns.scatterplot(data=ordenado, x='Orden', y=columna_valor, hue='Potencia', palette=palette, s=60, ax=axes[1])
        axes[1].set_title(titulo_ordenado)
        axes[1].set_xlabel(f'Orden (de menor a mayor {columna_valor})')
        axes[1].set_ylabel(columna_valor)
        axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    # ----- RESUMEN -----
    if resumen:
        resumen = df.groupby('Potencia')[columna_valor].agg(['count', 'mean', 'min', 'max']).rename(columns={'count':'n','mean':'promedio'})
        print('\nResumen por categoría:')
        print(resumen)

    # ----- TOP N -----
    if mostrar_top and columna_id is not None:
        top_cat_max = categorias[-1]  # última categoría es la más alta
        top = df[df['Potencia']==top_cat_max].sort_values(columna_valor, ascending=False).head(top_n)
        print(f'\nTop {top_n} registros de categoría "{top_cat_max}":')
        display(top[[columna_id, columna_valor, 'Potencia']])

