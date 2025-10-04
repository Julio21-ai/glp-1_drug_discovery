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
