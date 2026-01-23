# GLP-1 Project Environment Setup

Este documento detalla las instrucciones para configurar el entorno de desarrollo del proyecto, incluyendo herramientas de bioinformática, y modelos de Machine Learning.

Las instrucciones están optimizadas para **WSL (Windows Subsystem for Linux)**, integrando compilación de herramientas externas, entornos Conda y aceleración por GPU.

##  Prerrequisitos

* **WSL 2** (Ubuntu/Debian recomendado) instalado en Windows.
* **Anaconda** o **Miniconda**
* **Drivers de NVIDIA** actualizados en Windows

---

##  Guía de Instalación


### 1. Dependencias del Sistema y CD-HIT (WSL)

https://github.com/weizhongli/cdhit


La herramienta `CD-HIT` debe compilarse desde el código fuente y requiere librerías base del sistema.
Desde la linea de comandos de WSL ejecutar:

```bash
# 1.1 Actualizar e instalar compilador y librerías base 
sudo apt-get update
sudo apt-get install build-essential zlib1g-dev git -y

# 1.2 Clonar y compilar CD-HIT
cd ~
git clone [https://github.com/weizhongli/cdhit.git](https://github.com/weizhongli/cdhit.git)
cd cdhit
make

# 1.3 Instalar  (para usar 'cd-hit' globalmente)
sudo make install
```

2. Creación del Entorno Virtual
Entorno aislado con Python 3.11:
```Bash
conda create --name GLP-1 python=3.11
conda activate GLP-1
```
3. Instalación de Paquetes Conda
Instalación de librerías científicas, herramientas bioinformáticas y soporte CUDA.

```Bash
# Herramientas base de análisis y visualización
conda install networkx sklearn seaborn -y

# Bioinformática y Química Computacional
conda install biopython -y
conda install -c rdkit rdkit -y
conda install -c salilab dssp -y
conda install -c bioconda msms -y

# Aceleración por GPU (CUDA Runtime 12.8)
conda install nvidia/label/cuda-12.8.1::cuda-runtime
```

4. Instalación de Paquetes Pip
Instalación de librerías para Machine Learning, MLOps y utilidades específicas.

```Bash
# Herramientas CLI y utilidades de secuencias
pip install iFeatureOmegaCLI
pip install logomaker
pip install tqdm ipywidgets sentencepiece

# Optimización Bayesiana y Ray Tune
pip install "bayesian-optimization"
pip install ray "ray[tune]"

# Machine Learning & Deep Learning (Transformers, PyCaret, Scikit-learn)
pip install git+[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
pip install datasets accelerate evaluate
pip install scikit-learn
pip install "pycaret[models,tuners,parallel,analysis]"

# MLOps y Monitoreo
pip install wandb
pip install tensorboard

# Kernel para Jupyter Notebooks
pip install ipykernel
```
5. Configuración Manual: iFeatureOmega
El instalador automático de iFeatureOmega presenta conflictos de rutas en versiones recientes de Python. 

Clonar la biblioteca CLI: 
```
git clone https://github.com/Superzchen/iFeatureOmega-CLI
```

Copiar la carpeta entera dentro del directorio site-packages del entorno conda activo.

```Bash
cp -r iFeatureOmega-CLI $CONDA_PREFIX/lib/python3.11/site-packages/
```

6. Verificación y Uso
Para confirmar que la instalación fue exitosa:

Registrar el Kernel en Jupyter:

```Bash
python -m ipykernel install --user --name=GLP-1 --display-name "Python (GLP-1)"
```
Verificar CD-HIT:

```Bash
cd-hit --help
(Debería mostrar las opciones de ayuda).
```
Verificar GPU (PyTorch/CUDA):

```Bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```