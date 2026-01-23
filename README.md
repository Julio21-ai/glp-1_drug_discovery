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
