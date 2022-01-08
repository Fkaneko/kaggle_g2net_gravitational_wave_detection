
conda install mamba -n base -c conda-forge
# basic install
mamba install \
    numpy \
    pandas \
    matplotlib \
    scipy \
    seaborn \
    scikit-learn

mamba install -c plotly plotly=4.14.3

# for local installation, not for ngc pytorch container
mamba install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

pip install hydra-core --upgrade

pip install \
    kaggle \
    albumentations \
    timm==0.4.12 \
    segmentation-models-pytorch==0.2.0 \
    pytorch-lightning \
    kornia==0.5.10

# neptune-client[pytorch-lightning] \

# project specific
pip install \
    nnAudio==0.2.6\
    librosa

mamba install \
    h5py

# pip install spyder==5.1.3
# # jupyterlab installation
# mamba install -c conda-forge 'jupyterlab>=3.0.0,<4.0.0a0' jupyterlab-lsp
# mamba install -c conda-forge python-lsp-server jupytext jupyterlab_code_formatter
