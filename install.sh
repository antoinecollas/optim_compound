# create conda environment
conda create -n optim_compound python=3.9 --yes
eval "$(conda shell.bash hook)"  # bug fix: https://github.com/conda/conda/issues/7980#issuecomment-492784093 
conda activate optim_compound

# install libraries
pip install -r requirements.txt

# install pymanopt separately
# because of https://github.com/pymanopt/pymanopt/issues/161
pip install git+https://github.com/antoinecollas/pymanopt@master
pip install git+https://github.com/antoinecollas/pyCovariance@master
