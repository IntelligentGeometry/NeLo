
############################################################
# Install requirements for the project
# 
# * Please check the CUDA version of your machine and change
# * the pytorch-cuda version in the following accordingly.


# install pyg-related packages
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
conda install pytorch-scatter -c pyg
conda install pytorch-sparse -c pyg
conda install pytorch-cluster -c pyg
conda install pytorch-spline-conv -c pyg

# install pytorch3d
#conda install pytorch3d -c pytorch3d

# install others
pip install -r requirements.txt