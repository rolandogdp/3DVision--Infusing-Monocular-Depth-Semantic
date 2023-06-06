#run with source "source load_module.sh"
#module load new

env2lmod
module load gcc/8.2.0
module load python/3.9.9
module load cuda/11.3.1
module load eth_proxy

#module load pytorch 
#module load h5py/3.8.0
#export PATH=$PATH:"$HOME/.local/bin"
#source $HOME/.local/bin/virtualenvwrapper.sh
#workon cil-env


pip install Pillow
pip install h5py==3.8.0
pip install torch==2.0.0
pip install torchvision==0.15.1
pip install numpy
pip install pandas
pip install scipy
pip install matplotlib  
pip install scikit-learn

export THREED_VISION_ABSOLUTE_DOWNLOAD_PATH=/cluster/project/infk/courses/252-0579-00L/group37/downloads/ 

