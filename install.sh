conda create -n livingscenes python=3.9 -y
conda activate livingscenes
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d=0.7.4 -c pytorch3d -y
conda install pyg -c pyg -y

pip install cython
cd lib_shape_prior
python setup.py build_ext --inplace

pip install -U python-pycg[all] -f https://pycg.huangjh.tech/packages/index.html
pip install -r requirements.txt
