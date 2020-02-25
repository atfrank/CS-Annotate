# CS-Annotate
 Chemical shift-based annotation of RNA structure

# Install DeepChem
git clone git@github.com:atfrank/CS-Annotate.git
cd CS-Annotate/

module load anaconda3/2019.10
module load gcc/9.2.0
conda create -n csannotate
source activate csannotate
apt-get -qq install -y python-rdkit librdkit1 rdkit-data
pip install -q joblib sklearn tensorflow pillow deepchem

# Get model
wget https://drive.google.com/open?id=1xSOtxGnZZohoRBpS8WssPjGG-lbL4Iad