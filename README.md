# CS-Annotate
 Chemical shift-based annotation of RNA structure

# Clone
git clone https://github.com/atfrank/CS-Annotate.git
cd CS-Annotate/

# Install Dependences
conda create -n csannotate python=3.7
conda activate csannotate
conda install -c conda-forge rdkit deepchem==2.3.0
pip install tensorflow==1.14.0

# Get pre-trained model
wget https://drive.google.com/open?id=1xSOtxGnZZohoRBpS8WssPjGG-lbL4Iad