# CS-Annotate
 Chemical Shift-Based Annotation of RNA Structure

# Clone
```
git clone https://github.com/atfrank/CS-Annotate.git
cd CS-Annotate/
```

# Install Dependences
```
conda create -n csannotate python=3.7
conda activate csannotate
conda install -c conda-forge rdkit deepchem==2.3.0
pip install tensorflow==1.14.0
```

# Deploy pre-trained model
```
python deploy.py -f data/test_deploy.csv
```

# Caveat
* The classifier expects that missing chemical shifts have been imputed. In the example above ``data/test_deploy.csv`` corresponded to chemical shift dataset in which all missing values were imputed
* Missing chemical shifts can imputed using:
 * [CS2BPS](https://github.com/atfrank/CS2Structure/tree/master/cs2bps).
 * CS-Impute, which can be access via the Science Gateway, [SMALTR](http://smaltr.org/).

# CS-Annotate via SMALTR
* CS-Annotate can also be accessed via: [SMALTR](http://smaltr.org/).
* Given input chemical shifts, the web-app first imputes missing chemical shift values and then feds the imputed data to the classifier to annotate structure.
* A video demonstrating how to run CS-Annotate via SMALTR can be accessed using [this](https://youtu.be/ZfpxLe46SvE) link.

