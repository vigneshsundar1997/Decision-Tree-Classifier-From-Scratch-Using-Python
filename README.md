# Decision-Tree-Classifier-From-Scratch-Using-Python
Implementation of Decision Tree Classifier from scratch using Python on Dating Dataset and analysis of the various hyper parameters.


The project folder contains 4 python files: 
1. preprocess-assg4.py
2. trees.py
3. cv_depth.py
4. cv_frac.py

############
1. preprocess.py

This script contains the preprocessing steps like removing the columns, normalization, label encoding, discretization and split the dataset. It makes use of dating-full.csv as the input. It outputs trainingSet.csv and testSet.csv.

Execution : python3 preprocess-assg4.py

2. trees.py

This script contains the training and testing of the model for Decision Tree. It takes in two arguments, the training file name, the test file.

Execution : python3 trees.py trainingFileName testFileName

eg: 
Run command for DT model
python3 trees.py trainingSet.csv testSet.csv  

3. cv_depth.py

This script performs the ten fold validation for the Decision Tree model based on tree depth.

Execution : python3 cv_depth.py

4. cv_frac.py

This script performs the ten fold validation for the Decision Tree model with training size as the parameter.

Execution : python3 cv_frac.py
