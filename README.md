# Bio-informatics

Copy of the code found here: https://github.com/NabaviLab/Multimodal-GNN-for-Cancer-Subtype-Clasification

Improvements include adding two other GNN models: GIN and GSage. 
We also reformatted the code to help adding new models, corrected some errors in the code.

# Datasets

The dataset are available in the original repository: https://github.com/NabaviLab/Multimodal-GNN-for-Cancer-Subtype-Clasification

# Graphs

To create the graphs found in a report, you need to launch `graph_creation.py`, found in the main directory.

# Organisation of the code

We kept the original code in the directory `original_code` and our code is in the directory `our_code`.


# Running the code

To run the code, the command is `python cancer_test.py` to run into the `our_code directory`.
To change the different parameters of the models, it is only needed to change the parameters in the code, in the dictionary `desired_combinations`.
