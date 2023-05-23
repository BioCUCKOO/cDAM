# cDAM
**Note**: The cDAM framework, which incorporated graph neural network (GNN) and deep neural network (DNN), was developed to predict potentially new autophagy regulators and their MMAs from big omics datasets. 

## Requirements

The main requirements are listed below:

* Python 3.8
* torch
* Scikit-Learn
* Matplotlib

## The description of cDAM source codes

* Teacher_model_predictor.py

    The code is used for the pre-training model loading and performance evaluation.
* Dsf_model_predictor.py

    The code is used for the Disulfiram induction model loading and data prediction.
* Sdn_model_predictor.py

    The code is used for the Nitrogen starvation induction model loading and data prediction.
* Sdg_model_predictor.py

    The code is used for the Glucose starvation induction model loading and data prediction.


## The models in cDAM

* teacher_model

    The model is used for the pre-training performance evaluation.
* Dsf_model

    The model is used for the Disulfiram induction model data prediction.
* Sdn_model

    The model is used for the Nitrogen starvation induction model data prediction.
* Sdg_model

    The model is used for the Glucose starvation induction model data prediction.
