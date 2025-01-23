# License / 许可证

This project is released under a custom non-commerical license, prohibiting its use for any commerical purposes.

本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途

# This project is the code of the below paper

X. Fu, Y. Wang, Y. Lin, T. Ohtsuki, G. Gui and H. Sari, "Toward Robust Open-Set Radiofrequency Signal Identification in Internet of Things Using Hypersphere Manifold Embedding," in IEEE Internet of Things Journal, vol. 11, no. 24, pp. 41235-41247, 15 Dec.15, 2024, doi: 10.1109/JIOT.2024.3457832.

# First Step
run "./ADS-B_Close-Set/train.py"

obtain the identification model and save it in "./ADS-B_Close-Set/model_weight/"

# Second Step
move the model to "./ADS-B_Open-Set/model_weight"

run "./ADS-B_Open-Set/test.py"

obtain the features of training samples and testing samples, and save them in "./ADS-B_Open-Set/EVT/data/"

# Third Step
run the "./ADS-B/EVT/plot_sphere_bl.m"

obtain the consine similarity of training samples and testing samples, and save them in "./ADS-B_Open-Set/EVT/data/"

run the "./ADS-B/EVT/getResultsMLOSR.m"

obtain the classification results, including the ACC and predicted labels, where the predicted labels are save in "./ADS-B_Open-Set/EVT/cm/" 

you can draw the confusion matrix by "./ADS-B_Open-Set/EVT/cm/CM.m" using the predicted labels and real labels

# Results
