# License / 许可证

This project is released under a custom non-commerical license, prohibiting its use for any commerical purposes.

本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途

# Open-Set Radiofrequency Signal Identification Using Hypersphere Manifold Embedding

Radiofrequency signal Identification (RSI) provides a critical security solution for device authentication in the Internet of Things (IoT), characterized by extensive interconnections and interactions among numerous entities. By analyzing received radiofrequency signals, device-specific features are extracted at the receiver and used for identification. In a dynamic and ever-changing communication environment, where some devices not visible during the training process may appear during testing, a robust RSI method must not only identify devices encountered during training but also reject those that were not. In this paper, we propose an open-set RSI method based on hypersphere manifold embedding. This approach leverages hypersphere projection for radiofrequency signal feature extraction on a hypersphere manifold, thereby avoiding the need to optimize intra-device variation in the radial direction. Additionally, we introduce an open-set identification approach based on generalized Pareto distribution, which does not rely on any radiofrequency signals from unknown devices. Extensive experimental results demonstrate that the proposed method achieves state-of-the-art identification performance.

You can find the details in our paper: X. Fu, Y. Wang, Y. Lin, T. Ohtsuki, G. Gui and H. Sari, "Toward Robust Open-Set Radiofrequency Signal Identification in Internet of Things Using Hypersphere Manifold Embedding," in IEEE Internet of Things Journal, vol. 11, no. 24, pp. 41235-41247, 15 Dec.15, 2024, doi: 10.1109/JIOT.2024.3457832.

# How to run the code?
(1) First Step

run "./ADS-B_Close-Set/train.py"

obtain the identification model and save it in "./ADS-B_Close-Set/model_weight/"

(2) Second Step

move the model to "./ADS-B_Open-Set/model_weight"

run "./ADS-B_Open-Set/test.py"

obtain the features of training samples and testing samples, and save them in "./ADS-B_Open-Set/EVT/data/"

(3) Third Step

run the "./ADS-B/EVT/plot_sphere_bl.m"

obtain the consine similarity of training samples and testing samples, and save them in "./ADS-B_Open-Set/EVT/data/"

run the "./ADS-B/EVT/getResultsMLOSR.m"

obtain the classification results, including the ACC and predicted labels, where the predicted labels are save in "./ADS-B_Open-Set/EVT/cm/" 

you can draw the confusion matrix by "./ADS-B_Open-Set/EVT/cm/CM.m" using the predicted labels and real labels

# Results

