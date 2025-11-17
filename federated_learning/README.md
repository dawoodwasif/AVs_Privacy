# Federated Learning Object Detection with YOLOv8

This repository contains the necessary code and configurations for running object detection tasks using YOLOv8 within a Federated Learning framework, specifically tailored for environments where data privacy is crucial. The system has been designed to work effectively with the KITSE Dataset for autonomous vehicle applications.

## Error Handling Improvement

During the development and initial runs, an error was identified related to empty metric arrays when interpolating for plotting purposes. The issue occurs when the metric array `y` intended for plotting remains empty due to conditional errors in data collection or failures in metric computation during training phases. To handle this scenario gracefully and avoid runtime errors, the following conditional check was introduced in the plotting function:

instead of line 89 in ultralytics/utils/callbacks/wb.py:
`y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()`

replace it by:

```
if y.size == 0:
    y_mean = 0
    y_log = 0
else:
    y_mean = np.mean(y, axis=0)
    y_log = np.interp(x_new, x, y_mean).round(3).tolist()
```   
This adjustment ensures that the system handles cases of missing data without interruption and avoids the ValueError that occurs when attempting to interpolate with an empty dataset.

### Environment Setup and Running Instructions


In yolov8.yaml, in line 5 change the number of classes according to the dataset:

`nc: 1  # number of classes 8 for KITTI and 1 for FACET`

### Run the Training Script

Execute the training process by running:

`python FedAvg_train.py`

Configuration Files

Before running the training script, ensure that you have all the necessary configuration files in place, as specified in config_paths within train.py. These files should define the dataset paths and other training parameters specific to each federated client.


## Extra FL Baselines (for completeness; **not reported in the paper**)

This repo ships several optimizer/communication variants to support reproducibility and future extensions.  
They are **not part of the evaluated baselines or tables** and were **not tuned** beyond defaults, since our study centers on fairness/privacy algorithms.

**Included extras**
- `FedProx_train.py` — proximal regularization for heterogeneous clients  
- `FedAdam_train.py` — server-side Adam aggregation  
- `FedNova_train.py` — normalization for varying local steps  
- `FedMA_train.py` — matching/averaging with permutation alignment  
- `FedGH_train.py` — gradient homogenization  
- `CHFL_train.py` — compressed/heuristic FL variant

**Quick run (example)**
```bash
# Example extra baselines (not used in the paper’s tables)
python FedProx_train.py
python FedAdam_train.py
```

**References**

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*.  
2. Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems (MLSys)*, 2, 429–450.  
3. Ju, L., Qiao, X., Liu, S., Xu, Z., & Zhao, L. (2024). Accelerating fair federated learning: Adaptive Federated Adam. *IEEE Transactions on Machine Learning in Communications and Networking*, 2, 1017–1032.  
4. Wang, J., Charles, Z., Xu, Z., McMahan, H. B., & Ramaswamy, S. (2020). Tackling the objective inconsistency problem in heterogeneous federated optimization. *NeurIPS*, 33, 7611–7623.  
5. Wang, H., Yurochkin, M., Sun, Y., Papailiopoulos, D., & Khazaeni, Y. (2020). Federated learning with matched averaging. *arXiv preprint arXiv:2002.06440*.  
6. Yi, L., Chen, D., Xu, J., et al. (2023). FedGH: Heterogeneous federated learning with generalized global header. *Proceedings of the 31st ACM International Conference on Multimedia*.  
7. Mori, J., Teranishi, I., & Furukawa, R. (2022). Continual horizontal federated learning for heterogeneous data. *IJCNN 2022*.  
