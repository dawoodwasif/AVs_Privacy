# Privacy in AVs under Federated Learning with Differential Privacy (AVs_Privacy)

## Technical Approach

Our work explores the trade-off between privacy and fairness in federated learning-based object detection systems. We introduce RESFL (Robust and Equitable Sensitive Federated Learning), which combines adversarial privacy disentanglement with uncertainty-guided fairness-aware aggregation to optimize both privacy protections and fairness guarantees.

### Key Technical Components

1. **Adversarial Privacy Disentanglement**:
   - Gradient reversal layer to remove sensitive attributes while preserving utility
   - Feature extraction optimization that minimizes privacy leakage

2. **Uncertainty-Guided Fairness-Aware Aggregation**:
   - Evidential neural networks for uncertainty quantification
   - Adaptive client contribution weighting based on fairness metrics
   - Constrained optimization balancing performance and fairness objectives

## Benchmark Methods

This project implements and evaluates several federated learning algorithms focused on privacy-fairness trade-offs:

1. **FedAvg**: Standard federated averaging baseline
   ```bash
   python federated_learning/FedAvg_train.py
   ```

2. **FedAvg-DP (Objective)**: Differential privacy with noise injection
   ```bash
   # For epsilon = 0.1
   python differential_privacy/baseline.py --dp_method objective --epsilon 0.1
   
   # For epsilon = 0.5
   python differential_privacy/baseline.py --dp_method objective --epsilon 0.5
   ```

3. **FairFed**: Fairness-aware aggregation
   ```bash
   python federated_learning/FairFed_train.py
   ```

4. **PrivFairFL**: Privacy-fairness trade-off methods
   ```bash
   # Pre-aggregation variant
   python benchmark/PrivFairFL_Pre_train.py
   
   # Post-local update variant
   python benchmark/PrivFairFL_Post_train.py
   ```

5. **PUFFLE**: Unified privacy-fairness approach
   ```bash
   python benchmark/PUFFLE_train.py
   ```

6. **PFU-FL**: Privacy-fairness-utility balancing
   ```bash
   python benchmark/FL_PFU_train.py
   ```

7. **RESFL**: Our proposed approach
   ```bash
   python resfl/RESFL.py --clients 10 --rounds 100 --adv_lambda 0.1 --uncertainty_threshold 0.75
   ```

## Project Structure

- **baseline/**
  - YOLOv8 baseline implementations
  - Fairness evaluation tools (`fairness_evaluate.py`)

- **dataset/**
  - CARLA synthetic data generator (`generate_carla/`)

- **differential_privacy/**
  - DP-SGD implementations with adaptive noise calibration
  - Objective, input, and output perturbation methods

- **federated_learning/**
  - Standard FL implementations (FedAvg, FedProx, etc.)
  - Evaluation scripts for performance metrics

- **resfl/**
  - Our proposed approach implementation
  - Fairness evaluation tools

- **benchmark/**
  - Privacy-fairness trade-off implementations (PrivFairFL, PUFFLE, PFU-FL)

## Installation

```bash
# Clone or download the repository
cd AVs_Privacy

# Create and activate conda environment
conda create --name privacy python=3.10
conda activate privacy

# Install dependencies
pip install -r requirements.txt
```

## Dataset Generation

### CARLA Synthetic Dataset

Generate a synthetic dataset with diverse pedestrian demographics and environmental conditions:

```bash
cd dataset/generate_carla
./main_dataset.sh
```

This script generates data across:
- 3 different towns (Town01, Town03, Town05)
- 10 pedestrian skin tone groups
- 13 weather conditions (variations of cloud, rain, fog)

## Running Federated Learning Models

### Training Models

```bash
# Navigate to the federated learning directory
cd federated_learning

# Run standard FedAvg baseline
python FedAvg_train.py

# Run our proposed RESFL approach
cd ../resfl
python RESFL.py --clients 4 --rounds 100 --adv_lambda 0.1
```

### Evaluating Models

```bash
# Evaluate model performance at specific rounds
python eval_round_map.py

# Generate performance plots
python eval_plot.py

# Assess fairness across demographic groups
python eval_fairness.py --model path/to/model.pt --test_path path/to/test/images --metadata path/to/metadata.json
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
