# Privacy Attacks and Defenses in Autonomous Vehicles under Federated Learning with Differential Privacy (AVs_Privacy)

## Overview
AVs_Privacy is a repository dedicated to implementing and evaluating attack models such as membership inference attacks and data reconstruction attacks. The project enables systematic comparison of privacy and performance trade-offs between centralized detection model and federated learning models, with and without differential privacy mechanisms. This approach helps demonstrate the effectiveness of differential privacy in mitigating privacy attacks in federated learning settings.

## Contents
- **baseline/**: Contains baseline models for privacy analysis.
- **dataset/**: Includes datasets used for training and evaluation.
- **differential_privacy/**: Implements differential privacy in federated learning.
- **federated_learning/**: Contains code for federated learning models.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/dawoodwasif/AVs_Privacy.git
    cd AVs_Privacy
    ```

2. Create and activate a virtual environment:
    ```bash
    conda create --name privacy python=3.10
    conda activate privacy
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the models and experiments, use the following commands:

1. Navigate to the desired module (e.g., `baseline`, `differential_privacy`, or `federated_learning`):
    ```bash
    cd baseline
    ```

2. Execute the main script:
    ```bash
    python main.py
    ```

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Updates

- **15.05.2024 YOLOv8 Baseline Scripts Added** : [Details]
- **01.06.2024 Federated Learning Scripts Added** : [Details]
- **15.06.2024 Differential Privacy Scripts Added** : [Details]
- **Attack Model Scripts Upcoming** : [Details]

### References

[1]: [Glancy, Dorothy J. "Privacy in autonomous vehicles." Santa Clara L. Rev. 52 (2012): 1171.](https://digitalcommons.law.scu.edu/cgi/viewcontent.cgi?article=2728&context=lawreview) 

[2]: [Jallepalli, Deepthi, et al. "Federated learning for object detection in autonomous vehicles." 2021 IEEE Seventh International Conference on Big Data Computing Service and Applications (BigDataService). IEEE, 2021.](https://ieeexplore.ieee.org/abstract/document/9564384/) 
