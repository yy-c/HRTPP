# HRTPP: Interpretable Hybrid-Rule Temporal Point Processes

## Overview
Hybrid-Rule Temporal Point Processes (HRTPP) is a novel framework that integrates temporal logic rules with numerical features to enhance both interpretability and predictive accuracy in event modeling. HRTPP is particularly effective in medical applications, such as disease onset prediction, progression analysis, and clinical decision support.

## Key Features
- **Interpretable Event Modeling**: Combines rule-based mechanisms with numerical feature augmentation for structured and explainable predictions.
- **Three-Component Intensity Function**:
  - **Basic Intensity**: Captures intrinsic event likelihood.
  - **Rule-Based Intensity**: Encodes structured temporal dependencies.
  - **Numerical Feature Intensity**: Dynamically adjusts event probabilities.
- **Efficient Rule Mining**:
  - Two-phase rule mining strategy to balance complexity and accuracy.
  - Bayesian optimization for efficient rule space exploration.
- **Multi-Criteria Evaluation Framework**:
  - Assesses rule validity, model fitting, and temporal predictive accuracy.
  - Case studies validate HRTPP’s ability to explain disease progression.

## Installation
Python 3.11. (recommended)

PyTorch version 2.1.0.

Download the repository (for double-blind reviewing).
```bash
# Clone the repository
# git clone https://github.com/anonymized/HRTPP.git
cd HRTPP

# Install dependencies
pip install -r requirements.txt
```

## Usage
Train the model using the dataset located in `data/[data_name].csv` and store the results in `results/results_[data_name].txt`.

```bash
python demo.py
```

## Datasets
The dataset used in this project is stored in CSV format, with each row representing a recorded event. It consists of four columns:

- **`id`**: A unique identifier for each sample.
- **`t`**: The timestamp indicating when the event occurred.
- **`v`**: The observed value of the variable. If a variable has no observed value, its value is set to `1` by default.
- **`k`**: The name of the variable associated with the event.

Each sample (identified by `id`) may contain multiple events with different timestamps (`t`) and variables (`k`). The dataset is structured to support temporal modeling and analysis of event sequences.

HRTPP has been evaluated on real-world medical datasets, demonstrating superior predictive performance and clinical interpretability.

## Citation
If you use HRTPP in your research, please cite:
```
@article{HRTPP2025,
  author    = {Anonymous author},
  title     = {Interpretable Hybrid-Rule Temporal Point Processes},
  journal   = {ECML-PKDD},
  year      = {2025}
}
```

## License
This project is licensed under the Apache-2.0 License - see the [LICENSE] file for details.

## Contact
For questions or collaborations, please contact [Anonymous.email@example.com].

