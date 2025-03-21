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
Download the repository (for double-blind reviewing).
```bash
# Clone the repository
# git clone https://github.com/anonymized/HRTPP.git
cd HRTPP

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
python demo.py
```

## Datasets

HRTPP has been evaluated on real-world medical datasets, demonstrating superior predictive performance and clinical interpretability compared to state-of-the-art interpretable TPPs.

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
This project is licensed under the License - see the [LICENSE] file for details.

## Contact
For questions or collaborations, please contact [Anonymous.email@example.com].

