# DeepDPI: Advancing Drug Discovery through Deep Learning-Enhanced Prediction of Drug-Protein Interactions

![DeepDPI Banner](https://via.placeholder.com/800x200?text=DeepDPI)

## Overview

DeepDPI is a comprehensive framework for predicting drug-protein interactions (DPIs) using state-of-the-art machine learning and deep learning techniques. By accurately predicting how drugs interact with specific protein targets, DeepDPI aims to accelerate drug discovery, reduce development costs, and enable more personalized therapeutic approaches.

## Key Features

- **Multi-modal feature engineering** for both small molecules and proteins
- **Ensemble learning approaches** with traditional ML models
- **Advanced deep learning architectures** including CNNs, Transformers, and Graph Neural Networks
- **Hyperparameter optimization** via Gray Wolf Optimization, Genetic Algorithms, and Bayesian methods
- **Comprehensive performance evaluation** using metrics designed for imbalanced classification

## Protein Targets

DeepDPI focuses on the following high-value protein targets:

| Protein | Disease Association |
|---------|---------------------|
| EPHX2 (sEH) | High blood pressure, diabetes progression |
| BRD4 | Cancer progression |
| ALB (HSA) | Drug absorption and distribution |
| Beta-Amyloid (APP) | Alzheimer's disease |
| CFTR | Cystic fibrosis |
| Tau (MAPT) | Alzheimer's disease, tauopathies |
| HER2 (ERBB2) | Breast cancer |
| BCR-ABL1 | Chronic Myeloid Leukemia |
| PD-1 | Cancer (immunotherapy target) |

## Methodology

Our pipeline consists of:

1. **Data Collection & Preprocessing**
   - Sources: ChEMBL and BELKA databases
   - Statistical validation (Mann-Whitney U test)
   - Classification of compounds as active/inactive/intermediate
   - Balancing techniques for skewed datasets

2. **Feature Engineering**
   - **Small Molecules**: 
     - Extended Connectivity Fingerprints (ECFP)
     - PubChem fingerprints
     - Lipinski descriptors
   - **Proteins**:
     - Amino acid composition (up to 3-mers)
     - Conjoint Triad features
     - Quasi-sequence order descriptors
     - Explainable Substructure Partition Fingerprints (ESPF)

3. **Model Development**
   - **Traditional ML**:
     - Random Forest, CatBoost, XGBoost
     - SVM (linear and RBF kernels)
     - KNN, Decision Trees, Extra Trees
     - Gaussian Naive Bayes and more
   - **Deep Learning**:
     - CNN for molecular and protein sequences
     - RNN for sequence dependencies
     - Transformer-based models for contextual sequence modeling
     - Graph Neural Networks for molecular structures

4. **Hyperparameter Optimization**
   - Genetic Algorithms
   - Bayesian Optimization with Gaussian Processes
   - Optuna framework
   - Gray Wolf Optimization

5. **Evaluation Metrics**
   - F1-score
   - Area Under ROC Curve (AUROC)
   - Area Under Precision-Recall Curve (AUPRC)
   - Cross-entropy loss

## Results

### Traditional Machine Learning

- **PubChem Fingerprints**: 
  - Ensemble of 24 classifiers
  - 97% training accuracy, 85% test accuracy
  - Random Forest as top performer
  - Gray Wolf Optimization improved test accuracy to 87%

- **ECFP Fingerprints**:
  - 97% training accuracy with CatBoost
  - Evaluated using mean Average Precision (mAP)

### Deep Learning

| Encoding Combination | Training Performance | Test Performance |
|----------------------|----------------------|------------------|
| MPNN (drug), CNN (target) | AUROC: 0.997, F1: 0.933 | AUROC: 0.996, F1: 0.938 |
| CNN (drug), Transformer (target) | AUROC: 1.000, F1: 0.980 | AUROC: 0.999, F1: 0.983 |
| Morgan (drug), AAC (target) | AUROC: 1.000, F1: 0.974 | AUROC: 0.999, F1: 0.938 |
| ESPF (drug), AAC (target) | AUROC: 1.000, F1: 0.993 | AUROC: 1.000, F1: 0.978 |
| CNN-RNN (drug), Conjoint-triad (target) | AUROC: 0.999, F1: 0.962 | AUROC: 0.999, F1: 0.958 |

The **CNN + Transformer** and **ESPF + AAC** combinations delivered the best overall performance.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepDPI.git
cd DeepDPI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from deepdpi import DeepDPIPredictor

# Initialize predictor with pre-trained model
predictor = DeepDPIPredictor(model_type="cnn_transformer")

# Predict interaction for a drug-protein pair
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
protein_sequence = "MKKFFDSRREQ..."  # Example protein sequence

result = predictor.predict(smiles, protein_sequence)
print(f"Interaction probability: {result['probability']:.4f}")
print(f"Active compound: {'Yes' if result['is_active'] else 'No'}")
```

## Future Work

- Expand evaluation to standard datasets (BindingDB, Davis)
- Enhance model explainability using SHAP
- Develop combined protein-compound interaction prediction models
- Explore transfer learning and pre-training techniques
- Implement feature importance analysis for both ML and DL models

## Citation

If you use DeepDPI in your research, please cite:

```
@article{author2025deepdpi,
  title={DeepDPI: Advancing Drug Discovery through Deep Learning-Enhanced Prediction of Drug-Protein Interactions},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  year={2025},
  publisher={Publisher Name}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- ChEMBL and BELKA databases for providing valuable data
- RDKit library for cheminformatics functionality
- The research community for continuous advancements in drug discovery methods
