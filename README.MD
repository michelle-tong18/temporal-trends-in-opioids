# temporal-trends-in-nsLBP-opioids

## Description
Opioid Prescription Trends for Non-Specific Lower Back Pain: Time Series and Patient-Level Modeling. Understanding how opioid prescribing patterns evolve over time and for whom is critical to building responsible clinical decision support tools and policies. This analysis models 6-month opioid prescription outcomes following first presentation for non-specific lower back pain (ns-LBP) using EHR data from 2012–2024. Change point detection and interrupted time series analysis capture population-level shifts in prescribing behavior, while multivariate logistic regression quantifies patient-level variation across demographics, social determinants, and diagnoses using comparative subgroup interaction metrics. The pipeline demonstrates practical integration of time series analysis and interpretable modeling, with emphasis on real-world clinical data, data-driven insight, and reproducible ML workflows.

## Getting Started

### Dependencies
The scripts in this repository have been tested with Python 3.9.


### Installing
1. Set up a github API Key
2. Clone the repo
```
git clone git@github.com:michelle-tong18/temporal-trends-in-opioids.git
```

## Pipeline

### Extract electronic medical records (EMR). 
1. Specific inclusion and exclusion criteria to identify a patient cohort. 
2. Extract tabular medical chart data for patient cohort.
3. Impute missing demographic information after evaluating missingness type (MCAR, MAR).

### Prepare patient profiles of clinical-relevant features.
4. Change point detection and interrupted time series analysis (ITS) using Wagner's segmented regression identify significant shifts in opioid prescribing rates over time. ITS  quantifies level and slope changes in prescription trends from 2012–2024 to evaluate the impact of potential interventions or policy changes.
5. Multivariate logistic regression with comparative interaction metrics assess relative changes in opioid prescriptions across time and patient characteristics. The ratio odds ratios (ROR) and relative excess risk due to interaction (RERI) evaluate disparities across patient subgroups.

## Acknowledgements
- Modified forest plot visualization code based on the original implementation by [LSYS/forestplot](https://github.com/LSYS/forestplot)

## License
Distributed under the MIT License. See LICENSE.txt for more information.

## Citations
* Tong MW, Joseph GB, Majumdar S. Trends in Opioid Prescribing for Nonspecific Low Back Pain After CDC Guidelines and California Prescription Drug Monitoring Programs, 2012-2024. Public Health Reports®. 2026;0(0). doi:[10.1177/00333549261486260](https://doi.org/10.1177/00333549261486260)
* Tong MW, Joseph GB, Ziegeler K, Kreutzinger V, Majumdar S. Opioid Prescribing Trajectories for Nonspecific Low Back Pain: Sociodemographic and Clinical Heterogeneity Across Three Policy Periods, 2012–2024. Pain Reports. 2026;11(6). doi:[10.1097/PR9.0000000000001516](https://doi.org/10.1097/PR9.0000000000001516) 
