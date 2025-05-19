# Covid-19_Drug-Analysis

## Project Description
This project focuses on the analysis of molecular structures and their properties to facilitate COVID-19 drug discovery. It offers a complete workflow from data preparation, exploratory data analysis (EDA), and molecular feature extraction to advanced machine learning for predicting pIC50 values, followed by rigorous model evaluation and feature importance assessment to highlight key drivers of bioactivity.

## Key Features

🔹 Streamlined data preparation and cleaning process.

🔹 Comprehensive molecular descriptor generation using RDKit.

🔹 Identification of activity cliffs to highlight structurally similar compounds with significant activity differences.

🔹 Scaffold analysis for structural classification of compounds.

🔹 Application of multiple regression models for accurate pIC50 prediction.

🔹 Detailed model evaluation and feature importance assessment.

## Dataset Information

The [**Data**](./data/) folder consists of two files (*DDH Data with Properties.csv* and *DDH Data*) highlighting prospective anti-COVID-19 drug molecules.

 **DDH Data.csv**: Contains SMILES as well as pIC50 values of chemical compounds against the COVID-19 virus.

**DDH Data with Properties.csv**: Contains additional molecular properties fetched using the pubchempy Python library, which accesses PubChem—a comprehensive database of millions of chemical compounds.

The dataset is publicly provided by the Government of India as part of their Drug Discovery Hackathon.


