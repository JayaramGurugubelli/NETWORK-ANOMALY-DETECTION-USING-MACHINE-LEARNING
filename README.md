📡 Network Anomaly Detection Using Machine Learning

This project detects malicious or abnormal network traffic using a combination of machine learning models, heuristics, and a Streamlit-based user interface. 
The goal is to identify threats such as DoS, DDoS, scanning, and suspicious packet behavior in network flow data.
🚀 Features
✔ Supervised Anomaly Detection

Uses label-based datasets to classify traffic as:

BENIGN

DDoS / DoS

Port Scan

Suspicious Traffic

Potential Malicious Attack

(Handled inside model.py)

✔ Unsupervised Detection

When no labels exist in uploaded files, the system:

Analyzes packet sizes

Detects abnormal IP activity

Predicts attack type using heuristics

(Handled inside ui1.py)

✔ Interactive Streamlit Interface

Upload CSV or Parquet network flow files and instantly:

View dataset preview

See attack labels or predicted behavior

Inspect suspicious packets

Read automatically generated conclusions
