1. Introduction
The Human Resource Department of a technology-oriented company is about to start a program to mitigate mental health issues of the companies’ employees.
To increase the quality of this program the key hints of a representative survey shall be included. Unfortunately, this survey consists only of raw data making it difficult to interpret.
Therefore, the main objective of this document is to perform an unsupervised machine learning to cluster the data into different groups for easier interpretation and to mitigate some results for the program. 

2. Approach
The survey from https://www.kaggle.com/osmi/mental-health-in-tech-2016 serves as input for unsupervised machine learning. The learning was realized using PyCharm Community Edition 2024.1.3 and Python 3.12. The complete source code consists of following components: 
•	data analysis.py: Main module containing explorative data analysis, preprocessing, modelling, dimension reduction and quality assessment 
•	functions.py: Module with redundant functions used in the main module
•	mental-heath-in-tech-2016_20161114.csv: Initial data from the survey
•	model_results.xlsx: Results of unsupervised machine learning process
•	feature_translation_table.xlsx: Initial questions have been transformed into two-digit alphabetical code. This file can be used for traceability of 2 digit code and initial question.

3. Usage 
Download the files and execute the "data analysis.py"
