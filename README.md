# HUNTINGTON'S DISEASE (HD)

## Supervised Learning Analysis of the Enroll-HD Registry in Relation to Weight and Ability to Eat
This study aims to use the Enroll-HD database to examine the relationship between the progression of HD-related symptoms and participants’ BMI.

## AIMS
To determine the relationship between Huntington’s disease (HD) progression (as measured within the Enroll-HD registry) and BMI in persons with manifest HD.
<br>To compare the performance of a trained logistic regression (logit) and support vector machine (SVM) models in predicting fifth follow-up BMI classes in terms of accuracy, recall, precision, and f1-score.

## ABSTRACT
To determine the progression of Huntington’s Disease (HD) in a well-defined sample of persons with HD followed at over 155 sites by the Enroll-HD observational study, BMI was assessed in 866 adults, 546 manifest (with a confirmed diagnosis and manifestation of HD symptoms) and 320 control (not having the HD genetic mutation). They were followed for an estimated (mean ± SD) 5.4±0.6 years. Using the expansive Enroll-HD registry cohort, we investigated the relationship(s) between HD progression measures and body mass index (BMI) in patients with motor-manifest HD. The simple vector machine (SVM) and logistic regression models were trained to predict BMI classes at fifth follow-up based on selected variables which were analysed to determine their association with the target variable. After tuning both models, the SVM model gained 11% (51% - 62%) in accuracy, while the GLM gained 9% (45% - 54%). The SVM model showed superiority to the GLM model. Both models’ precision/sensitivity and f1-score of the underweight class of fifth follow-up BMI was notably poorer than other classes. We found CCC (combined clinical characteristics) and combined feedself, novel derived variables, to be relevant predictors of BMI in addition to other established variables such as baseline BMI, CAP score, age of onset of impairment (motor, cognitive, etc.), etc. While the predictive influence of variables like cross-sectional chorea, baseline age, and gender appeared to be minimal. As more data become available, we are likely to unravel more causal relationships within the Enroll-HD variables.
Detailed version of this research can be [here](https://github.com/Beegie01/Supervised-Learning-Analysis-of-Enroll-HD-Features-in-Relation-to-BMI-in-the-Manifest-Stage/blob/main/Final%20Project%20Thesis.pdf)

## DATASET
![handling missing data](https://github.com/Beegie01/Supervised-Learning-Analysis-of-Enroll-HD-Features-in-Relation-to-BMI-in-the-Manifest-Stage/blob/main/Handling%20Missing%20Entries.png)<br>
The above illustration highlights how the data wrangling/preparation process led to the use of only the first 6 visit records of 866 participants.

## EXPERIMENTAL APPROACH
![experimental flow-chart](https://github.com/Beegie01/Supervised-Learning-Analysis-of-Enroll-HD-Features-in-Relation-to-BMI-in-the-Manifest-Stage/blob/main/Experimental%20design.png?raw=true) 
<br>The above diagram is a flow-chart representation of the entire experimental process, where each arrow indicates progression from one step/process to the next.<br>
The target labels (five classes of BMI) can also be seen.
