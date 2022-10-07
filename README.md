# HUNTINGTON'S DISEASE (HD)

## Supervised Learning Analysis of the Enroll-HD Registry in Relation to Weight and Ability to Eat
This study aims to use the Enroll-HD database to examine the relationship between the progression of HD-related symptoms and participants’ BMI.

## AIMS
To determine the relationship between Huntington’s disease (HD) progression (as measured within the Enroll-HD registry) and BMI in persons with manifest HD.
<br>To compare the performance of a trained logistic regression (logit) and support vector machine (SVM) models in predicting fifth follow-up BMI classes in terms of accuracy, recall, precision, and f1-score.

## ABSTRACT
To determine the progression of Huntington’s Disease (HD) in a well-defined sample of persons with HD followed at over 155 sites by the Enroll-HD observational study, BMI was assessed in 866 adults, 546 manifest (with a confirmed diagnosis and manifestation of HD symptoms) and 320 control (not having the HD genetic mutation). They were followed for an estimated (mean ± SD) 5.4±0.6 years. Using the expansive Enroll-HD registry cohort, we investigated the relationship(s) between HD progression measures and body mass index (BMI) in patients with motor-manifest HD. The simple vector machine (SVM) and logistic regression models were trained to predict BMI classes at fifth follow-up based on selected variables which were analysed to determine their association with the target variable. After tuning both models, the SVM model gained 11% (51% - 62%) in accuracy, while the GLM gained 9% (45% - 54%). The SVM model showed superiority to the GLM model. Both models’ precision/sensitivity and f1-score of the underweight class of fifth follow-up BMI was notably poorer than other classes. We found CCC (combined clinical characteristics) and combined feedself, novel derived variables, to be relevant predictors of BMI in addition to other established variables such as baseline BMI, CAP score, age of onset of impairment (motor, cognitive, etc.), etc. While the predictive influence of variables like cross-sectional chorea, baseline age, and gender appeared to be minimal. As more data become available, we are likely to unravel more causal relationships within the Enroll-HD variables.
<br><br>NOTE: 
<br>Detailed version of this research can be found [here](https://github.com/Beegie01/Supervised-Learning-Analysis-of-Enroll-HD-Features-in-Relation-to-BMI-in-the-Manifest-Stage/blob/main/Final%20Project%20Thesis.pdf)
<br>However, below is a summarized overview of this research work.

## DATASET
![handling missing data](https://github.com/Beegie01/Supervised-Learning-Analysis-of-Enroll-HD-Features-in-Relation-to-BMI-in-the-Manifest-Stage/blob/main/Handling%20Missing%20Entries.png)<br>
The above illustration highlights how the data wrangling/preparation process led to the use of only the first 6 visit records of 866 participants.

## EXPERIMENTAL APPROACH
![experimental flow-chart](https://github.com/Beegie01/Supervised-Learning-Analysis-of-Enroll-HD-Features-in-Relation-to-BMI-in-the-Manifest-Stage/blob/main/Experimental%20design.png?raw=true) 
<br>The above diagram is a flow-chart representation of the entire experimental process, where each arrow indicates progression from one step/process to the next.<br>
The target labels (five classes of BMI) can also be seen.

## RESULTS
![model evaluation](https://github.com/Beegie01/Supervised-Learning-Analysis-of-Enroll-HD-Features-in-Relation-to-BMI-in-the-Manifest-Stage/blob/main/model%20performance%20evaluation.png)
<br>As shown above, after training and tuning of hyperparameters, the logit model had 57% test accuracy at predicting the outcome, while the SVM classifier showed its superiority with a better accuracy of 62%. 
<br>The SVM classifier also outperformed the logit model in terms of predicting the normal BMI class, while the logit model showed better f1-score at predicting the underweight class. However, both models had similar performance at predicting all other BMI classes (i.e., overweight, obese, and severely obese).
<br><br>![UHDRS scores]()

## DISCUSSION
Besides the clear association between baseline BMI and the outcome variable, this result supports (Ghazaleh et al., 2021; Langbehn et al., 2019; Tabrizi et al., 2013) who found that CAP score and CAG repeat length are among the most important factors for predicting HD progression. Other top-ranked predictors such as cross-sectional motor impairment and behaviour_score (higher in manifest than controls, p<0.001) have a negative correlation with the outcome. While cognitive_score, tfcscore, and feedself (less in manifest than controls, p<0.001) each have positive correlation with the outcome variable. Interestingly, baseline age and cross-sectional chorea, and gender had no significant correlation with the outcome.
<br>Investigation into the association between CAG repeat length and fifth follow-up BMI revealed that the CAG length of manifest has a weak negative correlation with outcome in manifest and no correlation in control. It was observed that, at baseline, the mean BMI of reduced penetrant participant was significantly higher than that of fully penetrant participants and no significant difference was found between their fifth follow-up BMI. This may indicate that having a fully penetrant CAG repeat length causes a faster drop in BMI than having reduced penetrant CAG, which is consistent with the findings of Rosenblatt et al. (2012). However, the median values suggest that there is no significant difference across both periods. Hence, more data may help to clarify this observation.
<br>In the future, more data would help in training the machine learning algorithms more effectively without having to make too many copies of the data. Additionally, predicting a change in BMI or the magnitude of cross-sectional weight change may be more relevant in terms of helping manifest HD patients in better managing their health.

## CONCLUSION
The result of this classification task confirms the superiority of the SVM over the GLM especially when dealing with a highly dimensional dataset with multiple classes as in this case, as both models benefitted equally from the tuning of their hyperparameters. In this research, we found CCC (combined clinical characteristics) and combined feedself, novel derived variables, to be relevant predictors of BMI in addition to other established variables such as baseline BMI, CAP score, age of onset of impairment (motor, cognitive, etc.), etc. While the predictive influence of variables like cross-sectional chorea, baseline age, and gender appeared to be minimal. The insights gained from this work can serve as a background for future investigations. As more data become available, we are likely to unravel more causal relationships within the Enroll-HD variables.
