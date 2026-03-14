# Credit Risk Model With Python

**Overview**

Although many machine learning techniques deliver strong predictive performance, logistic regression remains the preferred choice in finance and risk management for its computational efficiency and interpretability.

This project analyzes the LendingClub peer-to-peer lending dataset to build a credit risk model predicting loan defaults using logistic regression. It follows a full credit scorecard pipeline: from data cleaning and WOE (Weight of Evidence) binning to model training, prediction, and scorecard generation for interpretable risk scores.

Key goal: Transforming data to build logistic regression and create a threshold-based policy to minimize losses while optimizing approvals.

What makes this project unique: Built credit risk pipeline from scratch—custom one-hot encoding, WOE binning class, and preprocessing steps instead of using library functions. Pure logistic regression with handmade feature engineering for production-ready interpretability.

**Data set**
- Source: LendingClub loan data (loan_amnt, int_rate, grade, dti, etc.)
- Target: Binary default (e.g., Good/Bad Loan)

- Size: 466k+ rows, 40+ features (continuous/categorical)

- Challenges: Imbalanced classes (~15% defaults), missing values (e.g., emp_length), Preprocessing Continuous Variables ( WOE Binning/Feature Engineering)

**Result**
- AUC in Sample: 0.8630
- AUC in Test: 0.8629
- Calculate the Threshold with Youden Index (J) as J= TPR - FPR
- Set the threshold at 0.448989 as threshold with score 529 with Approval Rate 0.75 and Rejection Rate at 0.25. Reduce the rejection rate around 0.02 (compared with setting threshold at 0.5) to help the company not lose their customer
