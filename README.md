# Matplotlib_Iris-loans

## Puropse
 Diving into using Matplotlib to create visuals for iris_dataset from sklearn to determine differences in physical traits between the different species, and then uses a loan dataset and creating visuals to then identify any conclusions that I can come up with the visuals.


This project's aim was to focus on “Matplotlib” to create visual models.
The project reads the iris dataset that is within “Scikit learn”, and loan dataset from a Kaggle repository.
The goal is to generate visual models and identify diffrences/conclusions respectfully for each dataset.
  
Key analyses include:
- Extracted the data (X) and target (y).
- Split the data into 80% training and 20% test_size.
- Created a scatter plot, Box plot, and Violin Plot for iris data.
- Created a Bar plot, Scatter plot, and Box Plot for Loan data.
- Once graph is made, review the diffrences in the iris species, and conclusions in Loans.
- After each graph notes on what I noticed, and for the Loan section also write a brief paragraph on the conclusions I found.
  
## Class Design
The project can be classified into 2 parts, first with 3 sections and the later with 4:

For each part (Part 1: Iris Dataset, Part 2: LoanDataset)
- Load Dataset (Sciklearn Iris, and Kaggle 'LoanDataset')
- Creating visuals
- Disscuss changes

- * Specifc for Part 2:
- Write brief paragraph on conclusions from the visuals.

## Class Attributes and Methods

### Part 1: 

**Attributes:**
- b_cancer.data - Feature data from the dataset
-
  
**Methods:**
- datasets.load_breast_cancer() - Loads the breast cancer dataset


---

### Part 2: 

**Attributes:**
- b_cancer.data - Feature data from the dataset
-
  
**Methods:**
- datasets.load_breast_cancer() - Loads the breast cancer dataset

---
## Limitations
- I limited myself to what I learn in a class related to the project (e.g., not performing any scaled features or cross-validation).
- In addition to the limit of class-related, I also limited myself from generating models since such an area has not been discussed (Once more, such a thing is possible now, however, these limits are to test what the library can do).
- Focused solely on using sklearn to establish a better understanding of its usage. 

## Post Implementation/Thoughts
These are some of the implementation/Thoughts post methods to highlight the areas that I improved and/or discussed upon the initial methods:

- Messed with the n-neigbors = 5 - After reviewing other areas like 3, or 13, they did little to improve more than 5, if I could uses something like StandardScaler , and gridsearch, then those would greatly help determine the best possible k value.
- RandomForestClassifier(n_estimators=200) - Change from 100 to 200 and saw improvements to the accuracy, precision, and F1 scores.
