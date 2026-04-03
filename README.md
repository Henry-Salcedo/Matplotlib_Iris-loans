# Matplotlib_Iris/loans Analysis

## Puropse
 Diving into using Matplotlib to create visuals for iris_dataset from sklearn to determine differences in physical traits between the different species, and then uses a loan dataset and creating visuals to then identify any conclusions that I can come up with the visuals.

This project's aim was to focus on “Matplotlib” to create visual models.
The project reads the iris dataset that is within “Scikit learn”, and loan dataset from a Kaggle repository.
The goal is to generate visual models and identify diffrences/conclusions respectfully for each dataset.
  
Key analyses include:

- Created a Scatter, Box, and Violin plot for iris data (Part 1).
- Created a Bar, Scatter, and pie plot for Loan data (Part 2).
- Once the graphs are made, review the diffrences in the iris species, and conclusions in Loans.
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
- load_iris() – Loads data from sklearn
- iris.data – Feature matrix (sepal length, sepal width, petal length, petal width)
- iris.target – Numeric labels for species
- iris.target_names – Names of the species
  
**Methods:**
- loans = ('LoanDataset.csv') – Loan dataset CSV file
- plt.scatter() - Creates scatter plot for sepal length vs width
- plt.boxplot() – Compares distribution of petal widths
- plt.violinplot() – Shows distribution density of sepal lengths


---

### Part 2: 

**Attributes:**
- loans – Pandas DataFrame containing the loan dataset
- loans.data – Feature data used for visualization/analysis
- loans ['loan_intent'] – Loan purpose categories
- loans['loan_amnt'] – Loan amount values
- loans['Current_loan_status'] – Status of the loan (Default / No Default)
  
**Methods:**
- pd.read_csv() – Reads the CSV file and loads it into a DataFrame (Difculty with file so changes to desktop path must be change if other users want to run the file)
- loans = pd.read_csv('LoanDataset.csv') – Loads the loan dataset
- loans.head() – Displays the first few rows of the dataset
  
---

## Limitations
- I limited myself to what I learn in a class related to the project, using only pandas and matlib to create a dataframe for loans.
- Due to minimal experience in usage designs while modified could uses more enhancments when more experience is accumulated. 
  
- Focused mainly on using matlib to better understand it usages and fundamental nature. 

## Post Implementation/Conclusions
These are some of the implementation/Thoughts post methods to highlight the areas that I improved and/or discussed upon the initial methods (Following are notes on each plot for part 1, and the paragragh for the loans section in one place):

### Part 1 Notes:

#### Scatter Plot
- Scatter plot shows 
- Within the scatter plot, we can observe that the setosa species (red) is very distinct from the versicolor (green) and virginica (blue) species, which overlap more. 
- This indicates that sepal length and width are effective features for distinguishing setosa from the other two species, while versicolor and virginica may require additional features for better separation.
- Another aspect is that the overlaping between versicolor and virginica suggests that these two species have similar sepal dimensions, which may lead to potential misclassification if only these features are used.
- Finally, the scatter plot visually confirms that the dataset can be used for classification tasks, due to the noticable clustoring of the species.

#### Scatter Plot
- Setosa's distribution appears to be far less with no noticable overlap with the other species similar to the scatter plot. It also contains 2 outliners within its set. The species also has its median exactly on the lower quartile (Q1) which could be an indication that the values of at least 50% are equal or less than Q1 (indication of heavy clustering).
- With the overlap with Versicolor and Virginica with their upper and lower whiskers respectfully showing that some samples could be of the same size as one another. Which lines up with what was disscussed in the scatter plot. This could lead to misidentifcation in machine learning models reguarding the petal width.
- Virginica has a large IQR from is size, indicating more variety in measurements. The whiskers also extend more (espechually in its lower one) indicating wider total range of values.

#### Scatter Plot
- Noticed features from the boxplot (width) can be notice even more with the violen plot even though we are looking at length (the setosa's is a evenly spread and focused within the median, then both versicolor and virginica do show peaks below the median which is can be a repersentation of the there long whiskers).
- The Virgincia even now show signs of a downward skewness since a few plants have very significantly shorter sepal lengths.
- There is overlap across each species but more noticablly with Versicolor and Virginica. While having averages differ there species could be hard to identify with length alone. A machine-learning model could have a tough time correctly identifying their groups.
- All 3 species violin graph appear to be unimodal (one main peak/common value).

---

Part 2 (Paragraph summary):

 With the bar plots it shows that education and medical loans are the largest taken out, suggesting that borrowers do so out of a necessity towards their lives. Then aside of the outliners that I then remove from the scatter plot for a clearer look into the dataset it show that loans have a very strong predictor due to the high density loan rates that can be seen around the 15% area and above. Lastly the pie chart showed a 4:1 ratio of loans with every 4 borrowers their is 1 that fails payment (Default), and with the very high 21% if the group is a bank then this value is severaly high resulting in a massive financial loss that may be covered with the intrests from the other loans, but such speculation must be further expanded upon to confirm.
