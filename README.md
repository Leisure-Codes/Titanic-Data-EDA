Titanic Dataset Analysis
The Titanic dataset is a well-known dataset for exploring machine learning concepts. It includes information about passengers on the RMS Titanic, and the goal is often to predict whether a given passenger survived the shipwreck.

1. Importing the Dataset
python
Copy
Edit
# Importing Pandas Library
import pandas as pd

# Loading Data
titanic = pd.read_csv('https://raw.githubusercontent.com/eclarson/DataMiningNotebooks/master/data/titanic.csv')
2. Basic Data Operations
Viewing the Dataset
By default:

python
Copy
Edit
titanic
By slicing:

python
Copy
Edit
titanic[:10]
Using head():

python
Copy
Edit
titanic.head()  # Default is 5 rows
titanic.head(10)  # View top 10 rows
Using iloc[]:

python
Copy
Edit
titanic.iloc[:10]
Viewing specific columns:

python
Copy
Edit
titanic[['PassengerId', 'Survived', 'Name']].head(10)
Basic Statistics
Summary statistics:

python
Copy
Edit
titanic.describe()
Mean of age:

python
Copy
Edit
titanic['Age'].mean()
Median:

python
Copy
Edit
titanic.median(numeric_only=True)
Mode:

python
Copy
Edit
titanic.mode(axis=0)
Count:

python
Copy
Edit
titanic.count()
Maximum and minimum values:

python
Copy
Edit
titanic.max(numeric_only=True)
titanic.min(numeric_only=True)
Dataset size:

python
Copy
Edit
titanic.size
Shape:

python
Copy
Edit
titanic.shape
Number of dimensions:

python
Copy
Edit
titanic.ndim
Column data types:

python
Copy
Edit
titanic.dtypes
3. Cleaning and Transforming the Dataset
Handling Missing Values
Dropping rows with missing Embarked values:

python
Copy
Edit
titanic = titanic.dropna(subset=['Embarked'])
Dropping the Cabin column due to extensive missing values:

python
Copy
Edit
titanic.drop('Cabin', inplace=True, axis=1)
Handling missing Age:

Dataset 1: Drop rows with missing Age.
python
Copy
Edit
titanic_age = titanic.dropna()
Dataset 2: Impute missing Age with the median value.
python
Copy
Edit
from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values=np.nan, strategy='median')
imr = imr.fit(titanic)
titanic_imputed = imr.transform(titanic.values)
Dropping Irrelevant Columns
Columns such as PassengerId, Name, Ticket, and Fare are irrelevant for analysis:

python
Copy
Edit
titanic_imputed = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)
titanic_age = titanic_age.drop(['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)
Encoding Categorical Data
One-hot encode Sex and Embarked columns:

python
Copy
Edit
titanic_imputed = pd.get_dummies(titanic_imputed, columns=['Sex', 'Embarked'])
4. Exploratory Data Analysis and Hypothesis Testing
Hypothesis:
Women and children were given evacuation priority, so we expect more female survivors and child survivors among males.

Testing with Dataset 1
Female and Male Survivors:
python
Copy
Edit
Test1 = titanic_age.groupby(['Survived'])
Test2 = Test1.get_group(1)  # Survivors

Female = Test2['Sex'].value_counts()['female']
Male = Test2['Sex'].value_counts()['male']

print(f"Number of Female Passengers That Survived: {Female}")
print(f"Number of Male Passengers That Survived: {Male}")
Male Child Survivors:
python
Copy
Edit
Test3 = Test2[Test2['Sex'] == 'male']
Test4 = Test3[Test3['Age'] <= 18.0]

Child_Survivors = Test4.count()
print(f"Number of Male Child Passengers That Survived: {Child_Survivors}")
Testing with Dataset 2
Female and Male Survivors:
python
Copy
Edit
Test_Imp_1 = titanic_imputed_df.groupby(['Survived'])
Test_Imp_2 = Test_Imp_1.get_group(1)

Female = int(Test_Imp_2['Sex_female'].sum())
Male = int(Test_Imp_2['Sex_male'].sum())

print(f"Number of Female Passengers That Survived: {Female}")
print(f"Number of Male Passengers That Survived: {Male}")
5. Conclusion
Out of the total survivors, the majority were women, supporting the hypothesis that women were prioritized during evacuation.
Among male survivors, children had a higher survival rate compared to adult males.
Dataset 1 and Dataset 2 both confirm the hypothesis, showcasing the importance of exploratory data analysis and preprocessing in understanding dataset patterns.

This is a concise version of your Titanic analysis workflow. Let me know if you want any further explanations or visualizations!
