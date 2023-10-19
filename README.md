# Ex: 06 FEATURE TRANSFORMATION
## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Algorithm:
Step1: Read the given Data.
Step2: Clean the Data Set using Data Cleaning Process.
Step3: Apply Feature Transformation techniques to all the features of the data set.
Step4: Print the transformed features.

## Program:
```
NAME: JENISHA.J
REG.NO: 212222230056
```
```
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
#### Function Transformation
```
df.skew()

np.log(df["Highly Positive Skew"])

np.reciprocal(df["Moderate Positive Skew"])

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Positive Skew"])

```
#### Power Transformation
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Negative Skew_boxcox"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
#### Quantile Transformation
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

sm.qqplot(df["Moderate Negative Skew_1"],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
df

sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

## Output:
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/2e054f62-b835-43d2-8f47-efcee728fab2)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/a3f3bc3e-e663-4cd1-99ed-c734850c684f)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/9477a098-5f49-4203-a987-c31643eb2cb6)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/6496ab77-fb77-4d81-af83-a3680a5c7e55)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/dbcadc87-d908-4249-bae2-d7dda75e3cf6)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/374f801b-1638-48cd-a5e0-8c90a3c9ed1f)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/435d2d2d-565f-46ef-bbf1-0c6c54146782)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/7364d223-9700-4f2b-9387-903478cf7324)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/6a8935e0-0333-4f5c-90d4-4aeeac32fee9)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/8d556ad9-31b7-4111-8a61-225cca8c0eba)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex06/assets/119405070/971119cf-1d7e-47d7-b69d-ce13c9c5b5ac)

## Result:
Thus feature transformation is done for the given dataset.
