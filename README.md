# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()
```
<img width="419" height="245" alt="image" src="https://github.com/user-attachments/assets/b489b55d-9f5f-47d7-847a-3af43d78109e" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```

<img width="224" height="118" alt="image" src="https://github.com/user-attachments/assets/a93f2662-c5c4-4967-b0fd-4070d4bdd833" />

```
df.dropna()
```
<img width="651" height="542" alt="image" src="https://github.com/user-attachments/assets/1cafc6eb-32e9-49df-a558-0501ca0387c7" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
```
<img width="444" height="241" alt="image" src="https://github.com/user-attachments/assets/759385b3-6df6-46a9-be21-edee018943c6" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
<img width="409" height="440" alt="image" src="https://github.com/user-attachments/assets/c9c87883-b975-4d7b-b9b9-3322484978d4" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="484" height="433" alt="image" src="https://github.com/user-attachments/assets/61d52cfc-6081-49e4-92e0-648595bc0958" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```


```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```

<img width="500" height="242" alt="image" src="https://github.com/user-attachments/assets/9952c563-36ec-4dd7-a3cb-384e23c345ca" />


```
df=pd.read_csv("income(1) (1).csv")
df.info()
```
<img width="476" height="426" alt="image" src="https://github.com/user-attachments/assets/cd0603db-5cbe-4fa6-bf8c-a5037f2cc27a" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="345" height="316" alt="image" src="https://github.com/user-attachments/assets/8047e0e4-106f-4d11-b83a-356b6601c374" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```


<img width="859" height="728" alt="image" src="https://github.com/user-attachments/assets/07894f55-dd83-45fc-8f21-7d539922560a" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```


<img width="845" height="521" alt="image" src="https://github.com/user-attachments/assets/41b96a5d-fa8a-4315-8b28-e5554037bb17" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

<img width="587" height="851" alt="image" src="https://github.com/user-attachments/assets/dc9e704f-5843-48e5-864a-6a0b199db310" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("income(1) (1).csv")
df.info()
```
<img width="884" height="725" alt="image" src="https://github.com/user-attachments/assets/fe3ac66b-9200-4da4-a78c-cb140c717b31" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```


<img width="856" height="513" alt="image" src="https://github.com/user-attachments/assets/4ff6bc67-772a-48f6-92ef-c374652ac033" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="856" height="513" alt="image" src="https://github.com/user-attachments/assets/b11985b2-9285-4a0f-af7c-d9f5275bed93" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```

<img width="698" height="102" alt="image" src="https://github.com/user-attachments/assets/ba9df6a9-c343-41d5-9a92-6a06168e9bb5" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

<img width="511" height="839" alt="image" src="https://github.com/user-attachments/assets/3412747c-2ac3-4feb-89d4-aeed6a221fcc" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```

<img width="571" height="26" alt="image" src="https://github.com/user-attachments/assets/b85ed77d-ffeb-44f4-812c-bc6fc6aa8572" />

```
!pip install skfeature-chappers
```


<img width="847" height="547" alt="image" src="https://github.com/user-attachments/assets/efe9304c-e93a-4e5f-a277-abf37cf68234" />


```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```


<img width="852" height="512" alt="image" src="https://github.com/user-attachments/assets/6470a3af-0e62-464e-ba58-5bf18a685b14" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```

<img width="853" height="59" alt="image" src="https://github.com/user-attachments/assets/57e07a95-0c2d-4f90-8b84-2e243bf6de93" />

```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("income(1) (1).csv")
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```


<img width="875" height="514" alt="image" src="https://github.com/user-attachments/assets/a79df35d-2842-49cc-b391-6f3233c196f7" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```


<img width="822" height="723" alt="image" src="https://github.com/user-attachments/assets/b6242b01-7d74-450f-87b6-364e1f7eb0d3" />

# RESULT:
Thus the code has been executed successfully.
