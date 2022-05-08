# import libraries 
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 



# import Dataset
dataset = pd.read_csv("Data.csv")

# devide dataset to dependent and independent 
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# handling missing data
imputer = SimpleImputer(missing_values=np.nan , strategy= "mean") 
x[:,1:3] = imputer.fit_transform(x[:,1:3])

# Encoding Categorical Data
ct_X = ColumnTransformer([('0', OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct_X.fit_transform(x)
LabelEncoder_y = LabelEncoder()
y= LabelEncoder_y.fit_transform(y)


# Spiliting data to training set and testing set.
x_train, x_test, y_train, y_test = train_test_split(x,y , test_size = .2, random_state = 0)

#feature scaling.
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_train)
