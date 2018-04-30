import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#read date
houses= pd.read_csv("house_prices.csv")
houses_y = houses[['Neighborhood']]
houses_x = houses[['Price','SqFt','Bedrooms','Bathrooms','Offers','Brick']]
houses_x['Brick'].replace(['No','Yes'],[0,1],inplace = True)
houses_x_train,houses_x_test,houses_y_train,houses_y_test = train_test_split(houses_x,houses_y,test_size = 0.4, random_state = 0)
print "train data:",houses_x_train.shape
print "test data:",houses_x_test.shape

def knn_class (k):
    clf_KNC = KNeighborsClassifier(n_neighbors = k)
    clf_KNC.fit(houses_x_train,houses_y_train)
    pred = clf_KNC.predict(houses_x_test)
    pred_df = pd.DataFrame(pred,columns=['Predict_Neighborhood'])
    comp = pd.merge(houses_y_test,pred_df,left_index=True, right_index=True)
    return( pd.crosstab(comp.Neighborhood, comp.Predict_Neighborhood))

for i in range (1,9):
    print "k = ",i
    print knn_class(i)
