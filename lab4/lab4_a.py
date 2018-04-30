import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


#read date
houses= pd.read_csv("house_prices.csv")

#exploratory data
print houses.head()
print "shape of data is ", houses.shape

#manually caculated
samples = [[2050,2,1]]
samples_x = pd.DataFrame(samples,columns= ['SqFt','Bedrooms','Bathrooms'])
house_x = houses[['SqFt','Bedrooms','Bathrooms']]
clf_NN = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(house_x)
dist,indices = clf_NN.kneighbors(samples_x)
print "dist is ",dist
print "indices is ", indices
if isinstance(indices.item(0,0), int):
    tem_list =[]
    for i in range (0,3):
        tem_list.append(houses.iloc[indices.item(0,i),1])
price_sample = sum(tem_list)/3
print "sample price is ",price_sample

#Building a KNN regressor
houses_y = houses[['Price']]
houses_x = houses[['SqFt','Bedrooms','Bathrooms','Offers','Brick','Neighborhood']]
houses_x['Brick'].replace(['No','Yes'],[0,1],inplace = True)
houses_x['Neighborhood'].replace(['West','North','East'],[1,2,3],inplace = True)
houses_x_train,houses_x_test,houses_y_train,houses_y_test = train_test_split(houses_x,houses_y,test_size = 0.4, random_state = 0)
print "train data:",houses_x_train.shape
print "test data:",houses_x_test.shape
clf_KNN = KNeighborsRegressor(n_neighbors=3)
clf_KNN.fit(houses_x_train,houses_y_train)
pred = clf_KNN.predict(houses_x_test)
pred_df = pd.DataFrame(pred,columns=['Predict_Price'])
comp = pd.merge(houses_y_test,pred_df,left_index=True, right_index=True)
print comp.head(10)

#Visualize
plt.scatter(comp.Price, comp.Predict_Price)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Predicted vs actual prices on the test data")
plt.savefig('lab4_a_output.png')
plt.show()
