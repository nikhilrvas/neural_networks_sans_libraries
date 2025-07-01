import pandas as pd

#Reading data
data = pd.read_csv('ce889_dataCollection.csv',names=['x1','x2','y1','y2'])  

#Removing the null values & duplicates
data.replace("",float("NaN"),inplace=True) 
data.dropna(inplace=True)
data.drop_duplicates(keep='first',inplace=True) 

#Shuffling the rows.
data = data.sample(frac=1).reset_index(drop=True)  

#Scaling data.
def min_max_scaling(column):      
    return (column-column.min())/(column.max()-column.min())
data_scaled = data.apply(min_max_scaling)

#Partitioning the data into training, testing & validation datasets. 
x =data_scaled[['x1','x2']] 
y =data_scaled[['y1','y2']]
num_rows = len(data)
train_size = int(0.8*num_rows)

x_train = x.iloc[:train_size]
y_train = y.iloc[:train_size]

x_validate = x.iloc[train_size:]
y_validate = y.iloc[train_size:]

# Creating training, testing and validation csv files. 
x_train.to_csv('x_train.csv',index=False)
y_train.to_csv('y_train.csv',index=False)
x_validate.to_csv('x_validate.csv',index=False)
y_validate.to_csv('y_validate.csv',index=False)



