from ftplib import all_errors
import math
from random import random
from csv import reader
import csv
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.7
lambda_rate = 0.7
momentum_rate = 0.1
#Input, output and hidden layer neeurons.
no_inp_nrns = 2
no_op_nrns = 2
no_hd_nrns = 8
#No. of weights from input layer to hidden layer, and from hidden layer to output layer. 
no_wts_inp_lyr = (no_inp_nrns * no_hd_nrns)  
no_wts_op_lyr = (no_hd_nrns * no_op_nrns)  

#Class which is used to calculate rmse. 
class calculate_rmse_error:
    
    #Feed forwards loop.
    def feed_forward(self,inp_lyr_wts, op_lyr_wts, X, inp_lyr_bias_wts, op_lyr_bias_wts):
        #Value of hidden layer neurons.
        valuesof_hd_lyr_nrns = []
        valuesof_op_lyr_nrns = []
        index_wts1 = 0
        index_hd = 0
        for index_hd in range(no_hd_nrns):
            index_wts2 = index_wts1 + no_hd_nrns
            z = (inp_lyr_wts[index_wts1] * float(X[0])) + (inp_lyr_wts[index_wts2] * float(X[1])) + (1 * inp_lyr_bias_wts[index_hd])
            x = self.activation_func(z)
            valuesof_hd_lyr_nrns.append(x)
            index_wts1 = index_wts1 + 1
        #Values of output layer neurons
        o = 0
        n = o
        for i in range(no_op_nrns):
            m = []
            m.append(n)
            for j in range(no_hd_nrns - 1):
                n = n + no_op_nrns
                m.append(n)
            r = 0
            t = 0
            b = len(m)
            for k in range(len(m)):
                r = r + (op_lyr_wts[m[k]] * valuesof_hd_lyr_nrns[t])
                t = t + 1
            r = r + (1 * op_lyr_bias_wts[i])
            s = self.activation_func(r)
            valuesof_op_lyr_nrns.append(s)
            n = o + 1
        return valuesof_op_lyr_nrns
    
    def __init__(self,inp_lyr_wts,op_lyr_wts,X,Y, inp_lyr_bias_wts,op_lyr_bias_wts):
        self.X = X
        self.Y = Y
        self.inp_lyr_wts = inp_lyr_wts
        self.op_lyr_wts = op_lyr_wts
        self.inp_lyr_bias_wts = inp_lyr_bias_wts
        self.op_lyr_bias_wts = op_lyr_bias_wts
        self.valuesof_hd_lyr_nrns = []
        self.valuesof_op_lyr_nrns = []
        iffo = self.feed_forward(self.inp_lyr_wts,self.op_lyr_wts, self.X,self.inp_lyr_bias_wts, self.op_lyr_bias_wts)
        #Error Calculation of the Output.
        error = 0
        for i in range(no_op_nrns):
            e = float(self.Y[i]) - iffo[i]
            e = e * e 
            error = error + e
        error = error / 2
        all_errors.append(error)  
    
    #Sigmoid activation function.
    def activation_func(self,inp_val):
        return 1 / (1 + np.exp(-inp_val*lambda_rate))


#Class which runs feed forward & back propagation loops
class Network: 
    def __init__(self, no_inp_nrns, no_op_nrns, no_hd_nrns, no_wts_inp_lyr, no_wts_op_lyr,
                 inp_lyr_wts, op_lyr_wts, inp_lyr_delta_wts, op_lyr_delta_wts,
                 X, Y, inp_lyr_bias_wts, op_lyr_bias_wts, inp_delta_bias_wts,
                 op_delta_bias_wts):

        #Number of neurons in the input, hidden & output.
        self.no_inp_nrns = no_inp_nrns 
        self.no_hd_nrns = no_hd_nrns  
        self.no_op_nrns = no_op_nrns
        #Number of weights from different layers in the network.
        self.no_wts_inp_lyr = no_wts_inp_lyr  
        self.no_wts_op_lyr = no_wts_op_lyr  
        self.inp_lyr_wts = inp_lyr_wts
        self.op_lyr_wts = op_lyr_wts
        self.inp_lyr_bias_wts = inp_lyr_bias_wts
        self.op_lyr_bias_wts = op_lyr_bias_wts
        self.inp_delta_bias_wts = inp_delta_bias_wts
        self.op_delta_bias_wts = op_delta_bias_wts
        self.X = X
        self.Y = Y
        self.values_hd_lyr_nrns = []  
        self.values_op_lyr_nrns = []  
        self.op_lyr_errors = []  
        self.op_lyr_local_gradient = []
        self.hd_lyr_local_gradient = []
        self.inp_lyr_wts_updated = []
        self.op_lyr_wts_updated = []
        self.inp_bias_wts_updated = []
        self.op_bias_wts_updated = []
        self.inp_lyr_delta_wts = inp_lyr_delta_wts
        self.op_lyr_delta_wts = op_lyr_delta_wts

    def feed_forward(self):
        #Value of hidden layer neurons.
        index_wts1 = 0
        index_hd = 0
        for index_hd in range(no_hd_nrns):
            index_wts2 = index_wts1 + no_hd_nrns
            z = (self.inp_lyr_wts[index_wts1] * float(X[0])) + (self.inp_lyr_wts[index_wts2] * float(X[1])) + (1 * self.inp_lyr_bias_wts[index_hd])
            x = self.activation_func(z)
            self.values_hd_lyr_nrns.append(x)
            index_wts1 = index_wts1 + 1
        #Value of output layer neurons.
        o = 0
        n = o
        for i in range(self.no_op_nrns):
            m = []
            m.append(n)
            for j in range(self.no_hd_nrns - 1):
                n = n + self.no_op_nrns
                m.append(n)
            r = 0
            t = 0
            b = len(m)
            for k in range(len(m)):
                r = r + (self.op_lyr_wts[m[k]] * self.values_hd_lyr_nrns[t])
                t = t + 1
            r = r + (1 * self.op_lyr_bias_wts[i])
            s = self.activation_func(r)
            self.values_op_lyr_nrns.append(s)
            n = o + 1
        #Error computation.
        for i in range(self.no_op_nrns):
            e = float(self.Y[i]) - self.values_op_lyr_nrns[i]
            self.op_lyr_errors.append(e)

    def back_propagation(self):
        #Local gradient calculation.
        for i in range(self.no_op_nrns):
            lg = learning_rate * self.values_op_lyr_nrns[i] * (1 - self.values_op_lyr_nrns[i]) * \
                 self.op_lyr_errors[i]
            self.op_lyr_local_gradient.append(lg)
        k = 0
        #Delta weights calculation.
        for i in range(self.no_hd_nrns):
            for j in range(self.no_op_nrns):
                a = (learning_rate * self.op_lyr_local_gradient[j] * self.values_hd_lyr_nrns[i]) + (momentum_rate * self.op_lyr_delta_wts[k])
                self.op_lyr_delta_wts[k] = a
                k = k + 1
        #Delta bias weights calculation.
        for i in range(self.no_op_nrns):
            a = (learning_rate * self.op_lyr_local_gradient[i] * 1) + (momentum_rate * self.op_delta_bias_wts[i])
            self.op_delta_bias_wts[i] = a
        #Updating output layer weights.
        for i in range(self.no_wts_op_lyr):
            a = self.op_lyr_wts[i] + self.op_lyr_delta_wts[i]
            self.op_lyr_wts_updated.append(a)
        #Updating output layer delta bias weights.
        for i in range(self.no_op_nrns):
            a = self.op_lyr_bias_wts[i] + self.op_delta_bias_wts[i]
            self.op_bias_wts_updated.append(a)

        #Local gradient calculation.
        sums = 0
        weight_index = 0
        for i in range(self.no_hd_nrns):
            temp = 0
            for j in range(len(self.op_lyr_local_gradient)):
                temp = temp + (self.op_lyr_wts[weight_index] * self.op_lyr_local_gradient[j])
                weight_index = weight_index + 1
            lg = learning_rate * self.values_hd_lyr_nrns[i] * (1 - self.values_hd_lyr_nrns[i]) * temp
            self.hd_lyr_local_gradient.append(lg)
        #Calculating delta weights.
        k = 0
        for i in range(self.no_inp_nrns):
            for j in range(self.no_hd_nrns):
                a = (learning_rate * self.hd_lyr_local_gradient[j] * float(self.X[i])) + (momentum_rate * self.inp_lyr_delta_wts[k])
                self.inp_lyr_delta_wts[k] = a
                k = k + 1
        #Calculating bias delta weights.
        for i in range(self.no_hd_nrns):
            a = (learning_rate * self.hd_lyr_local_gradient[i] * 1) + (momentum_rate * self.inp_delta_bias_wts[i])
            self.inp_delta_bias_wts[i] = a
        #Updating input layer weights.
        for i in range(self.no_wts_inp_lyr):
            a = self.inp_lyr_wts[i] + self.inp_lyr_delta_wts[i]
            self.inp_lyr_wts_updated.append(a)
        #Updating input layer bias weights.
        for i in range(self.no_hd_nrns):
            a = self.inp_delta_bias_wts[i] + self.inp_lyr_bias_wts[i]
            self.inp_bias_wts_updated.append(a)

    #Sigmoid function.
    def activation_func(self, inp_val):
        return 1 / (1 + math.exp(-lambda_rate * inp_val))



epoch_count = []  
rmse_array = [] 
rmse1_array = []
epoch_no = 0  
prev_rmse_validation = 1  
rmse_validation = 0.9
#Stopping criteria when RMSE plateaus 
while rmse_validation < prev_rmse_validation : 
    #Updating previous values.
    prev_rmse_validation = rmse_validation 
    #Reading training data's first row
    with open('x_train.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        header = next(csv_reader)
    X = header
    with open('y_train.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        header = next(csv_reader)
    Y = header
    #Assigning random values to input, output, input bias & output bias weights for first iteration of the first epoch.  
    if epoch_no == 0:
        itoh_lyr_wts = []  
        for i in range(no_wts_inp_lyr):  
            a = random()
            itoh_lyr_wts.append(a)
        htoo_lyr_wts = []  
        for i in range(no_wts_inp_lyr):  
            a = random()
            htoo_lyr_wts.append(a)
        itoh_bias_wts = []
        for i in range(no_hd_nrns): 
            a = random()
            itoh_bias_wts.append(a)
        htoo_bias_wts = []
        for i in range(no_op_nrns):  
            a = random()
            htoo_bias_wts.append(a)
    else:
        #Second epcoch onwards.
        itoh_lyr_wts = itoh
        htoo_lyr_wts = htoo
        itoh_bias_wts = itoh_bias
        htoo_bias_wts = htoo_bias

    #Delta weights declaration.
    itoh_delta_wts = []
    htoo_delta_wts = []
    itoh_bias_delta_wts = []
    htoo_bias_delta_wts = []
    #Delta weights are zero in first iteration of an epoch. 
    for i in range(no_wts_inp_lyr):
        itoh_delta_wts.append(0)
    for i in range(no_wts_inp_lyr):
        htoo_delta_wts.append(0)
    for i in range(no_hd_nrns):
        itoh_bias_delta_wts.append(0)
    for i in range(no_op_nrns):
        htoo_bias_delta_wts.append(0)
    #First iteration of the epoch.
    a = Network(no_inp_nrns,no_op_nrns,no_hd_nrns,no_wts_inp_lyr,
                no_wts_inp_lyr,itoh_lyr_wts,htoo_lyr_wts,itoh_delta_wts,htoo_delta_wts,
                X,Y,itoh_bias_wts,htoo_bias_wts,itoh_bias_delta_wts,htoo_bias_delta_wts)
    a.feed_forward()
    a.back_propagation()
    #Reading data for remaining epochs.
    with open('x_train.csv', 'r') as read_x:
        with open('y_train.csv', 'r') as read_y:
            csv_x_reader = reader(read_x)
            csv_y_reader = reader(read_y)
            x_header = next(csv_x_reader)
            x_header = next(csv_x_reader)
            y_header = next(csv_y_reader)
            y_header = next(csv_y_reader)
            sum = 1
            
            while sum <= 36000:  
                #Number of rows in the training data.
                sum = sum + 1 
                X = next(csv_x_reader)
                Y = next(csv_y_reader)
                #Updating weights after each iteration.
                itoh_lyr_wts = a.inp_lyr_wts_updated
                htoo_lyr_wts = a.op_lyr_wts_updated
                itoh_delta_wts = a.inp_lyr_delta_wts
                htoo_delta_wts = a.op_lyr_delta_wts
                itoh_bias_wts = a.inp_bias_wts_updated
                htoo_bias_wts = a.op_bias_wts_updated
                itoh_bias_delta_wts = a.inp_delta_bias_wts
                htoo_bias_delta_wts = a.op_delta_bias_wts

                a = Network(no_inp_nrns,no_op_nrns,no_hd_nrns,no_wts_inp_lyr,
                            no_wts_inp_lyr,itoh_lyr_wts,htoo_lyr_wts,itoh_delta_wts,
                            htoo_delta_wts,
                            X, Y, itoh_bias_wts, htoo_bias_wts, itoh_bias_delta_wts, htoo_bias_delta_wts)
                a.feed_forward()
                a.back_propagation()
    #Updating final weights after each epoch.
    itoh = a.inp_lyr_wts_updated
    htoo = a.op_lyr_wts_updated
    itoh_bias = a.inp_bias_wts_updated
    htoo_bias = a.op_bias_wts_updated
    all_errors = []
    #Updating final weights for rmse.
    itoh_lyr_wts = a.inp_lyr_wts_updated
    htoo_lyr_wts = a.op_lyr_wts_updated
    itoh_bias_wts = a.inp_bias_wts_updated
    htoo_bias_wts = a.op_bias_wts_updated
    
    with open('x_train.csv', 'r') as read_x:
        with open('y_train.csv', 'r') as read_y:
            csv_x_reader = reader(read_x)
            csv_y_reader = reader(read_y)
            x_header = next(csv_x_reader)
            y_header = next(csv_y_reader)
            sum = 1
            while sum <= 36000:
                sum = sum + 1
                X = next(csv_x_reader)
                Y = next(csv_y_reader)
                b = calculate_rmse_error(itoh_lyr_wts, htoo_lyr_wts, X, Y, itoh_bias_wts, htoo_bias_wts)
    summation = 0
    for i in range(len(all_errors)):
        #Adding all errors for rmse calculation.
        summation = summation + all_errors[i]
    summation = summation / len(all_errors)
    rmse_train = math.sqrt(summation)
    #Rounding error to five decimal places.
    rmse_train = round(rmse_train, 5)
    rmse1_array.append(rmse_train)

    all_errors = []
    #RMSE calculation for validation set
    itoh_lyr_wts = a.inp_lyr_wts_updated
    htoo_lyr_wts = a.op_lyr_wts_updated
    itoh_bias_wts = a.inp_bias_wts_updated
    htoo_bias_wts = a.op_bias_wts_updated
    with open('x_validate.csv', 'r') as read_x:
        with open('y_validate.csv', 'r') as read_y:
            csv_x_reader = reader(read_x)
            csv_y_reader = reader(read_y)
            x_header = next(csv_x_reader)
            y_header = next(csv_y_reader)
            sum = 1
            while sum <=8500:
                sum = sum + 1
                X = next(csv_x_reader)
                Y = next(csv_y_reader)
                c = calculate_rmse_error(itoh_lyr_wts, htoo_lyr_wts,X, Y, itoh_bias_wts, htoo_bias_wts)
    summation = 0
    for i in range(len(all_errors)):
        summation = summation + all_errors[i]
    summation = summation / len(all_errors)
    rmse_validation = math.sqrt(summation)
    rmse_validation = round(rmse_validation,6)

    print(rmse_validation)

    epoch_count.append(epoch_no)
    rmse_array.append(rmse_validation)
    epoch_no = epoch_no + 1
print(epoch_count)
plt.plot(epoch_count, rmse_array, label='rmse_validation')
plt.xlabel('epoch count')
plt.ylabel('RMSE')
plt.show()
plt.plot(epoch_count, rmse1_array, label='rmse_train')
plt.xlabel('epoch count')
plt.ylabel('RMSE')
plt.show()
    
#Initializing variables to store min and max values for columns y1 and y2.
min_col3 = float('inf')
max_col3 = float('-inf')
min_col4 = float('inf')
max_col4 = float('-inf')
min_values = []
max_values = []

with open('ce889_dataCollection.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    #Skip the header row.
    next(reader, None)
    #Iterate through the rows to find min and max values in columns y1 & y2.
    for row in reader:
        val_col3 = float(row[2])  
        val_col4 = float(row[3])  
        min_col3 = min(min_col3,val_col3)
        max_col3 = max(max_col3,val_col3)
        min_col4 = min(min_col4,val_col4)
        max_col4 = max(max_col4,val_col4)  
    min_values.append(min_col3)
    min_values.append(min_col4)
    max_values.append(max_col3)
    max_values.append(max_col3)

#Saving the final weights in a text file.
with open('Weights.txt', 'w') as output:
    output.write(str(c.inp_lyr_wts))
    output.write('\n')
    output.write(str(c.op_lyr_wts))
    output.write('\n')
    output.write(str(c.inp_lyr_bias_wts))
    output.write('\n')
    output.write(str(c.op_lyr_bias_wts))
    output.write('\n')
    output.write(str(min_values))
    output.write('\n')
    output.write(str(max_values))


    


