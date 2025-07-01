import math
from optparse import Values
import csv
import numpy as np

#The hyper parameters.
learning_rate=0.7
lambda_rate=0.35
momentum_rate=0.1
no_inp_nrns=2
no_op_nrns=2
no_hd_nrns=8


class NeuralNetHolder:
    inp_lyr_wts=[]
    op_lyr_wts=[]
    inp_lyr_bias_wts=[]
    op_lyr_bias_wts=[]
    min_vals=[]
    max_vals=[]
    
    #Activation Function.
    def activation_func(self,inp_val):
        if inp_val>=-100:
            return 1/(1 + np.exp(-lambda_rate*inp_val))
        else:
            return 1/(1 + np.exp(-lambda_rate*-100))
            
    
    #Loading the trained model
    def __init__(self):
        self.load_lists()
    
    def load_lists(cls):
        with open('Weights.txt','r') as file:
            #Reading the contents of the file.            
            content=file.read()
        lists=content.split(']\n[')
        #Removing brackets and split each list. Then append to respective containers.
        for i,lst in enumerate(lists):
            values=lst.strip('[]').split(', ')
            if i==0:
                cls.inp_lyr_wts=values
            elif i==1:
                cls.op_lyr_wts=values
            elif i==2:
                cls.inp_lyr_bias_wts=values
            elif i==3:
                cls.op_lyr_bias_wts=values
            elif i==4:
                cls.min_vals=values
            elif i==5:
                cls.max_vals=values
        cls.inp_lyr_wts=[float(val) for val in cls.inp_lyr_wts]
        cls.inp_lyr_bias_wts=[float(val) for val in cls.inp_lyr_bias_wts]
        cls.op_lyr_wts=[float(val) for val in cls.op_lyr_wts]
        #Convert elements of inp_lyr_bias_wts to float.
        cls.op_lyr_bias_wts=[float(val) for val in cls.op_lyr_bias_wts]
        cls.min_vals=[float(val) for val in cls.min_vals]
        cls.max_vals=[float(val) for val in cls.max_vals]
                
        
    def predict(self,inp_row):
        split_vals=inp_row.split(',')
        val_0=float(split_vals[0])
        val_1=float(split_vals[1])
        valuesof_hd_lyr_nrns=[]
        valuesof_op_lyr_nrns=[]
        
        index_wts1 = 0
        index_hd = 0
        for index_hd in range(no_hd_nrns):
            index_wts2 = index_wts1 + no_hd_nrns
            z = (self.inp_lyr_wts[index_wts1] * float(val_0)) + (self.inp_lyr_wts[index_wts2] * float(val_1)) + (1 * self.inp_lyr_bias_wts[index_hd])
            x = self.activation_func(z)
            valuesof_hd_lyr_nrns.append(x)
            index_wts1 = index_wts1 + 1
        #Calculate values of output layer neurons.
        o=0
        n=o
        for i in range(no_op_nrns):
            m = []
            m.append(n)
            for j in range(no_hd_nrns - 1):
                n = n+no_op_nrns
                m.append(n)
            r=0
            t=0
            b = len(m)
            for k in range(len(m)):
                r = r+(self.op_lyr_wts[m[k]]*valuesof_hd_lyr_nrns[t])
                t = t+1
            r = r+(1*self.op_lyr_bias_wts[i])
            s = self.activation_func(r)
            
            out = s * (self.max_vals[i] - self.min_vals[i]) + self.min_vals[i]
            valuesof_op_lyr_nrns.append(out)
            n = o+1
        return valuesof_op_lyr_nrns
    
                                                                                                                                                                  
    


        