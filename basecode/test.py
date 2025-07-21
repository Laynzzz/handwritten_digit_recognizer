# This script is just for testing things out. Is not given by the project zip file
import numpy as np
import torch
import nnScript
from scipy.io import loadmat
import facennScript
import pickle

#-----------------------------------------------------------------------------------------------------------------------------------
#testing sigmoid
scale = 0
#print(nns.sigmoid(scale))

vector = np.array([0,1,2])
#print(nns.sigmoid(vector))

matrix = np.array([[1,2,3],[4,5,0]])
#print(nns.sigmoid(matrix))

#--------------------------------------------------------------------------------------------------------------------------------
#paly around with datasets
mat = loadmat('mnist_all.mat')
train_data, train_label, validation_data, validation_label, test_data, test_label = facennScript.preprocess()
print(train_data.shape)

'''test  = [[[1,2,3],[1,5,3]],
         [[1,8,3],[1,11,3]],
         [[1,14,3],[1,17,3]]]

test2 = np.array([[1,2],
                [3,4],
                  [5,6]])

test3 = np.array([[1,2,3],
                 [3,4,1]])'''

#array = np.array(test)
#test = np.vstack(test)
#print(array.shape)
#print(np.ones((test.shape[0], 1)))
#test[-1] = np.ones((test.shape[1]))
#print(test)
#print(type(test[0].shape[0]))
'''
bias = np.ones((test2.shape[0],1))
print(bias)
test2 = np.hstack((test2,bias))

test2 = nnScript.sigmoid(np.dot(test2,test3.T))
print(test2)
print(np.argmax(test2, axis=1))


#np.delete(test[0],[0, 1])
'''

'''index_list=[]
for index in range(test[0].shape[0]):
    count = 0
    for vector in test:
        count+=vector[index]
    if count/test.shape[0] == test[0][index]:
        index_list.append(index)
test = np.delete(test, index_list,axis=1)
print(test)'''


'''for vector in test:
    np.delete(vector,index_list)
print(index_list)

print(test)'''
#rand = np.random.permutation(3)
#print(type(rand))




'''print(train_data.shape)
print(train_label.shape)'''


#print(mat.values()[0])

#pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
#print(pickle_obj.values())

