import numpy as np

# 3 lists containing 4 items. Shape of this is (2,4). matrix of 3 rows and 4 columns
# rand() produces normal randown as compared to randn() which produces  gaussian distribution
arr = np.random.rand(3,4) 

# empty array, same as  zeros/fake_vals. First argument is the shape hence double bracs. second is dtype

# 3D Array of 2 lists each of those lists contains 3 items,
    # and each of those items contains 4 items.
    # Dimension 0 has 2 items,  dim1 has 3 items and dimension 2 has 4.
arr = np.random.rand(2,3,4)     # numbers between 0 and 1.
arr.size                        # total items in the array. 2 * 3 * 4. In this case 24.
arr.ndim                        # total dimensions in the array.  really a len(arr.shape). In this case 3 since its a 3D array.
arr.shape                       # the shape(2,3,4).
np.random.randint(1,5,size=(2,4))      # This one returns rand ints between(inclusive) 1 and 4 with shape 2,4

arr.reshape(12,2)      # 12 rows 2 columns. arr.size should be rows*column. in this case 24

# this makes a 
    # reshape it to 2 rows and 3 columns. simply 2 times 3
rows = 2
columns = 3
arr = np.arange(-1,3,0.25)      # Vector of numbers between -1 to 3 stepping by 0.25,
# calling an index is calling the dimension of the array
arr = arr[:(rows*columns)]      # trim the vector upto a [:i]
arr.reshape(rows,columns)       # since there are 6 items in the vector, row*column should be 6

# 30 equidistant numbers between -5 and 5. reshape it such that axis
arr = np.linspace(-5,5,30).reshape((5,2,3))    
# move columns to rows.
# args esentially act as index/dimension of the array. (0,row,column) becomes (0,column,row).
arr.transpose(0,2,1)    

# find a value within an array: in this case find index of the max value
arr = np.array([3,2,1,3,2])
np.where(arr == np.max(arr))

# return index of the maximum value.
np.argmax(arr)
# keeps dimensions intact:
np.argmax(arr,axis=1,keepdims=True)

x_vals = arr[:,0]         # says all rows and 0th column
y_vals = arr[:,1]         # all rows, 1st column
x_values, y_values = arr[:,0], arr[:,1]     # : is all rows and 0th column; 1st column

# switching columns around
arr = np.array([[1, 2], [3, 4], [5, 6]])
arr[:, [0, 1]] = arr[:, [1, 0]]

# flip array left to right:
arr[::-1]
np.flip(arr)

# flip array upside down.
    # take mean, subtract mean from the data
    # then do -1 times that array that makes positive numbers -ve and -ve nums positive
-1 * (arr - np.mean(arr))

# normalize
aa=arr
aax = aa/np.sum(aa,axis=0,keepdims=True)      # Normalize using numpy np
