#!/usr/bin/env python
# coding: utf-8

# # Numpy - Learn Data Analysis From Scratch

# In[1]:


import numpy as np


# In[2]:


list1 = [1,2,3,4,5]


# In[4]:


numpyArray = np.array(list1,dtype=np.int16)


# In[5]:


print(numpyArray)


# In[6]:


list2 = [6,7,8,9,10]


# In[9]:


numpyArray2d = np.array([list1,list2],dtype=np.int16)


# In[10]:


print(numpyArray2d)


# In[11]:


print(type(numpyArray2d))


# # Arrange Method to Generate Numpy Array

# In[15]:


arangeArray = np.arange(3,50)


# In[16]:


print(arangeArray)


# In[19]:


zeroArray = np.zeros((3,3))


# In[20]:


print(zeroArray)


# In[21]:


oneArray = np.ones((4,4))


# In[22]:


print(oneArray)


# In[27]:


Iarray = np.eye(3)


# In[28]:


print(Iarray)


# In[33]:


eSpaceArray = np.linspace(1,5,20)


# In[34]:


print(eSpaceArray)


# In[35]:


import matplotlib.pyplot as pp


# In[36]:


pp.plot(eSpaceArray)


# # Numpy Random Method

# In[39]:


ndisArray = np.random.rand(3,2)


# In[40]:


print(ndisArray)


# In[41]:


pp.plot(ndisArray)


# In[46]:


intArray = np.random.randint(30,50,size=(2,2))


# In[47]:


print(intArray)


# In[49]:


pp.plot(intArray)


# # What is Dimension Array

# In[54]:


arr1d = [12,15,16]


# In[55]:


print(arr1d[0])


# In[57]:


arr2d = np.random.randint(30,50,size=(2,2))


# In[58]:


print(arr2d)


# In[59]:


print(arr2d[0][0])


# In[60]:


print(arr2d[1][0])


# In[64]:


arr1darr=np.array(arr1d,dtype=np.int16)


# In[66]:


print(arr1darr.ndim)


# In[67]:


arr2darr=np.array(arr2d,dtype=np.int16)


# In[68]:


print(arr2darr.ndim)


# # What is Shape

# In[69]:


arr1darr.shape


# In[70]:


arr2darr.shape


# # What is Strides

# In[72]:


sArray = np.array([[1,2,3],[4,5,6]], dtype=np.int16)


# 8bits=1byte,
# 16/8=2bytes

# In[74]:


sArray.strides


# # UPCASTING & DOWNCASTING

# In[75]:


intArray = np.arange(6)


# In[76]:


print(intArray)


# In[77]:


intArray.dtype


# In[78]:


floatArray = np.random.uniform(0,5,5)


# In[79]:


print(floatArray)


# In[80]:


floatArray.dtype


# In[81]:


concatArray = np.concatenate((floatArray,intArray))


# In[82]:


print(concatArray)


# In[83]:


concatArray.dtype


# In[84]:


downCast = concatArray.astype('int64')


# In[85]:


print(downCast)


# In[86]:


downCast.dtype


# # Slicing & Indexing

# In[89]:


arr1d=np.arange(10)


# In[90]:


print(arr1d)


# # Extract Even Numbers from arr1d

# In[97]:


even=arr1d[0:10:2]


# In[98]:


print(even)


# # Indexing

# In[100]:


arr1d[3]


# In[101]:


arr1d[0]


# In[102]:


arr1d[-1]


# In[103]:


arr1d[::-1]


# In[104]:


arr1d[-2]


# In[105]:


arr1d[-3]


# # Multidimensional Slicing and Indexing

# In[106]:


arr2d = np.random.randint(30,50,size=(2,2))


# In[109]:


print(arr2d)


# In[110]:


arr2d.ndim


# In[113]:


arr2d[1:]


# In[114]:


arr2d[0:2,0:1]


# In[115]:


arr2d[0][0]


# In[116]:


arr2d[-1][0]


# In[117]:


arr2d[-1][-1]


# # Numpy Array Operations

# # np.newaxis

# In[119]:


list1 = [1,2,3,4]
numpy_arr = np.array(list1)


# In[120]:


type(numpy_arr)


# In[121]:


type(list1)


# In[122]:


numpy_arr.shape


# In[124]:


row_vector = numpy_arr[:,np.newaxis]


# In[126]:


row_vector.shape


# In[127]:


print(row_vector)


# In[129]:


col_vector = numpy_arr[np.newaxis,:]


# In[130]:


col_vector.shape


# In[131]:


print(col_vector)


# # Reshape

# In[132]:


arr=np.arange(15)


# In[133]:


print(arr)


# In[134]:


arr2d=arr.reshape(3,5)


# In[135]:


print(arr2d)


# In[136]:


arr2d.ndim


# # Arithmetic Operation On Numpy Array

# In[139]:


arr1 = np.random.randint(1,10,size=(3,3))


# In[140]:


print(arr1)


# In[141]:


arr2 = np.random.randint(10,20,size=(3,3))


# In[142]:


print(arr2)


# In[143]:


np.add(arr1,arr2)


# In[144]:


np.subtract(arr1,arr2)


# In[145]:


np.multiply(arr1,arr2)


# In[146]:


np.divide(arr1,arr2)


# In[147]:


np.dot(arr1,arr2)


# x1*x2-5*x2=30

# In[148]:


ansArr = arr1*arr2-5*arr2 + 30


# In[149]:


print(ansArr)


# # Broadcasting

# In[153]:


a=np.arange(5)
print(a)
print(a.size)
print(a.shape)


# In[154]:


b=2


# In[155]:


np.add(a,b)


# [2,2,2,2,2]
# 
# [a0+b0,a1+b1,a2+b2...]

# In[166]:


a=np.random.randint(1,10,size=(2,2))
print(a.shape)
print(a.size)


# In[167]:


b=np.random.randint(11,30,size=(1,2))
print(b.shape)
print(b.size)


# In[165]:


np.add(a,b)


# # VECTORIZATION

# In[174]:


a = np.arange(0,5)
print(a)


# In[177]:


b = np.arange(5,10)
print(b)


# In[183]:


res=[]
for i in range(len(a)):
    res.append(a[i]+b[i])
get_ipython().run_line_magic('timeit', 'np.array(res)')


# In[184]:


get_ipython().run_line_magic('timeit', 'a + b')


# In[185]:


a * b


# In[186]:


a / b


# In[188]:


c = [0,1,2,3,4]
d = [5,6,7,8,9]
c + d


# In[ ]:




