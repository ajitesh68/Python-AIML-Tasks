import numpy as np 


np.random.seed(42)
marks = np.random.randint(0,100,size=(100,3))

attendance = np.random.uniform(40,100,size=(100,1))

data = np.hstack((marks,attendance))



#subject statistics 

subject_means = np.mean(data[:,:3],axis=0)
subject_median = np.median(data[:,:3],axis=0)
subject_stds = np.std(data[:,:3],axis=0) 

print("Subject means:",subject_means)
print("Subject median:",subject_median)
print("Subject stds:",subject_stds) 

min_vals = np.min(data[:,:3],axis=0)
max_vals = np.max(data[:,:3],axis=0)

normalized = (data[:,:3]-min_vals)/(max_vals-min_vals)

mean_vals = np.mean(data[:,:3],axis=0)
std_vals = np.std(data[:,:3],axis=0)

standardized = (data[:,:3]-mean_vals)/std_vals
print("Normalized (first 5 rows):\n", normalized[:5])
print("Standardized (first 5 rows):\n", standardized[:5])


#one hot encoding 
grades = np.random.choice(['A','B','C'], size=100)

uniques = np.unique(grades)
one_hot = np.zeros((len(grades),len(uniques)))

for i,grade in enumerate(uniques):
    one_hot[:,i] = (grade==grades).astype(int)


print("Unique categories:", uniques)
print("One-hot shape:", one_hot.shape)
print("First 5 rows:\n", one_hot[:5])


#filtering and reshaping 
high_math = data[:,0]>80
print("Students scoring above 80 in math:",np.sum(high_math))
print("First 3 such students:\n",high_math[:3])

math_marks = data[:,0]
math_column = math_marks.reshape(-1,1)
print(f"Original shape: {math_marks.shape}, Reshaped: {math_column.shape}")
