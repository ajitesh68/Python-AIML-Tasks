import numpy as np 
np.random.seed(42)
arr = np.random.randint(0,23,size=(5,4)) 
print(arr)


row_sums = arr.sum(axis=1, keepdims=True)
print("\nRow sums:", row_sums)


normalized = arr/row_sums
print("\nNormalized:",normalized)


grades = np.random.choice(['A','B','C','D'], size=(100,3))


for col in range(3):
    unique_vals , counts = np.unique(grades[:,col],return_counts=True)

    max_index = counts.argmax()
    most_frequent = unique_vals[max_index]

    print(f"Column {col}: {most_frequent}") 



np.random.seed(42)
marks = np.random.randint(0,101,size=(100,3))
attendance = np.random.uniform(40,100,size=(100,1))
data = np.hstack((marks,attendance))

cond1 = data[:,0] > 70 
cond2 = data[:,3] > 75


mask = (cond1) & (cond2) 

filtered_data = data[mask]
if filtered_data.size > 0:
    avg_science = np.mean(filtered_data[:,1])
    print(f"Average Sience marks: {avg_science:.2f}")



math_marks = data[:,0]
new_marks = np.where(math_marks<35,0,math_marks)
print("Original (first 10):", math_marks[:10])
print("Corrected (first 10):", new_marks[:10])



science_marks = data[:,1].copy()
science_marks = np.where(data[:,0]<35,0,science_marks)
sorted_vals = np.sort(science_marks)
print("Top 3:", sorted_vals[-3:])