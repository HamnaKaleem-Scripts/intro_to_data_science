def filter_names(users):
    names=[]
    for user in users:
        if user[2] > 30 and (user[3] == 'USA' or user[3] == 'Canada'):
            names.append(user[1]) 
    return names


def function2(users):
    users.sort(key=lambda user: user[2]) 
    old = users[-10:] 

    names = []  
    for i in users:
        name = i[1]  
        names.append(name)
    
    unique_names = set() 
    duplicates = set()  

    for name in names:
        if name in unique_names:
            duplicates.add(name) 
        else:
            unique_names.add(name)  

    return old, duplicates


users = [
    (1, "Alice", 28, "USA"),
    (2, "Bob", 34, "Canada"),
    (3, "Charlie", 40, "USA"),
    (4, "David", 32, "UK"),
    (5, "Alice", 29, "USA")    
]

old, duplicates = function2(users)

print("Top 10 Oldest Users:")
for i in old:
    print(i)

print("Duplicate Names:", duplicates)
 
