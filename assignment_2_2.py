def function_1(t):
    users = set()  

    for i in t:
        user_id = i[1]
        users.add(user_id) 
    
    return len(users) 

t = [
    (1, 101, 250.75, '2024-10-01 12:00:00'),
    (2, 102, 150.00, '2024-10-01 12:05:00'),
    (3, 103, 320.50, '2024-10-01 12:10:00'),
    (4, 101, 180.00, '2024-10-01 12:15:00'), 
    (5, 104, 500.00, '2024-10-01 12:20:00'), 
]

unique_user = function_1(t)
print("Total number of unique users:", unique_user)




def function_2(t):
    amount = max(transactions, key=lambda transaction: transaction[2])

    transaction_ids = [] 
    user_ids = []  

    for i in t:        
        transaction_ids.append(i[0])
        user_ids.append(i[1])

    return amount, transaction_ids, user_ids

transactions = [
    (1, 101, 250.75, '2024-10-01 12:00:00'),
    (2, 102, 150.00, '2024-10-01 12:05:00'),
    (3, 103, 320.50, '2024-10-01 12:10:00'),
    (4, 101, 180.00, '2024-10-01 12:15:00'),
    (5, 104, 500.00, '2024-10-01 12:20:00'),
]

highest_transaction, transaction_ids, user_ids = function_2(transactions)

print("Transaction with the highest amount:", highest_transaction)
print("Transaction IDs:", transaction_ids)
print("User IDs:", user_ids)
