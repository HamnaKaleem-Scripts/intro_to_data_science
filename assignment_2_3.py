def page_visitors(A, B, C):
    users_A_and_b = A.intersection(B)
    users_A_or_c = A.symmetric_difference(C)
    return users_A_and_b, users_A_or_c

def update_A(A, new_id):
    A.update(new_id)


def remove_from_B(B, user_id):
    B.difference_update(user_id)



A_visitors = {"user1", "user2", "user3", "user4"}
B_visitors = {"user3", "user4", "user5"}
C_visitors = {"user2", "user6", "user7"}

c, e = page_visitors(A_visitors, B_visitors, C_visitors)

print("Users who visited both Page A and Page B:", c)
print("Users who visited either Page A or Page C but not both:", e)

a = {"user5", "user8"}
update_A(A_visitors,a)
print("Updated visitors for Page A:", A_visitors)

b= ["user4"]
remove_from_B(B_visitors, b)
print("Updated visitors for Page B:", B_visitors)
