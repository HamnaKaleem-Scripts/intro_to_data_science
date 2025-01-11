def function_1(feedback):
    user = {}
    for user_id, details in feedback.items():
        if details['rating'] >= 4:
            user[user_id] = details['rating']
    return user

def function_2(feedback):
    sorted = sorted(feedback.items(), key=lambda item: item[1]['rating'], reverse=True)
    top = {}
    for user_id, detail in sorted[:5]: 
        top[user_id] = detail['rating']
    return top

def function_3(feedback_list):
    combined_feedback = {}
    for i in feedback_list:
        for user_id, detail in i.items():
            if user_id in combined_feedback:
                if detail['rating'] > combined_feedback[user_id]['rating']:
                    combined_feedback[user_id]['rating'] = detail['rating']
                combined_feedback[user_id]['comments'].append(detail['comments'])
            else:
                combined_feedback[user_id] = {
                    'rating': detail['rating'],
                    'comments': [detail['comments']]
                }
    return combined_feedback


def function_4(feedback):
    user = {}
    for user_id, details in feedback.items():
        if details['rating'] > 3:
            user[user_id] = details['rating']
    return user



user_feedback = {
    1: {'rating': 5, 'comments': 'Excellent!'},
    2: {'rating': 3, 'comments': 'Good.'},
    3: {'rating': 4, 'comments': 'Very good!'},
    4: {'rating': 2, 'comments': 'Not satisfied.'},
    5: {'rating': 4, 'comments': 'Quite nice!'},
    6: {'rating': 5, 'comments': 'Outstanding!'},
}
user = function_1(user_feedback)
print("Users with ratings 4 or higher:", user)
top_users = function_2(user_feedback)
print("Top 5 users by rating:", top_users)
feedback_dicts = [
    {
        1: {'rating': 5, 'comments': 'Excellent!'},
        2: {'rating': 3, 'comments': 'Good.'},
    },
    {
        2: {'rating': 4, 'comments': 'Very good!'},
        3: {'rating': 5, 'comments': 'Great service!'},
    },
]
combined_feedback = function_3(feedback_dicts)
print("Combined feedback:", combined_feedback)
high_ratings = function_4(user_feedback)
print("Users with ratings greater than 3:", high_ratings)
