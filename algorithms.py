import numpy as np

def item_attribute_generation_matrix(i_old, i_new, item_attributes: dict, A_set: tuple):
    '''
    i_old is a set containing Old Items\n
    i_new is a set containg New Items\n
    item_attributes is a dictionary containg Attributes\n
    key: item_id, value: attribute
    '''

    IA_matrix_old = dict()
    IA_matrix_new = dict()

    for item_id in i_old:
        attributes = item_attributes.get(item_id, [])
        attribute_vector = tuple(1 if attr in attributes else 0 for attr in A_set)
        IA_matrix_old[item_id] = attribute_vector

    for item_id in i_new:
        attributes = item_attributes.get(item_id, [])
        attribute_vector = tuple(1 if attr in attributes else 0 for attr in A_set)
        IA_matrix_new[item_id] = attribute_vector

    IA_matrix = {
        "old_items": IA_matrix_old,
        "new_items": IA_matrix_new
    }
    
    return IA_matrix

# Eq. (1)
def ratings_based_similarity(UImatrix, item1, item2):
    item1_ratings = UImatrix[UImatrix['movie id'] == item1]['rating']
    item2_ratings = UImatrix[UImatrix['movie id'] == item2]['rating']

    item1_ratings_mean = np.mean(item1_ratings)
    item2_ratings_mean = np.mean(item2_ratings)

    numerator = np.sum((item1_ratings - item1_ratings_mean) * (item2_ratings - item1_ratings_mean))
    denominator_x = np.sqrt(np.sum((item1_ratings - item1_ratings_mean) ** 2))
    denominator_y = np.sqrt(np.sum((item2_ratings - item2_ratings_mean) ** 2))

    denominator = denominator_x * denominator_y

    if denominator != 0:
        pcc = numerator / denominator
        return pcc
    return 0

# Eq. (8)
def attribute_based_similarity(IAmatrix, old_item, new_item):
    old_item_attributes = IAmatrix['old_items'][old_item]
    new_item_attributes = IAmatrix['new_items'][new_item]

    old_item_attributes_mean = np.mean(old_item_attributes)
    new_item_attributes_mean = np.mean(new_item_attributes)

    denom1 = np.sum((old_item_attributes - old_item_attributes_mean) ** 2)
    denom2 = np.sum((new_item_attributes - new_item_attributes_mean) ** 2)

    denominator = np.sqrt(denom1) * np.sqrt(denom2)

    numerator = np.sum((old_item_attributes - old_item_attributes_mean) * (new_item_attributes - new_item_attributes_mean))

    if denominator != 0:
        similarity = numerator / denominator
        return similarity
    return 0

# Eq. (10)
def time_penalty(old_item_timestamp, new_item_timestamp, min_timestamp):
    time_penalty_factor = 1 - ((new_item_timestamp - old_item_timestamp) / (new_item_timestamp - min_timestamp))
    
    return time_penalty_factor

# Eq. (11)
def rating_prediction(IAmatrix, UImatrix, user, item, neighbours):
    user_ratings = UImatrix[UImatrix['user id'] == user]

    ru_mean = np.mean(user_ratings['rating'])
    numerator = 0
    denominator = 0

    for j in neighbours:
        r_uj = UImatrix[UImatrix['movie id'] == j][UImatrix['user id'] == user]['rating']
        sim_ratings = ratings_based_similarity(UImatrix, item, j)
        sim_attributes = attribute_based_similarity(IAmatrix, item, j)
        time_penalty_factor = time_penalty(UImatrix[UImatrix['movie id'] == j]['timestamp'], UImatrix[UImatrix['movie id'] == item]['timestamp'], min_timestamp=UImatrix['timestamp'].min())

        numerator += time_penalty_factor * sim_attributes * (r_uj - ru_mean)
        denominator += sim_ratings

    if denominator != 0:
        r_hat_ui = ru_mean + (numerator / denominator)
        return r_hat_ui
    return ru_mean

def user_preferences(IAmatrix, UImatrix, user, item, neighbours):
    user_ratings = UImatrix[UImatrix['user id'] == user]

    ru_mean = np.mean(user_ratings['rating'])
    numerator = 0
    denominator = 0

    for j in neighbours:
        r_uj = UImatrix.loc[(UImatrix['movie id'] == j) & (UImatrix['user id'] == user), 'rating']
        sim_ratings = ratings_based_similarity(UImatrix, item, j)
        sim_attributes = attribute_based_similarity(IAmatrix, j, item)
        time_penalty_factor = time_penalty(UImatrix[UImatrix['movie id'] == j]['timestamp'], UImatrix[UImatrix['movie id'] == item]['timestamp'], min_timestamp=UImatrix['timestamp'].min())

        numerator += time_penalty_factor * sim_attributes * (r_uj - ru_mean)
        denominator += sim_ratings

    if denominator != 0:
        user_preference = numerator / denominator
        return user_preference
    return 0

def update_predicted_UIMatrix(predicted_UImatrix, user_id, item_id, user_preference):
    matching_rows = predicted_UImatrix[(predicted_UImatrix['user id'] == user_id) & (predicted_UImatrix['movie id'] == item_id)]

    if len(matching_rows) == 1:
        predicted_UImatrix.loc[matching_rows.index, 'predicted rating'] = user_preference


def attribute_based_knn(IAMatrix, UIMatrix, k):
    predicted_UIMatrix = UIMatrix[UIMatrix['movie id'].isin(IAMatrix['new_items'].keys())].copy()
    predicted_UIMatrix = predicted_UIMatrix.drop(['rating'], axis=1)

    for new_item in IAMatrix['new_items'].keys():
        similar_list = []
        for old_item in IAMatrix['old_items'].keys():
            similar_list.append((old_item, attribute_based_similarity(IAMatrix, old_item, new_item)))

        similar_list.sort(key=lambda x: x[1], reverse=True)
        neighbours = [item[0] for item in similar_list[:k]]

        for user_id in predicted_UIMatrix['user id'].unique():
            UserPreference = user_preferences(IAMatrix, UIMatrix, user_id, new_item, neighbours)
            update_predicted_UIMatrix(predicted_UIMatrix, user_id, new_item, UserPreference)

    return predicted_UIMatrix.drop('timestamp', axis=1).reset_index(drop=True)