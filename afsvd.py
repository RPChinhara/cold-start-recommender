import numpy as np
import pandas as pd

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
def ratings_based_similarity(UImatrix: pd.DataFrame, item1, item2):
    item1_ratings = UImatrix[UImatrix['movie id'] == item1]
    item2_ratings = UImatrix[UImatrix['movie id'] == item2]

    common_users = np.intersect1d(item1_ratings['user id'], item2_ratings['user id'])
    if len(common_users) < 2:
        return 0
    
    item1_ratings = item1_ratings.loc[item1_ratings['user id'].isin(common_users)]
    item2_ratings = item2_ratings.loc[item2_ratings['user id'].isin(common_users)]

    item1_ratings = item1_ratings.groupby(['user id', 'movie id'])['rating'].mean().reset_index()
    item2_ratings = item2_ratings.groupby(['user id', 'movie id'])['rating'].mean().reset_index()

    item1_ratings = item1_ratings['rating'].to_numpy()
    item2_ratings = item2_ratings['rating'].to_numpy()

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
    old_item_attributes = np.array(IAmatrix['old_items'][old_item])
    new_item_attributes = np.array(IAmatrix['new_items'][new_item])

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
def rating_prediction(IAmatrix, UImatrix: pd.DataFrame, user, item, neighbours, minimum_timestamp):
    user_ratings = UImatrix[UImatrix['user id'] == user]

    ru_mean = user_ratings['rating'].mean()
    numerator = 0
    denominator = 0

    for j in neighbours:
        r_uj = UImatrix.loc[(UImatrix['movie id'] == j) & (UImatrix['user id'] == user), 'rating']
        sim_ratings = ratings_based_similarity(UImatrix, item, j)
        sim_attributes = attribute_based_similarity(IAmatrix, j, item)
        time_penalty_factor = time_penalty(UImatrix[UImatrix['movie id'] == j]['timestamp'], UImatrix[UImatrix['movie id'] == item]['timestamp'], min_timestamp=minimum_timestamp)

        numerator += time_penalty_factor * sim_attributes * (r_uj - ru_mean)
        denominator += sim_ratings

    if denominator != 0:
        user_preferences = numerator / denominator
        r_hat_ui = ru_mean + user_preferences
        return r_hat_ui, user_preferences
    return ru_mean, 0

def attribute_based_knn(IAMatrix, UIMatrix: pd.DataFrame, k, users):
    predicted_UIMatrix = UIMatrix.copy()
    predicted_UIMatrix['user preferences'] = 0
    minimum_timestamp = predicted_UIMatrix['timestamp'].min()

    for new_item in IAMatrix['new_items'].keys():
        similar_list = []
        for old_item in IAMatrix['old_items'].keys():
            similar_list.append((old_item, attribute_based_similarity(IAMatrix, old_item, new_item)))

        similar_list.sort(key=lambda x: x[1], reverse=True)
        neighbours = [item[0] for item in similar_list[:k]]

        for user_id in users:
            predicted_rating = rating_prediction(IAMatrix, predicted_UIMatrix, user_id, new_item, neighbours, minimum_timestamp)
            if isinstance(predicted_rating[0], float):
                entry_dict = {'user id': [user_id], 'movie id': [new_item], 'rating': [predicted_rating[0]], 'user preferences': [predicted_rating[1]]}
                entry = pd.DataFrame(entry_dict)
                predicted_UIMatrix = pd.concat([predicted_UIMatrix, entry], ignore_index=True)            

    return predicted_UIMatrix

def matrix_factorization(train_matrix: np.ndarray, learning_rate=0.01, num_iterations=100, regularization_param=0.02, num_factors=21, eta=1):
    num_users, num_items = train_matrix.shape

    user_factors = np.random.randn(num_users, num_factors)
    item_factors = np.random.randn(num_items, num_factors)

    u_mean = np.mean(train_matrix)
    bu = 0

    while num_iterations > 0:
        for u in range(num_users):
            for i in range(num_items):
                if train_matrix[u][i] > 0:
                    # Calculate the predicted rating
                    predicted_rating = np.dot(user_factors[u], item_factors[i]) + bu + u_mean

                    # Calculate the prediction error
                    prediction_error = train_matrix[u][i] - predicted_rating - bu - u_mean

                    # Update user and item latent factors
                    for k in range(num_factors):
                        bu += learning_rate * (prediction_error - (regularization_param * bu))
                        user_factors[u][k] += learning_rate * (prediction_error * item_factors[i][k] - regularization_param * user_factors[u][k])
                        item_factors[i][k] += learning_rate * (prediction_error * user_factors[u][k] - regularization_param * item_factors[i][k])
                        learning_rate *= eta

        num_iterations -= 1

    return user_factors, item_factors