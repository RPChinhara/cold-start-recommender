import numpy as np
from math import exp

def learnBPR(feedback_matrix, A, items_vectors, bias_vector, num_iterations=1000, alpha = 0.01, alpha_b = 0.01, lambda_u = 0.1, lambda_b = 0.01):
    user_latent_vectors = np.zeros((len(feedback_matrix), items_vectors.shape[1]))
    
    for _ in range(num_iterations):
        i, p, q = A[np.random.randint(len(A))]
        x_hat_ipq = feedback_matrix[i][p] - feedback_matrix[i][q]

        exp_factor = exp(-x_hat_ipq) / (1 + exp(-x_hat_ipq))

        for k in range(items_vectors.shape[1]):
            user_latent_vectors[i][k] = user_latent_vectors[i][k] + alpha * (exp_factor * (items_vectors[p][k] - items_vectors[q][k]) - lambda_u * user_latent_vectors[i][k])
            
            items_vectors[p][k] = items_vectors[p][k] + alpha * (exp_factor * user_latent_vectors[i][k] - lambda_u * items_vectors[p][k])

            items_vectors[q][k] = items_vectors[q][k] + alpha * (exp_factor * user_latent_vectors[i][k] - lambda_u * items_vectors[q][k])

            bias_vector[p] = bias_vector[p] + alpha_b * (exp_factor - lambda_b * bias_vector[p])

            bias_vector[q] = bias_vector[q] - alpha_b * (exp_factor + lambda_b * bias_vector[q])

    return user_latent_vectors