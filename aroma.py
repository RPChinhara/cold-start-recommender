import numpy as np
from math import exp

def learnBPR(feedback_matrix, A, items_vectors, bias_vector, num_iterations=100, alpha = 0.01, alpha_b = 0.01, lambda_u = 0.1, lambda_b = 0.01):
    for _ in range(num_iterations):
        i, p, q = A[np.random.randint(len(A))]
        x_hat_ipq = feedback_matrix[i][p] - feedback_matrix[i][q]

        exp_factor = exp(-x_hat_ipq) / 1 + (exp(-x_hat_ipq))

        for k in range(1, items_vectors.shape[1]):
            feedback_matrix[i][k] = feedback_matrix[i][k] + alpha * (exp_factor * (items_vectors[p][k] - items_vectors[q][k]) - lambda_u * feedback_matrix[i][k])
            
            items_vectors[p][k] = items_vectors[p][k] + alpha * (exp_factor * feedback_matrix[i][k] - lambda_u * items_vectors[p][k])

            items_vectors[q][k] = items_vectors[q][k] + alpha * (exp_factor * feedback_matrix[i][k] - lambda_u * items_vectors[q][k])

            bias_vector[p] = bias_vector[p] + alpha_b * (exp_factor - lambda_b * bias_vector[p])

            bias_vector[q] = bias_vector[q] - alpha_b * (exp_factor + lambda_b * bias_vector[q])

    return feedback_matrix