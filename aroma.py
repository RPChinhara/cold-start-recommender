def learnBPR(user_vectors, items_vectors, num_iterations=10, alpha = 0.01, alpha_b = 0.01, lambda_u = 0.1, lambda_b = 0.01):
    for _ in range(num_iterations):
        i, p, q = 