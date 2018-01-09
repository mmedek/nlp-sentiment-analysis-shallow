import numpy as np

def average_weight(results_matrix, weights = []):
        
    if len(weights) == 0:
        print('No weights are used. Default weight equal 1\'s will be used')
        weights = np.ones(len(results_matrix))
    
    if len(weights) != len(results_matrix):
        print('Size of matrix with results is different that size of weights matrix. Return')
        return
    
    classes = np.unique(results_matrix)
    print('Find ' + str(len(classes)) + ' unique classes')
    
    number_result_vectors = len(results_matrix)
    results_in_vector = len(results_matrix[0])
    print('Different results vectors: ' + str(number_result_vectors))
    print('Number of values in each result vector: ' + str(results_in_vector))

    index = 0
    often_arr = []
    result_average_weights_arr = []
    for x in range(results_in_vector):
        often_arr = np.zeros(len(classes))
        for y in range(number_result_vectors):
            index = np.where(results_matrix[y][x] == classes)[0][0]
            often_arr[index] += 1 * weights[y]
        result_average_weights_arr.append(classes[np.argmax(often_arr)])

    return result_average_weights_arr