import matplotlib.pyplot as plt
import numpy as np

def iteration_error_plot(model_error_dict):
    """
    @args: 
    model_error_dict: dict {'model_name': [model_error_list, iteration]} indicates training error according to each epoch
    """

    # Compute the smoothed values of the error using a rolling mean

    batch_size = 64

    for model_name, error_log in model_error_dict.items():
        error_list, iteration = np.array(error_log[0]), np.array(error_log[1])
        for i in range(0, error_list.shape[0], batch_size):
            error_list[i: (i + batch_size)] = np.mean(error_list[i: (i + batch_size)] )        

        # Replace the original array with the means array
        plt.plot(iteration, error_list, label=model_name)
       

    # add labels and title
    plt.xlabel('Number of Iterations')
    plt.ylabel('Training Error')
    plt.title('Training Error vs. Number of Iterations')
    plt.legend()

    # display the plot
    plt.show()

def predictive_factor_eval_plot(factor_arr, hr_dict, ndcg_dict, n):
    """
    @args: 
    factor_arr: list of predictive factors of NCF model 
    hr_dict: dict {'model_name': hr_score} of hit rate score according to factor
    ndcg_dict: dict {'model_name': ndcg_score} of ndcg_arr score according to factor
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))
    factor_arr = np.array(factor_arr).astype(str)

    for model_name, hr_score in hr_dict.items():
        ax1.plot(factor_arr, hr_score, label=model_name)
        ax1.scatter(factor_arr, hr_score)

    for model_name, ndcg_score in ndcg_dict.items():
        ax2.plot(factor_arr, ndcg_score, label=model_name)
        ax2.scatter(factor_arr, ndcg_score)

    # add labels and title
    ax1.set_xlabel('Number of Predictive Factors')
    ax1.set_ylabel('HR@{n}'.format(n=n))
    
    ax2.set_xlabel('Number of Predictive Factors')
    ax2.set_ylabel('NDCG@{n}'.format(n=n))
    plt.legend()
    
    # display the plot
    plt.show()


def k_eval_plot(k_arr, hr_dict, ndcg_dict, n=10):
    """
    @args: 
    k_arr: list of predictive factors of NCF model 
    hr_dict: dict {'model_name': hr_score} of hit rate score according to factor
    ndcg_dict: dict {'model_name': ndcg_score} of ndcg_arr score according to factor
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))
    

    for model_name, hr_score in hr_dict.items():
        ax1.plot(k_arr, hr_score, label=model_name)
        ax1.scatter(k_arr, hr_score)

    for model_name, ndcg_score in ndcg_dict.items():
        ax2.plot(k_arr, ndcg_score, label=model_name)
        ax2.scatter(k_arr, ndcg_score)

    # add labels and title
    ax1.set_xlabel('K')
    ax1.set_ylabel('HR')
    ax1.set_xticks(k_arr)

    ax2.set_xlabel('K')
    ax2.set_ylabel('NDCG')
    ax2.set_xticks(k_arr)
    
    plt.legend()
    
    # display the plot
    plt.show()

if __name__ == '__main__':
        
    # Toy example

    # example training error values
    training_error = {'test_model 1': [[0.2, 0.15, 0.1, 0.08, 0.06, 0.05], [1, 2, 3, 4, 5, 6]], 'test_model 2': [[0.5, 0.5, 0.4, 0.18, 0.16, 0.1], [1, 2, 3, 4, 5, 6]]}

    iteration_error_plot(training_error)

    hr_dict = {'test_model 1': [0.2, 0.15, 0.1, 0.09], 'test_model 2': [0.5, 0.5, 0.4, 0.18]}
    ndcg_dict = {'test_model 1': [0.2, 0.15, 0.1, 0.09], 'test_model 2': [0.5, 0.5, 0.4, 0.18]}
    factor_arr = [8, 16, 32, 64]
    predictive_factor_eval_plot(factor_arr, hr_dict, ndcg_dict, n=10)

    hr_dict = {'test_model 1': [0.2, 0.15, 0.1, 0.09], 'test_model 2': [0.5, 0.5, 0.4, 0.18]}
    ndcg_dict = {'test_model 1': [0.2, 0.15, 0.1, 0.09], 'test_model 2': [0.5, 0.5, 0.4, 0.18]}
    k_arr = [1, 2, 3, 4]
    k_eval_plot(k_arr, hr_dict, ndcg_dict)
