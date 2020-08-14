import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # sample size
    N = 1000
    x = np.linspace(0, 2*np.pi, N)

    # true data function
    y = 100*np.sin(x) + np.exp(x)/2 + 300

    # data with normal(0, sigma^2) distributed noise
    sigma = 10
    noise = np.random.randn(N) * sigma
    data = y + noise

    max_degree = 21

    diff = {} 

    r_matrix = np.ones((N, 1))
    for m in range(1, max_degree):
        # regression matrix, basically f(x) = x^m
        n_data = x.reshape(N, 1) ** m
        r_matrix = np.concatenate((r_matrix, n_data), axis=1)

        # magic equation
        w = np.dot(np.dot(np.linalg.inv((np.dot(r_matrix.T, r_matrix))), r_matrix.T), data)

        # w contains coefficients of predicted polynomial
        model = 0
        for k in range(len(w)):
            model = model + w[k] * x ** k

        # difference between data and predicted polynomial
        diff[m] = sum((data - model) ** 2) / N

        # plot: data and predicted polynomial
        ax = plt.subplot(4, 5, m)
        ax.plot(x, data, 'ro', markersize=2)
        ax.plot(x, model, linewidth=2)
        # ax.set_ylim(0, 700)
        ax.set_title(f'{m}-degree polynomial')

    plt.show()

    # another ax for difference
    diff_x = np.array(list(diff.keys()))
    diff_y = np.array(list(diff.values()))

    diff_ax = plt.subplot()
    diff_ax.plot(diff_x, diff_y)
    diff_ax.set_title('Difference')
    plt.show()
