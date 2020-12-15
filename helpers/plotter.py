import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def sample_plot_w_training(positions, targets, predictions, num_context = 50, num_samples = 1, title = ''):
    real_x = positions
    real_y = targets
    samples = np.zeros((num_samples, len(real_x)))
    samples[0, :(num_context - 1)] = targets[:(num_context - 1)]
    samples[0, (num_context - 1):] = predictions[(num_context - 1):]
    sorted_arr = np.argsort(real_x)
    plt.plot(real_x[sorted_arr], real_y[sorted_arr], 'black')
    plt.scatter(real_x[:num_context], real_y[:num_context], c = 'black', marker = "o", zorder = 1, s= 25)
    plt.plot(real_x[sorted_arr], samples[0, sorted_arr], c = 'lightskyblue', alpha = 0.6)
    plt.title(title)
    plt.show()