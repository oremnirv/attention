from helpers import masks 
import tensorflow as tf
import numpy as np


def evaluate(model, pos, tar, sample = True):
    '''
    Run a forward pass of the network
    ------------------
    Parameters:
    model: trained instace of GPT decoder class
    pos: position tensor with at least len(tar) + 1 values
    tar: targets tensor
    pos_mask: position mask tensor to hide unseen positions from current prediction 
    ------------------
    Returns:
    pred (tf tensor float64): the prediction of the next location in the sequence 
    pred_log_sig (tf tensor float64)
    
    '''
    combined_mask_pos = masks.create_masks(pos)
    pred = model(pos, tar, False, combined_mask_pos[:, 1:, :-1])
    if sample: 
        sample_y = np.random.normal(pred[ -1, 0], np.exp(pred[ -1, 1]))
    else: 
        sample_y = pred[:, 0]

    return pred[:, 0], pred[:, 1], sample_y 



def inference(model, em_te, tar, num_steps = 1, sample = True):
    '''
    how many steps to infer -- this could be used both for interpolation and extrapolation 
    ------------------
    Parameters:
    pos (2D np array): (n + num_steps) positions 
    tar (2D np array): n targets 
    num_steps (int): how many inference steps are required
    ------------------
    Returns:
    pred (tf.tensor float64): the predictions for all timestamps up to n + num_steps  
    pred_log_sig
    '''
    n = tar.shape[1]
    temp_pos = em_te[:, :(n + 1)]
    pred, pred_log_sig, sample_y = evaluate(model, temp_pos, tar)
    tar = tf.concat((tar, tf.reshape(sample_y, [1, 1])), axis = 1)
    if num_steps > 1:
        model, em_te, tar = inference(model, em_te, tar, num_steps - 1)
    
    return model, em_te, tar

def main():
    samples = np.zeros((50, 600))
    for sample in range(50):
         _, _, samples[sample, :] = infer.inference(decoder, pos = batch_pos_tr[1, :600].reshape(1, -1), tar = batch_tar_tr[1, :50].reshape(1, -1), num_steps = 550)

    samples[:, :50] = batch_tar_tr[1, :50]
    plt.style.use( 'ggplot')
    sorted_arr = np.argsort(batch_pos_tr[1, :])
    for i in range(4, 5):
        plt.plot(batch_pos_tr[1, sorted_arr], samples[i, sorted_arr], 'lightsteelblue', alpha = 0.6, zorder = -1)
    plt.plot(batch_pos_tr[1, sorted_arr], batch_tar_tr[1, sorted_arr], 'black')
    plt.scatter(batch_pos_tr[1, :50], batch_tar_tr[1, :50], c = 'black', marker = "o", zorder = 1, s= 25)
    plt.show()

    extrapo = True
    if extrapo:
        x = np.load('/Users/omernivron/Downloads/GPT_data_goldstandard/x_extra.npy')
        y = np.load('/Users/omernivron/Downloads/GPT_data_goldstandard/y_extra.npy')
    else:
        x = np.load('/Users/omernivron/Downloads/GPT_data_goldstandard/x_interpol.npy')
        y = np.load('/Users/omernivron/Downloads/GPT_data_goldstandard/y_interpol.npy')

    mse_metric = 0; r_sq_metric = 0; kuee_metric = 0;
    μ = []; σ = []
    m = int(x.shape[0] / 10)
    y_mean = np.mean(y[:m, :40])
    y_te = y[:m, 40]
    for j in range(0, m):
        x_tr = x[j, :41].reshape(1, -1)
        y_tr = y[j, :40].reshape(1, -1)
        μ_te = infer.inference(decoder, x_tr, y_tr)
    #     μ_te, log_σ_te = infer.inference(decoder, x_tr, y_tr, mh=True)


        μ.append(μ_te[0][-1].numpy()); 
    #     σ.append(log_σ_te[-1])
    #     kuee_metric += metrics.KUEE(y_te[j], μ_te[-1], np.exp(log_σ_te[-1]))
    #     if (j % 400 == 0): 
    #         print('J: ', j)
    #         axes = plt.gca()
    #         axes.set_ylim([-2, 2])
    #         plt.scatter(x_tr[:, :-1], y_tr, c = 'black')
    #         plt.scatter(x_tr[:, 1:], μ_te, c='navy')
    #         plt.scatter(x_tr[:, -1], y_te[j], c='purple')
    #         plt.scatter(x_tr[:, -1], μ_te[-1], c='red')
    # #         plt.errorbar(x = x_tr[:, 40], y = (μ_te[-1]), yerr = 2 * np.exp(log_σ_te[-1]), fmt='o', ecolor='g', capthick=2)


            
            
    # #         plt.fill_between(x_tr[:, 1:].squeeze(), μ_te -2 * np.exp(log_σ_te), μ_te  + 2 * np.exp(log_σ_te), alpha=.2)

    #         plt.show()
        
    mse_metric = metrics.mse(y_te, μ) 
    r_sq_metric = metrics.r_squared(y_te, μ, y_mean)  
    mse_metric *= (1 / m)
    # kuee_metric *= (1 / m)




if __name__ == '__main__':
    main()