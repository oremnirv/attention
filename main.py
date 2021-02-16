import os
import sys
sys.path.append("..")
from model import experimental_model, experimental2d_model, grapher
from data import batch_creator, loader
from helpers import helpers, plotter, metrics
import tensorflow as tf
import time
import numpy as np


kernel = input()
d = True
save_dir = os.path.expanduser('/home/azureuser/Downloads/GPT_' + kernel)
data = loader.load_data(kernel, size = 1, rewrite = False, diff_x= True, noise = False, d = True, ordered = True)
train_step, test_step, loss_object, train_loss, test_loss, m_tr, m_te = grapher.build_graph()
EPOCHS = 75; batch_s  = 64; run = 515; step = 0; train_steps = 35000; heads = 32; ℯ = 512; context = 10
l = [256, 256, 64, 32]
name_comp = 'run_' + str(run) 
logdir = save_dir + '/logs/' + name_comp
writer = tf.summary.create_file_writer(logdir)
folder = save_dir + '/ckpt/check_' + name_comp
#     lr_fn = tf.optimizers.schedules.PolynomialDecay(9e-3, train_steps, 1e-7, 2)
optimizer_c = tf.keras.optimizers.Adam(3e-4)
ℯ, l1, _, l2, l3 = helpers.load_spec(folder, ℯ, l, context,  d = True)
helpers.mkdir(folder)
if d:
    decoder = experimental2d_model.Decoder(ℯ, l1, l2, l3, num_heads = heads)
else:
    decoder = experimental_model.Decoder(ℯ, l1, l2, l3, num_heads = heads)
tf.random.set_seed(443)  
num_batches = int(data[5].shape[0] / batch_s) if d else int(data[4].shape[0] / batch_s)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer = optimizer_c, net = decoder)
manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")
with writer.as_default():
    for epoch in range(EPOCHS):
        
        start = time.time()

        for batch_n in range(num_batches):
            m_tr.reset_states(); train_loss.reset_states()
            if d:
#                     b_data = batch_creator.create_batch(em_x = data[-4], x = data[1], y = data[-3],  em_2 = data[-1], batch_s=64, d=d)
                b_data, c = batch_creator.create_batch_2d(em_x = data[-1], x = data[2], y = data[0],  em_2 = data[3], batch_s=64)
                # print('c: ', c)
                if type(c) is list:
                    cols = [np.arange(c[i], b_data[2].shape[1] -1, 1) for i in range(len(c))]
                    cc = np.concatenate(cols, axis=0 )
                    rows = [np.repeat(i, len(m)) for i, m in enumerate(cols)]
                    r = np.concatenate(rows, axis=0)
                    to_gather = np.concatenate((r.reshape(-1, 1), cc.reshape(-1,1)), 1)
                else:
                    to_gather = None
                pred, pred_log, weights, names, shapes = train_step(decoder, optimizer_c, train_loss, m_tr, b_data[2], b_data[0], d = True, x2 = b_data[3], to_gather=to_gather, context_p = c)
#                     pred, pred_log, weights, names, shapes = train_step(decoder, optimizer_c, train_loss, m_tr, b_data[2], b_data[0], d = True, x2 = b_data[3], to_gather=None, context_p = context)
            else:
                b_data = batch_creator.create_batch(em_x = data[3], x = data[0], y = data[4], batch_s=64, d=d)
                pred, pred_log, weights, names, shapes = train_step(decoder, optimizer_c, train_loss, m_tr, x = b_data[2], y = b_data[0], context_p = context)
            if (epoch == 0) & (batch_n == 0): helpers.write_speci(folder, names, shapes, context)
            if batch_n % 300 == 0:
                m_te.reset_states(); test_loss.reset_states()
                if d:
                    pass
                    # pred_te, pred_log_te = test_step(decoder, test_loss, m_te, x_te = data[3][:500, :], y_te = data[6][:500, :], d = True, x2_te = data[0][:500, :])
                else:
                    pred_te, pred_log_te = test_step(decoder, test_loss, m_te, x_te = data[2][:500, :], y_te = data[5][:500, :], context_p = context)
                helpers.print_progress(epoch, batch_n, train_loss.result(), test_loss.result(), m_tr.result(), m_te.result())
                helpers.tf_summaries(run, step, train_loss.result(), test_loss.result(), m_tr.result(), m_te.result(), weights, names)
                print('learning rate is {}'.format(optimizer_c._decayed_lr('float32').numpy()))
                if d:
                    m0, m1 = metrics.r_sq_2d(b_data[0][:, 1:], pred.numpy(), b_data[3][:, 1:], context_p = context)
                    # m0_te, m1_te = metrics.r_sq_2d(data[-2][:500, 1:], pred_te.numpy(), data[0][:500, 1:], context_p = context)
                    print('r squared training, series 0: {}, series 1: {}'.format(m0, m1))
                    # print('r squared testing, series 0: {}, series 1: {}'.format(m0_te, m1_te))

                else: 
                    print('r squared training: ', metrics.r_squared(m_tr.result(), b_data[0][:, (context + 1):]))
                    print('r squared testing: ', metrics.r_squared(m_te.result(), data[5][:500, (context + 1):], batch_s = 500))


                manager.save()
            step += 1
            ckpt.step.assign_add(1)

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))