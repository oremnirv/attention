import sys
sys.path.append("..")
import tensorflow as tf
from helpers import masks
from model import losses


def build_graph():


	loss_object = tf.keras.losses.MeanSquaredError()
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	test_loss = tf.keras.metrics.Mean(name='test_loss')
	m_tr = tf.keras.metrics.Mean()
	m_te = tf.keras.metrics.Mean()

	@tf.function
	def train_step(decoder, optimizer_c, train_loss, m_tr, pos, tar, context_p = 50):
		'''
		A typical train step function for TF2. Elements which we wish to track their gradient
		has to be inside the GradientTape() clause. see (1) https://www.tensorflow.org/guide/migrate 
		(2) https://www.tensorflow.org/tutorials/quickstart/advanced
		------------------
		Parameters:
		pos (np array): array of positions (x values) - the 1st/2nd output from data_generator_for_gp_mimick_gpt
		tar (np array): array of targets. Notice that if dealing with sequnces, we typically want to have the targets go from 0 to n-1. The 3rd/4th output from data_generator_for_gp_mimick_gpt  
		pos_mask (np array): see description in position_mask function
		------------------    
		'''
		tar_inp = tar[:, :-1]
		tar_real = tar[:, 1:]
	    
		combined_mask_pos = masks.create_masks(pos)
		# combined_mask_tar = masks.create_masks(tar_inp)

		# print('combined_mask_pos: ', combined_mask_pos)
		# print('combined_mask_tar: ', combined_mask_tar)


		with tf.GradientTape(persistent=True) as tape:
			pred = decoder(pos, tar_inp, True, combined_mask_pos[:, 1:, :-1])
			loss, mse, mask = losses.loss_function(tar_real[:, context_p:], pred = pred[:, context_p:, 0], pred_log_sig = pred[:, context_p:, 1])

		gradients = tape.gradient(loss, decoder.trainable_variables)
		optimizer_c.apply_gradients(zip(gradients, decoder.trainable_variables))
		train_loss(loss); m_tr.update_state(mse, mask)
		names = [v.name for v in decoder.trainable_variables]
		shapes = [v.shape for v in decoder.trainable_variables]

		return pred[:, :, 0], pred[:, :, 1], decoder.trainable_variables, names, shapes

	@tf.function
	def test_step(decoder, test_loss, m_te, pos_te, tar_te, context_p = 50):
		'''
		---------------
		Parameters:
		pos (np array): array of positions (x values) - the 1st/2nd output from data_generator_for_gp_mimick_gpt
		tar (np array): array of targets. Notice that if dealing with sequnces, we typically want to have the targets go from 0 to n-1. The 3rd/4th output from data_generator_for_gp_mimick_gpt  
		pos_mask_te (np array): see description in position_mask function
		---------------
		'''
		tar_inp_te = tar_te[:, :-1]
		tar_real_te = tar_te[:, 1:]
		combined_mask_pos_te = masks.create_masks(pos_te)
		# combined_mask_tar_te = masks.create_masks(tar_inp_te)    
		# training=False is only needed if there are layers with different
		# behavior during training versus inference (e.g. Dropout).

		pred_te = decoder(pos_te, tar_inp_te, False, combined_mask_pos_te[:, 1:, :-1])
		t_loss, t_mse, t_mask = losses.loss_function(tar_real_te[:, context_p:], pred = pred_te[:, context_p:, 0], pred_log_sig = pred_te[:, context_p:, 1])

		test_loss(t_loss); m_te.update_state(t_mse, t_mask)
		return pred_te[:, :, 0], pred_te[:, :, 1]

	tf.keras.backend.set_floatx('float64')
	return train_step, test_step, loss_object, train_loss, test_loss, m_tr, m_te
