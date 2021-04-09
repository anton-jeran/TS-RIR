from __future__ import print_function

try:
  import cPickle as pickle
except:
  import pickle
from functools import reduce
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import tensorflow as tf
from six.moves import xrange
import RT60
import loader
from TSRIRgan import TSRIRGANGenerator_synthetic, TSRIRGANGenerator_real, TSRIRGANDiscriminator_synthetic,TSRIRGANDiscriminator_real


"""
  Trains a TSRIRGAN
"""
# loss_obj = tf.keras.losses.binary_crossentropy(from_logits=True)

def mae_criterion(pred, target):
    return tf.reduce_mean((pred - target) ** 2)

LAMBDA = 10
def discriminator_loss(real, generated):
  # real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real), real)#,from_logits=True)
  real_loss = mae_criterion(tf.ones_like(real), real)
  # generated_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(generated), generated)#,from_logits=True)
  generated_loss = mae_criterion(tf.zeros_like(generated), generated)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5*LAMBDA 

def generator_loss(generated):
  # return tf.keras.losses.binary_crossentropy(tf.ones_like(generated), generated)#,from_logits=True)
  return mae_criterion(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

def RT_60_loss(real,generated,sess):
  # sess = tf.Session()
  # sess.run(tf.global_variables_initializer())
  _real = real.eval(session = sess)
  _generated = generated.eval(session = sess)
  # _generated  = generated 
  no_samples = len(_real)
  sampling_rate = 16000
  t60_loss_list =list()

  for i in range (no_samples):
    real_wav = _real[i]
    generated_wav = _generated[i]
    real_t60_val = RT60.t60_impulse(real_wav,sampling_rate)
    generated_t60_val = RT60.t60_impulse(generated_wav,sampling_rate)
    # print("real t60 ", real_t60_val)
    # print("generated t60 ", generated_t60_val)
    t60_loss = abs(real_t60_val-generated_t60_val)
    t60_loss_list.append(t60_loss)

  mean_t60_loss = sum(t60_loss_list)/len(t60_loss_list)

  return mean_t60_loss
 
 

def train(fps1,fps2, args):
  with tf.name_scope('loader'):
    x_real = loader.decode_extract_and_batch(
        fps1,
        batch_size=args.train_batch_size,
        slice_len=args.data1_slice_len,
        decode_fs=args.data1_sample_rate,
        decode_num_channels=args.data1_num_channels,
        decode_fast_wav=args.data1_fast_wav,
        decode_parallel_calls=4,
        slice_randomize_offset=False if args.data1_first_slice else True,
        slice_first_only=args.data1_first_slice,
        slice_overlap_ratio=0. if args.data1_first_slice else args.data1_overlap_ratio,
        slice_pad_end=True if args.data1_first_slice else args.data1_pad_end,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=4096,
        prefetch_size=args.train_batch_size * 4,
        prefetch_gpu_num=args.data1_prefetch_gpu_num)[:, :, 0]

    x_synthetic = loader.decode_extract_and_batch(
        fps2,
        batch_size=args.train_batch_size,
        slice_len=args.data2_slice_len,
        decode_fs=args.data2_sample_rate,
        decode_num_channels=args.data2_num_channels,
        decode_fast_wav=args.data2_fast_wav,
        decode_parallel_calls=4,
        slice_randomize_offset=False if args.data2_first_slice else True,
        slice_first_only=args.data2_first_slice,
        slice_overlap_ratio=0. if args.data2_first_slice else args.data2_overlap_ratio,
        slice_pad_end=True if args.data2_first_slice else args.data2_pad_end,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=4096,
        prefetch_size=args.train_batch_size * 4,
        prefetch_gpu_num=args.data1_prefetch_gpu_num)[:, :, 0]
  
  # print('length check', len(x_real))
  # Make z vector
  # z = tf.random_uniform([args.train_batch_size, args.TSRIRgan_latent_dim], -1., 1., dtype=tf.float32)

  # Make generator_synthetic
  with tf.variable_scope('G_synthetic'):
    G_synthetic = TSRIRGANGenerator_synthetic(x_real, train=True, **args.TSRIRgan_g_kwargs)
    if args.TSRIRgan_genr_pp:
      with tf.variable_scope('s_pp_filt'):
        G_synthetic = tf.layers.conv1d(G_synthetic, 1, args.TSRIRgan_genr_pp_len, use_bias=False, padding='same')
  G_synthetic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_synthetic')

  # Print G_synthetic summary
  print('-' * 80)
  print('Generator_synthetic vars')
  nparams = 0
  for v in G_synthetic_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

  # Summarize
  tf.summary.audio('x_real', x_real, args.data1_sample_rate)
  tf.summary.audio('G_synthetic', G_synthetic, args.data1_sample_rate)
  G_synthetic_rms = tf.sqrt(tf.reduce_mean(tf.square(G_synthetic[:, :, 0]), axis=1))
  x_real_rms = tf.sqrt(tf.reduce_mean(tf.square(x_real[:, :, 0]), axis=1))
  tf.summary.histogram('x_real_rms_batch', x_real_rms)
  tf.summary.histogram('G_synthetic_rms_batch', G_synthetic_rms)
  tf.summary.scalar('x_real_rms', tf.reduce_mean(x_real_rms))
  tf.summary.scalar('G_synthetic_rms', tf.reduce_mean(G_synthetic_rms))

   # Make generator_real
  with tf.variable_scope('G_real'):
    G_real = TSRIRGANGenerator_real(x_synthetic, train=True, **args.TSRIRgan_g_kwargs)
    if args.TSRIRgan_genr_pp:
      with tf.variable_scope('r_pp_filt'):
        G_real = tf.layers.conv1d(G_real, 1, args.TSRIRgan_genr_pp_len, use_bias=False, padding='same')
  G_real_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_real')

  # Print G_real summary
  print('-' * 80)
  print('Generator_real vars')
  nparams = 0
  for v in G_real_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

  # Summarize
  tf.summary.audio('x_synthetic', x_synthetic, args.data1_sample_rate)
  tf.summary.audio('G_real', G_real, args.data1_sample_rate)
  G_real_rms = tf.sqrt(tf.reduce_mean(tf.square(G_real[:, :, 0]), axis=1))
  x_synthetic_rms = tf.sqrt(tf.reduce_mean(tf.square(x_synthetic[:, :, 0]), axis=1))
  tf.summary.histogram('x_synthetic_rms_batch', x_synthetic_rms)
  tf.summary.histogram('G_real_rms_batch', G_real_rms)
  tf.summary.scalar('x_synthetic_rms', tf.reduce_mean(x_synthetic_rms))
  tf.summary.scalar('G_real_rms', tf.reduce_mean(G_real_rms))


  #Generating Cycled Image
  with tf.variable_scope('G_synthetic',reuse=True):
    cycle_synthetic = TSRIRGANGenerator_synthetic(G_real, train=True, **args.TSRIRgan_g_kwargs)
    if args.TSRIRgan_genr_pp:
      with tf.variable_scope('s_pp_filt'):
        cycle_synthetic = tf.layers.conv1d(cycle_synthetic, 1, args.TSRIRgan_genr_pp_len, use_bias=False, padding='same')
  G_synthetic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_synthetic')

  with tf.variable_scope('G_real', reuse=True):
    cycle_real = TSRIRGANGenerator_real(G_synthetic, train=True, **args.TSRIRgan_g_kwargs)
    if args.TSRIRgan_genr_pp:
      with tf.variable_scope('r_pp_filt'):
        cycle_real = tf.layers.conv1d(cycle_real, 1, args.TSRIRgan_genr_pp_len, use_bias=False, padding='same')
  G_real_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_real')

  #Generating Same Image
  with tf.variable_scope('G_synthetic', reuse=True):
    same_synthetic = TSRIRGANGenerator_synthetic(x_synthetic, train=True, **args.TSRIRgan_g_kwargs)
    if args.TSRIRgan_genr_pp:
      with tf.variable_scope('s_pp_filt'):
        same_synthetic = tf.layers.conv1d(same_synthetic, 1, args.TSRIRgan_genr_pp_len, use_bias=False, padding='same')
  G_synthetic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_synthetic')

  with tf.variable_scope('G_real', reuse=True):
    same_real = TSRIRGANGenerator_real(x_real, train=True, **args.TSRIRgan_g_kwargs)
    if args.TSRIRgan_genr_pp:
      with tf.variable_scope('r_pp_filt'):
        same_real = tf.layers.conv1d(same_real, 1, args.TSRIRgan_genr_pp_len, use_bias=False, padding='same')
  G_real_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_real') 

  #Synthetic
  # Make real discriminator
  with tf.name_scope('D_synthetic_x'), tf.variable_scope('D_synthetic'):
    D_synthetic_x = TSRIRGANDiscriminator_synthetic(x_synthetic, **args.TSRIRgan_d_kwargs)
  D_synthetic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_synthetic')

  # Print D summary
  print('-' * 80)
  print('Discriminator_synthetic vars')
  nparams = 0
  for v in D_synthetic_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
  print('-' * 80)

  # Make fake discriminator
  with tf.name_scope('D_G_synthetic'), tf.variable_scope('D_synthetic', reuse=True):
    D_G_synthetic = TSRIRGANDiscriminator_synthetic(G_synthetic, **args.TSRIRgan_d_kwargs)

  
  #Real
  # Make real discriminator
  with tf.name_scope('D_real_x'), tf.variable_scope('D_real'):
    D_real_x = TSRIRGANDiscriminator_real(x_real, **args.TSRIRgan_d_kwargs)
  D_real_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_real')

  # Print D summary
  print('-' * 80)
  print('Discriminator_real vars')
  nparams = 0
  for v in D_real_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
  print('-' * 80)
  
  # Make fake discriminator
  with tf.name_scope('D_G_real'), tf.variable_scope('D_real', reuse=True):
    D_G_real = TSRIRGANDiscriminator_real(G_real, **args.TSRIRgan_d_kwargs)
############stop here###########
  # Create loss
  D_clip_weights = None
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  if args.TSRIRgan_loss == 'cycle-gan':
    #Real IR
    gen_real_loss = generator_loss(D_G_real)
    gen_synthetic_loss = generator_loss(D_G_synthetic)

    cycle_loss_real = calc_cycle_loss(x_real,cycle_real)
    cycle_loss_synthetic = calc_cycle_loss(x_synthetic,cycle_synthetic)
    
    total_cycle_loss =  cycle_loss_real +  cycle_loss_synthetic
    
    same_real_loss = identity_loss(x_real,same_real)
    same_synthetic_loss = identity_loss(x_synthetic,same_synthetic)
    
    # RT60_loss_real = RT_60_loss(x_real,G_real,sess)
    # RT60_loss_synthetic = RT_60_loss(x_synthetic,G_synthetic,sess)

    total_gen_real_loss = gen_real_loss + 25*total_cycle_loss + 35*same_real_loss #+RT60_loss_real
    total_gen_synthetic_loss = gen_synthetic_loss + 25*total_cycle_loss + 35*same_synthetic_loss # +RT60_loss_synthetic

    disc_synthetic_loss = discriminator_loss(D_synthetic_x,D_G_synthetic)
    disc_real_loss = discriminator_loss(D_real_x,D_G_real)

  else:
    raise NotImplementedError()

  # tf.summary.scalar('RT60_loss_real', RT60_loss_real)
  # tf.summary.scalar('RT60_loss_synthetic',RT60_loss_synthetic)
  tf.summary.scalar('G_real_loss', total_gen_real_loss)
  tf.summary.scalar('G_synthetic_loss', total_gen_synthetic_loss)
  tf.summary.scalar('D_real_loss', disc_real_loss)
  tf.summary.scalar('D_synthetic_loss', disc_synthetic_loss)

  tf.summary.scalar('Generator_real_loss', gen_real_loss)
  tf.summary.scalar('Generator_synthetic_loss', gen_synthetic_loss)
  tf.summary.scalar('Cycle_loss_real',15*cycle_loss_real)
  tf.summary.scalar('Cycle_loss_synthetic', 15*cycle_loss_synthetic)
  tf.summary.scalar('Same_loss_real',20*same_real_loss)
  tf.summary.scalar('Same_loss_synthetic', 20*same_synthetic_loss)

  # Create (recommended) optimizer
  if args.TSRIRgan_loss == 'cycle-gan':
    # G_real_opt = tf.train.AdamOptimizer(
    #     learning_rate=2e-4,
    #     beta1=0.5)
    # G_synthetic_opt = tf.train.AdamOptimizer(
    #     learning_rate=2e-4,
    #     beta1=0.5)
    # D_real_opt = tf.train.AdamOptimizer(
    #     learning_rate=2e-4,
    #     beta1=0.5)
    # D_synthetic_opt = tf.train.AdamOptimizer(
    #     learning_rate=2e-4,
    #     beta1=0.5)
    G_real_opt = tf.train.RMSPropOptimizer(
        learning_rate=3e-5)
    G_synthetic_opt = tf.train.RMSPropOptimizer(
        learning_rate=3e-5)
    D_real_opt = tf.train.RMSPropOptimizer(
        learning_rate=3e-5)
    D_synthetic_opt = tf.train.RMSPropOptimizer(
        learning_rate=3e-5)
  else:
    raise NotImplementedError()
  
  # Create training ops
  G_real_train_op = G_real_opt.minimize(total_gen_real_loss, var_list=G_real_vars,
      global_step=tf.train.get_or_create_global_step())
  G_synthetic_train_op = G_synthetic_opt.minimize(total_gen_synthetic_loss, var_list=G_synthetic_vars,
      global_step=tf.train.get_or_create_global_step())
  D_real_train_op = D_real_opt.minimize(disc_real_loss, var_list=D_real_vars)
  D_synthetic_train_op = D_synthetic_opt.minimize(disc_synthetic_loss, var_list=D_synthetic_vars)

  # Run training
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_save_secs,
      save_summaries_secs=args.train_summary_secs) as sess:
    print('-' * 80)
    print('Training has started. Please use \'tensorboard --logdir={}\' to monitor.'.format(args.train_dir))
    # RT60_loss_real = RT_60_loss(x_real,G_real,sess)
    # RT60_loss_synthetic = RT_60_loss(x_synthetic,G_synthetic,sess)
    while True:
      # Train discriminator
      for i in xrange(args.TSRIRgan_disc_nupdates):
        sess.run(D_real_train_op)
        sess.run(D_synthetic_train_op)

        # Enforce Lipschitz constraint for WGAN
        # if D_clip_weights is not None:
        #   sess.run(D_clip_weights)

      # Train generator
      sess.run(G_real_train_op)
      sess.run(G_synthetic_train_op)
      # RT60_loss_real = RT_60_loss(x_real,G_real,sess)
      # RT60_loss_synthetic = RT_60_loss(x_synthetic,G_synthetic,sess)

def infer(args):
  infer_dir = os.path.join(args.train_dir, 'infer')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  samp_x_synthetic_n = tf.placeholder(tf.int32, [], name='samp_x_synthetic_n')
  samp_x_real_n = tf.placeholder(tf.int32, [], name='samp_x_real_n')

  # samp_z = tf.random_uniform([samp_z_n, args.TSRIRgan_latent_dim], -1.0, 1.0, dtype=tf.float32, name='samp_z')

  # Input zo
  x_real = tf.placeholder(tf.float32, [64, 16384, 1], name='x_real')
  x_synthetic = tf.placeholder(tf.float32, [64, 16384, 1], name='x_synthetic')


  synthetic_flat_pad = tf.placeholder(tf.int32, [], name='synthetic_flat_pad')
  x_synthetic_flat_pad = tf.placeholder(tf.int32, [], name='x_synthetic_flat_pad')
  real_flat_pad = tf.placeholder(tf.int32, [], name='real_flat_pad')
  x_real_flat_pad = tf.placeholder(tf.int32, [], name='x_real_flat_pad')
  print("shape  ", x_real.shape)
  # Execute generator
  with tf.variable_scope('G_synthetic'):
    G_synthetic_x = TSRIRGANGenerator_synthetic(x_real, train=False, **args.TSRIRgan_g_kwargs)
    if args.TSRIRgan_genr_pp:
      with tf.variable_scope('s_pp_filt'):
        G_synthetic_x = tf.layers.conv1d(G_synthetic_x, 1, args.TSRIRgan_genr_pp_len, use_bias=False, padding='same')
  G_synthetic_x = tf.identity(G_synthetic_x, name='G_synthetic_x')

  with tf.variable_scope('G_real'):
    G_real_x = TSRIRGANGenerator_real(x_synthetic, train=False, **args.TSRIRgan_g_kwargs)
    if args.TSRIRgan_genr_pp:
      with tf.variable_scope('r_pp_filt'):
        G_real_x = tf.layers.conv1d(G_real_x, 1, args.TSRIRgan_genr_pp_len, use_bias=False, padding='same')
  G_real_x = tf.identity(G_real_x, name='G_real_x')

  # Flatten batch
  synthetic_nch = int(G_synthetic_x.get_shape()[-1])
  G_synthetic_x_padded = tf.pad(G_synthetic_x, [[0, 0], [0, synthetic_flat_pad], [0, 0]])
  G_synthetic_x_flat = tf.reshape(G_synthetic_x_padded, [-1, synthetic_nch], name='G_synthetic_x_flat')

  xs_nch = int(x_synthetic.get_shape()[-1])
  x_synthetic_padded = tf.pad(x_synthetic, [[0, 0], [0, x_synthetic_flat_pad], [0, 0]])
  x_synthetic_flat = tf.reshape(x_synthetic_padded, [-1, xs_nch], name='x_synthetic_flat')


  real_nch = int(G_real_x.get_shape()[-1])
  G_real_x_padded = tf.pad(G_real_x, [[0, 0], [0, real_flat_pad], [0, 0]])
  G_real_x_flat = tf.reshape(G_real_x_padded, [-1, real_nch], name='G_real_x_flat')

  xr_nch = int(x_real.get_shape()[-1])
  x_real_padded = tf.pad(x_real, [[0, 0], [0, x_real_flat_pad], [0, 0]])
  x_real_flat = tf.reshape(x_real_padded, [-1, xr_nch], name='x_real_flat')

  # Encode to int16
  def float_to_int16(x, name=None):
    x_int16 = x * 32767.
    x_int16 = tf.clip_by_value(x_int16, -32767., 32767.)
    x_int16 = tf.cast(x_int16, tf.int16, name=name)
    return x_int16
  G_synthetic_x_int16 = float_to_int16(G_synthetic_x, name='G_synthetic_x_int16')
  G_synthetic_x_flat_int16 = float_to_int16(G_synthetic_x_flat, name='G_synthetic_x_flat_int16')
  G_real_x_int16 = float_to_int16(G_real_x, name='G_real_x_int16')
  G_real_x_flat_int16 = float_to_int16(G_real_x_flat, name='G_real_x_flat_int16')

  x_synthetic_int16 = float_to_int16(x_synthetic, name='x_synthetic_int16')
  x_synthetic_flat_int16 = float_to_int16(x_synthetic_flat, name='x_synthetic_flat_int16')
  x_real_int16 = float_to_int16(x_real, name='x_real_int16')
  x_real_flat_int16 = float_to_int16(x_real_flat, name='x_real_flat_int16')

  # Create saver
  G_synthetic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G_synthetic')
  global_step = tf.train.get_or_create_global_step()

  G_real_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G_real')###

  saver = tf.train.Saver(G_synthetic_vars + G_real_vars + [global_step])

  # Export graph
  tf.train.write_graph(tf.get_default_graph(), infer_dir, 'infer.pbtxt')

  # Export MetaGraph
  infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
  tf.train.export_meta_graph(
      filename=infer_metagraph_fp,
      clear_devices=True,
      saver_def=saver.as_saver_def())

  # # Create saver
  # G_real_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G_real')
  # global_step = tf.train.get_or_create_global_step()
  # saver = tf.train.Saver(G_real_vars + [global_step])

  # # Export graph
  # tf.train.write_graph(tf.get_default_graph(), infer_dir, 'infer.pbtxt')

  # # Export MetaGraph
  # infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
  # tf.train.export_meta_graph(
  #     filename=infer_metagraph_fp,
  #     clear_devices=True,
  #     saver_def=saver.as_saver_def())

  # Reset graph (in case training afterwards)
  tf.reset_default_graph()


"""
  Generates a preview audio file every time a checkpoint is saved
"""
def preview(fps1,fps2,args):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from scipy.io.wavfile import write as wavwrite
  from scipy.signal import freqz

  preview_dir = os.path.join(args.train_dir, 'preview')
  if not os.path.isdir(preview_dir):
    os.makedirs(preview_dir)

  ####################################################
  s_fps1 = fps1[0:args.preview_n]
  s_fps2 = fps2[0:args.preview_n]
  with tf.name_scope('samp_x_real'):
    x_real = loader.decode_extract_and_batch(
        s_fps1,
        batch_size=args.train_batch_size,
        slice_len=args.data1_slice_len,
        decode_fs=args.data1_sample_rate,
        decode_num_channels=args.data1_num_channels,
        decode_fast_wav=args.data1_fast_wav,
        decode_parallel_calls=4,
        slice_randomize_offset=False if args.data1_first_slice else True,
        slice_first_only=args.data1_first_slice,
        slice_overlap_ratio=0. if args.data1_first_slice else args.data1_overlap_ratio,
        slice_pad_end=True if args.data1_first_slice else args.data1_pad_end,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=4096,
        prefetch_size=args.train_batch_size * 4,
        prefetch_gpu_num=args.data1_prefetch_gpu_num)[:, :, 0]
 

  with tf.name_scope('samp_x_synthetic'):
    x_synthetic = loader.decode_extract_and_batch(
        s_fps2,
        batch_size=args.train_batch_size,
        slice_len=args.data2_slice_len,
        decode_fs=args.data2_sample_rate,
        decode_num_channels=args.data2_num_channels,
        decode_fast_wav=args.data2_fast_wav,
        decode_parallel_calls=4,
        slice_randomize_offset=False if args.data2_first_slice else True,
        slice_first_only=args.data2_first_slice,
        slice_overlap_ratio=0. if args.data2_first_slice else args.data2_overlap_ratio,
        slice_pad_end=True if args.data2_first_slice else args.data2_pad_end,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=4096,
        prefetch_size=args.train_batch_size * 4,
        prefetch_gpu_num=args.data1_prefetch_gpu_num)[:, :, 0]

  ####################################################
  x_synthetic = x_synthetic.eval(session=tf.Session()) 
  x_real = x_real.eval(session=tf.Session()) 
  
  # Load graph
  infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
  graph = tf.get_default_graph()
  saver = tf.train.import_meta_graph(infer_metagraph_fp)


  # Set up graph for generating preview images
  feeds = {}
  feeds[graph.get_tensor_by_name('x_synthetic:0')] = x_synthetic
  feeds[graph.get_tensor_by_name('synthetic_flat_pad:0')] = int(args.data1_sample_rate / 2)
  feeds[graph.get_tensor_by_name('x_synthetic_flat_pad:0')] = int(args.data1_sample_rate / 2)
  feeds[graph.get_tensor_by_name('x_real:0')] = x_real
  feeds[graph.get_tensor_by_name('real_flat_pad:0')] = int(args.data1_sample_rate / 2)
  feeds[graph.get_tensor_by_name('x_real_flat_pad:0')] = int(args.data1_sample_rate / 2)
  fetches = {}
  fetches['step'] = tf.train.get_or_create_global_step()
  fetches['G_synthetic_x'] = graph.get_tensor_by_name('G_synthetic_x:0')
  fetches['G_synthetic_x_flat_int16'] = graph.get_tensor_by_name('G_synthetic_x_flat_int16:0')
  fetches['x_synthetic_flat_int16'] = graph.get_tensor_by_name('x_synthetic_flat_int16:0')
  fetches['G_real_x'] = graph.get_tensor_by_name('G_real_x:0')
  fetches['G_real_x_flat_int16'] = graph.get_tensor_by_name('G_real_x_flat_int16:0')
  fetches['x_real_flat_int16'] = graph.get_tensor_by_name('x_real_flat_int16:0')
  if args.TSRIRgan_genr_pp:
    s_fetches['s_pp_filter'] = graph.get_tensor_by_name('G_synthetic_x/s_pp_filt/conv1d/kernel:0')[:, 0, 0]
    s_fetches['r_pp_filter'] = graph.get_tensor_by_name('G_real_x/r_pp_filt/conv1d/kernel:0')[:, 0, 0]
  
  # Summarize
  G_synthetic_x = graph.get_tensor_by_name('G_synthetic_x_flat:0')
  s_summaries = [
      tf.summary.audio('preview', tf.expand_dims(G_synthetic_x, axis=0), args.data1_sample_rate, max_outputs=1)
  ]
  fetches['s_summaries'] = tf.summary.merge(s_summaries)
  s_summary_writer = tf.summary.FileWriter(preview_dir)

  G_real_x = graph.get_tensor_by_name('G_real_x_flat:0')
  r_summaries = [
      tf.summary.audio('preview', tf.expand_dims(G_real_x, axis=0), args.data1_sample_rate, max_outputs=1)
  ]
  fetches['r_summaries'] = tf.summary.merge(r_summaries)
  r_summary_writer = tf.summary.FileWriter(preview_dir)



  # PP Summarize
  if args.TSRIRgan_genr_pp:
    s_pp_fp = tf.placeholder(tf.string, [])
    s_pp_bin = tf.read_file(s_pp_fp)
    s_pp_png = tf.image.decode_png(s_pp_bin)
    s_pp_summary = tf.summary.image('s_pp_filt', tf.expand_dims(s_pp_png, axis=0))

  if args.TSRIRgan_genr_pp:
    r_pp_fp = tf.placeholder(tf.string, [])
    r_pp_bin = tf.read_file(r_pp_fp)
    r_pp_png = tf.image.decode_png(r_pp_bin)
    r_pp_summary = tf.summary.image('r_pp_filt', tf.expand_dims(r_pp_png, axis=0))

  # Loop, waiting for checkpoints
  ckpt_fp = None
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      print('Preview: {}'.format(latest_ckpt_fp))

      with tf.Session() as sess:
        saver.restore(sess, latest_ckpt_fp)

        _fetches = sess.run(fetches, feeds)

        _step = _fetches['step']

      # with tf.Session() as sess:
      #   saver.restore(sess, latest_ckpt_fp)

      #   _r_fetches = sess.run(r_fetches, r_feeds)

      #   _r_step = _r_fetches['step']

      s_preview_fp = os.path.join(preview_dir, '{}.wav'.format(str(_step).zfill(8)+'synthetic'))
      wavwrite(s_preview_fp, args.data1_sample_rate, _fetches['G_synthetic_x_flat_int16'])
      s_original_fp = os.path.join(preview_dir, '{}.wav'.format('synthetic_original'))
      wavwrite(s_original_fp, args.data1_sample_rate, _fetches['x_synthetic_flat_int16'])
      
      s_summary_writer.add_summary(_fetches['s_summaries'], _step)

      r_preview_fp = os.path.join(preview_dir, '{}.wav'.format(str(_step).zfill(8)+'real'))
      wavwrite(r_preview_fp, args.data1_sample_rate, _fetches['G_real_x_flat_int16'])
      r_original_fp = os.path.join(preview_dir, '{}.wav'.format('real_original'))
      wavwrite(r_original_fp, args.data1_sample_rate, _fetches['x_real_flat_int16'])

      r_summary_writer.add_summary(_fetches['r_summaries'], _step)

      #I have to edit this
      # if args.TSRIRgan_genr_pp:
      #   s_w, s_h = freqz(_s_fetches['s_pp_filter'])

      #   fig = plt.figure()
      #   plt.title('Digital filter frequncy response')
      #   ax1 = fig.add_subplot(111)

      #   plt.plot(w, 20 * np.log10(abs(h)), 'b')
      #   plt.ylabel('Amplitude [dB]', color='b')
      #   plt.xlabel('Frequency [rad/sample]')

      #   ax2 = ax1.twinx()
      #   angles = np.unwrap(np.angle(h))
      #   plt.plot(w, angles, 'g')
      #   plt.ylabel('Angle (radians)', color='g')
      #   plt.grid()
      #   plt.axis('tight')

      #   _pp_fp = os.path.join(preview_dir, '{}_ppfilt.png'.format(str(_step).zfill(8)))
      #   plt.savefig(_pp_fp)

      #   with tf.Session() as sess:
      #     _summary = sess.run(pp_summary, {pp_fp: _pp_fp})
      #     summary_writer.add_summary(_summary, _step)

      print('Done')

      ckpt_fp = latest_ckpt_fp

    time.sleep(1)



if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'preview', 'infer'])
  parser.add_argument('train_dir', type=str,
      help='Training directory')

  data1_args = parser.add_argument_group('Data1')
  data1_args.add_argument('--data1_dir', type=str,
      help='Data directory containing *only* audio files to load')
  data1_args.add_argument('--data1_sample_rate', type=int,
      help='Number of audio samples per second')
  data1_args.add_argument('--data1_slice_len', type=int, choices=[16384, 32768, 65536],
      help='Number of audio samples per slice (maximum generation length)')
  data1_args.add_argument('--data1_num_channels', type=int,
      help='Number of audio channels to generate (for >2, must match that of data)')
  data1_args.add_argument('--data1_overlap_ratio', type=float,
      help='Overlap ratio [0, 1) between slices')
  data1_args.add_argument('--data1_first_slice', action='store_true', dest='data1_first_slice',
      help='If set, only use the first slice each audio example')
  data1_args.add_argument('--data1_pad_end', action='store_true', dest='data1_pad_end',
      help='If set, use zero-padded partial slices from the end of each audio file')
  data1_args.add_argument('--data1_normalize', action='store_true', dest='data1_normalize',
      help='If set, normalize the training examples')
  data1_args.add_argument('--data1_fast_wav', action='store_true', dest='data1_fast_wav',
      help='If your data is comprised of standard WAV files (16-bit signed PCM or 32-bit float), use this flag to decode audio using scipy (faster) instead of librosa')
  data1_args.add_argument('--data1_prefetch_gpu_num', type=int,
      help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')

  data2_args = parser.add_argument_group('Data2')
  data2_args.add_argument('--data2_dir', type=str,
      help='Data directory containing *only* audio files to load')
  data2_args.add_argument('--data2_sample_rate', type=int,
      help='Number of audio samples per second')
  data2_args.add_argument('--data2_slice_len', type=int, choices=[16384, 32768, 65536],
      help='Number of audio samples per slice (maximum generation length)')
  data2_args.add_argument('--data2_num_channels', type=int,
      help='Number of audio channels to generate (for >2, must match that of data)')
  data2_args.add_argument('--data2_overlap_ratio', type=float,
      help='Overlap ratio [0, 1) between slices')
  data2_args.add_argument('--data2_first_slice', action='store_true', dest='data2_first_slice',
      help='If set, only use the first slice each audio example')
  data2_args.add_argument('--data2_pad_end', action='store_true', dest='data2_pad_end',
      help='If set, use zero-padded partial slices from the end of each audio file')
  data2_args.add_argument('--data2_normalize', action='store_true', dest='data2_normalize',
      help='If set, normalize the training examples')
  data2_args.add_argument('--data2_fast_wav', action='store_true', dest='data2_fast_wav',
      help='If your data is comprised of standard WAV files (16-bit signed PCM or 32-bit float), use this flag to decode audio using scipy (faster) instead of librosa')
  data2_args.add_argument('--data2_prefetch_gpu_num', type=int,
      help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')


  TSRIRgan_args = parser.add_argument_group('TSRIRGAN')
  TSRIRgan_args.add_argument('--TSRIRgan_latent_dim', type=int,
      help='Number of dimensions of the latent space')
  TSRIRgan_args.add_argument('--TSRIRgan_kernel_len', type=int,
      help='Length of 1D filter kernels')
  TSRIRgan_args.add_argument('--TSRIRgan_dim', type=int,
      help='Dimensionality multiplier for model of G and D')
  TSRIRgan_args.add_argument('--TSRIRgan_batchnorm', action='store_true', dest='TSRIRgan_batchnorm',
      help='Enable batchnorm')
  TSRIRgan_args.add_argument('--TSRIRgan_disc_nupdates', type=int,
      help='Number of discriminator updates per generator update')
  TSRIRgan_args.add_argument('--TSRIRgan_loss', type=str, choices=['cycle-gan'],
      help='Which GAN loss to use')
  TSRIRgan_args.add_argument('--TSRIRgan_genr_upsample', type=str, choices=['zeros', 'nn'],
      help='Generator upsample strategy')
  TSRIRgan_args.add_argument('--TSRIRgan_genr_pp', action='store_true', dest='TSRIRgan_genr_pp',
      help='If set, use post-processing filter')
  TSRIRgan_args.add_argument('--TSRIRgan_genr_pp_len', type=int,
      help='Length of post-processing filter for DCGAN')
  TSRIRgan_args.add_argument('--TSRIRgan_disc_phaseshuffle', type=int,
      help='Radius of phase shuffle operation')

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_batch_size', type=int,
      help='Batch size')
  train_args.add_argument('--train_save_secs', type=int,
      help='How often to save model')
  train_args.add_argument('--train_summary_secs', type=int,
      help='How often to report summaries')

  preview_args = parser.add_argument_group('Preview')
  preview_args.add_argument('--preview_n', type=int,
      help='Number of samples to preview')

  # incept_args = parser.add_argument_group('Incept')
  # incept_args.add_argument('--incept_metagraph_fp', type=str,
  #     help='Inference model for inception score')
  # incept_args.add_argument('--incept_ckpt_fp', type=str,
  #     help='Checkpoint for inference model')
  # incept_args.add_argument('--incept_n', type=int,
  #     help='Number of generated examples to test')
  # incept_args.add_argument('--incept_k', type=int,
  #     help='Number of groups to test')

  parser.set_defaults(
    data1_dir=None,
    data1_sample_rate=16000,
    data1_slice_len=16384,
    data1_num_channels=1,
    data1_overlap_ratio=0.,
    data1_first_slice=False,
    data1_pad_end=False,
    data1_normalize=False,
    data1_fast_wav=False,
    data1_prefetch_gpu_num=0,
    data2_dir=None,
    data2_sample_rate=16000,
    data2_slice_len=16384,
    data2_num_channels=1,
    data2_overlap_ratio=0.,
    data2_first_slice=False,
    data2_pad_end=False,
    data2_normalize=False,
    data2_fast_wav=False,
    data2_prefetch_gpu_num=0,
    TSRIRgan_latent_dim=100,
    TSRIRgan_kernel_len=25,
    TSRIRgan_dim=64,
    TSRIRgan_batchnorm=False,
    TSRIRgan_disc_nupdates=2,
    TSRIRgan_loss='cycle-gan',
    TSRIRgan_genr_upsample='zeros',
    TSRIRgan_genr_pp=False,
    TSRIRgan_genr_pp_len=512,
    TSRIRgan_disc_phaseshuffle=2,
    train_batch_size=64,
    train_save_secs=300,
    train_summary_secs=120,
    preview_n=32)#,
    # incept_metagraph_fp='./eval/inception/infer.meta',
    # incept_ckpt_fp='./eval/inception/best_acc-103005',
    # incept_n=5000,
    # incept_k=10)

  args = parser.parse_args()

  # Make train dir
  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  # Save args
  with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

  # Make model kwarg dicts
  setattr(args, 'TSRIRgan_g_kwargs', {
    'slice_len': args.data1_slice_len,
    'nch': args.data1_num_channels,
    'kernel_len': args.TSRIRgan_kernel_len,
    'dim': args.TSRIRgan_dim,
    'use_batchnorm': args.TSRIRgan_batchnorm,
    'upsample': args.TSRIRgan_genr_upsample
  })
  setattr(args, 'TSRIRgan_d_kwargs', {
    'kernel_len': args.TSRIRgan_kernel_len,
    'dim': args.TSRIRgan_dim,
    'use_batchnorm': args.TSRIRgan_batchnorm,
    'phaseshuffle_rad': args.TSRIRgan_disc_phaseshuffle
  })


  fps1 = glob.glob(os.path.join(args.data1_dir, '*'))
  if len(fps1) == 0:
    raise Exception('Did not find any audio files in specified directory(real_IR)')
  print('Found {} audio files in specified directory'.format(len(fps1)))
  fps2 = glob.glob(os.path.join(args.data2_dir, '*'))
  if len(fps2) == 0:
    raise Exception('Did not find any audio files in specified directory(synthetic_IR)')
  print('Found {} audio files in specified directory'.format(len(fps2)))
  if args.mode == 'train':   
    infer(args)
    train(fps1,fps2, args)
  elif args.mode == 'preview':
    preview(fps1,fps2,args)
  elif args.mode == 'infer':
    infer(args)
  else:
    raise NotImplementedError()
