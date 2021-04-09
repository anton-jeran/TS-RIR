import tensorflow as tf
import loader
from IPython.display import display, Audio
import math
import os
import numpy as np 
import librosa


def generate_real(fps2,args):

  no_samples = len(fps2)
  no_set =int(no_samples/64)
  no_remain = no_samples%64

  for k in range (no_set+1):
    
    if(no_set == k):
      print("k is ",k)
      s_fps2 = fps2[no_samples-64:no_samples]
    else:
      s_fps2 = fps2[(64*k):(64*(k+1))]
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model.ckpt')
  
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
  
  
    _x_synthetic = x_synthetic.eval(session=tf.Session())
    print("input ", len(_x_synthetic), len(_x_synthetic[1]))
      # _z = (np.random.rand(1000, 100) * 2.) - 1
    # Synthesize G(z)
    x_synthetic = graph.get_tensor_by_name('x_synthetic:0')
    x_real = graph.get_tensor_by_name('x_real:0')
    G_real = graph.get_tensor_by_name('G_real_x:0')
    _G_real = sess.run(G_real, {x_synthetic: _x_synthetic,x_real: _x_synthetic})
    print("G_S" , len(_G_real), len(_G_real[1]))
    for i in range (64):
      print("i ",i)
      wav=_G_real[i][0:16000]
      name = 'IRs_for_GAN/' + s_fps2[i]
      print("name ",name)
      librosa.output.write_wav(path=name,y=wav,sr=16000)


if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()


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
    TSRIRgan_disc_nupdates=5,
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

  generate_real(fps2, args)