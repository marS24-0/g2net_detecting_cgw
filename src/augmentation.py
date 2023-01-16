import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.image import dense_image_warp

# Spectograms Augmentation have 3 steps for audio data augmentation.
#     first step is time warping using Tensorflow's image_sparse_warp function.
#     The second step is frequency masking, then last step is time masking.

def sparse_warp(mel_spectrogram, time_warping_para=80):
    """ 
    # Arguments:
        mel_spectrogram(numpy array): spectrograms.
        time_warping_para(float): Augmentation parameter, "time warp parameter W" (default = 80) 
    """

    mel_spectrogram_l1 = mel_spectrogram.copy()
    mel_spectrogram[1,:,:] = mel_spectrogram[0,:,:]
    mel_spectrogram_l1[0,:,:] = mel_spectrogram_l1[1,:,:]

    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]
    
    pt = tf.random.uniform([], time_warping_para, n - time_warping_para, tf.int32)  # radnom point along the time axis
    src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
    src_ctr_pts = tf.cast(src_ctr_pts, tf.float32)

    # Destination
    w = tf.random.uniform([], -time_warping_para, time_warping_para, tf.int32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, tf.float32)

    # warp
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

    warped_image = tfa.image.sparse_image_warp(mel_spectrogram,
                                        source_control_point_locations,
                                        dest_control_point_locations)[0].numpy()

    warped_image_l1 = tfa.image.sparse_image_warp(mel_spectrogram_l1,
                                        source_control_point_locations,
                                        dest_control_point_locations)[0].numpy()

    warped_image[0,:,:] = warped_image.mean(axis=0)
    warped_image[1,:,:] = warped_image_l1.mean(axis=0)
    return warped_image

def time_masking(mel_spectrogram, time_masking_para=20, time_mask_num=1):
    """
    # Arguments:
        mel_spectrogram(numpy array): spectrogram
        time_masking_para(float): Augmentation parameter
            If none, default = 20
        time_mask_num(float): number of time masking lines applied on the secptrogram
            If none, default = 1
    """
    # Step 2 : time masking
    sample_shape= tf.shape(mel_spectrogram)
    n, v = sample_shape[1], sample_shape[2]

    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        v = tf.cast(v, tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=v - t, dtype=tf.int32)

        mask = tf.concat((tf.ones(shape=(1, n, v - t0 - t)),
                          tf.zeros(shape=(1, n, t)),
                          tf.ones(shape=(1, n, t0)),
                          ), 2)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, tf.float32)


def frequency_masking(mel_spectrogram, mu, frequency_masking_para=10, frequency_mask_num=1):
    """
    # Arguments:
        mel_spectrogram(numpy array): spectrogram
        frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
            If none, default = 10
        frequency_mask_num(float): number of frequency masking lines, "m_F".
            If none, default = 1
    """
    sample_shape = tf.shape(mel_spectrogram)
    n, v = sample_shape[1], sample_shape[2]

    # Step 3 : Time masking
    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=mu - f, dtype=tf.int32)

        mask = tf.concat((tf.ones(shape=(1, n - f0 - f, v)),
                          tf.zeros(shape=(1, f, v)),
                          tf.ones(shape=(1, f0, v)),
                          ), 1)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, tf.float32)

def spectro_augmentation(mel_spectrogram): 
    while True:
        try:
            warped_mel_spectrogram = sparse_warp(mel_spectrogram, 150)
            break
        except:
            pass
    warped_frequency_spectrogram = frequency_masking(warped_mel_spectrogram, 30)
    warped_frequency_time_sepctrogram = time_masking(warped_frequency_spectrogram, 300)
    return warped_frequency_time_sepctrogram

def augment(spectrogram:list):
    s = spectro_augmentation(np.array([v.T for v in spectrogram])).numpy()
    return [s[i,:,:].T for i in range(s.shape[0])]
