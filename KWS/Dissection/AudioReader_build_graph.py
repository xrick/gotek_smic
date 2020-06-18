def prepare_processing_graph(self):
    """Builds a TensorFlow graph to apply the input distortions.
    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.
    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:
      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - mfcc_: Output 2D fingerprint of processed audio.
    Args:
      model_settings: Information about the current model being trained.
    """
    desired_samples = self.model_settings['desired_samples']
    self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])

    wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
    # Allow the audio sample's volume to be adjusted.
    self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
    scaled_foreground = tf.multiply(wav_decoder.audio, self.foreground_volume_placeholder_)

    # Shift the sample's start position, and pad any gaps with zeros.
    self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
    self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
    padded_foreground = tf.pad(scaled_foreground, self.time_shift_padding_placeholder_, mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground, self.time_shift_offset_placeholder_, [desired_samples, -1])
    # Mix in background noise.
    self.background_data_placeholder_ = tf.placeholder(tf.float32, [desired_samples, 1])
    self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
    background_mul = tf.multiply(self.background_data_placeholder_, self.background_volume_placeholder_)
    background_add = tf.add(background_mul, sliced_foreground)
    background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = contrib_audio.audio_spectrogram(background_clamp,
                                                  window_size=self.model_settings['window_size_samples'],
                                                  stride=self.model_settings['window_stride_samples'],
                                                  magnitude_squared=True)
    self.mfcc_ = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate,
                                    dct_coefficient_count=self.model_settings['dct_coefficient_count'])
    num_spectrogram_bins = spectrogram.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, self.model_settings['dct_coefficient_count']
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, self.model_settings['sample_rate'], lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    self.mel_ = mel_spectrograms
    self.log_mel_ = tf.log(mel_spectrograms + 1e-6)