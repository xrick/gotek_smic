def prepare_data_index(self):
    init_dict = {'validation': [], 'testing': [], 'training': []}
    if not (self.train):
        init_dict = {'testing': []}
    elif self.mode == 'train':
        init_dict = {'validation': [], 'training': []}

    self.data_index = init_dict
    unknown_index = init_dict
    all_words = {}
    for wav_file in self.audio_files:
        if self.train:
            word = get_parent_folder_name(wav_file).lower()
        else:
            word = self.words_list[1]  # Unknown label in case of inference
        # used to remove previous augmentation folder
        aug_dir = os.path.join(os.path.dirname(wav_file), 'augmentation')
        if os.path.isdir(aug_dir):
            shutil.rmtree(aug_dir)
        if word == self.background_noise:
            continue
        all_words[word] = True
        if len(self.validation_list) > 0 and len(self.testing_list) > 0:
            wav_index = get_file_index(wav_file)
            if wav_index in self.validation_list:
                set_index = 'validation'
            elif wav_index in self.testing_list:
                set_index = 'testing'
            else:
                set_index = 'training'
        else:
            set_index = which_set(wav_file, self.validation_percentage, self.testing_percentage,
                                  self.model_settings['max_num_wavs_per_class'])
        if not (self.train):
            set_index = 'testing'  # in case of inference set index will be always testing
        elif self.mode == 'train' and set_index == 'testing':
            # using test set in the training set for production system
            set_index = 'training'
        if word in self.words_list[2:]:
            self.data_index[set_index].append({'label': word, 'file': wav_file})
        else:
            unknown_index[set_index].append({'label': self.words_list[1], 'file': wav_file})
    if self.train:
        for index, wanted_word in enumerate(self.words_list[2:]):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                                ' in labels but only found ' +
                                ', '.join(all_words.keys()))
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in init_dict:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * self.silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': self.words_list[0],  # silence label
                    'file': silence_wav_path
                })
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(
                unknown_index[set_index])  # TODO might need to get examples from all words and all speakers
            # unknown will be sampled within each mini batch
            unknown_size = int(math.ceil(set_size * self.unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])

    if self.train:
        self.augment_data('training')
        for set_index in init_dict:
            random.shuffle(self.data_index[set_index])
    for word in all_words:
        if word in self.words_list[2:]:
            self.word_to_index[word] = self.words_list.index(word)
        else:
            self.word_to_index[word] = self.words_list.index(self.words_list[1])
    self.word_to_index[self.words_list[1]] = 1  # unknown label
    self.word_to_index[self.words_list[0]] = 0  # silence label