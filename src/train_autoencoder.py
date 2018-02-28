from autoencoder import *
from speech_feature import *
from embedding import *
from collections import defaultdict
from collections import Counter
import numpy as np
import os
from os import walk
import sys
import time
import random
import json

feat_extractor = speech_feat()

class Ace291():
    def __init__(self):
        self.n_epoch = 1000
        self.n_hidden_layers = [500, 180, 120]
        self.feat_size = 12
        self.learning_rate = 1e-5
        self.frames = 100
        self.batch_size = 50
        self.n_channel = 1
        self.shift = 5
        self.feat_fn = feat_extractor.mfcc
        self.normalize = True
        self.pretrain_epoch = 10
        self.istesting = True
        self.pretrain = True
        self.load_pretrain = True
        
        start = time.time()
        self.check_input()
        data_folder = sys.argv[1]
        
        file_names = []
        self.get_file_under_folder(data_folder, file_names)
        
        audioData, audioLens = {}, {}
        vocab = defaultdict(list)
        
        self.read_data(audioData, audioLens, file_names)
        
        audioFiles = list(audioData.keys())
        print('Data reading done.\t Spend %.4f seconds' % (time.time() - start))
        
        start = time.time()
        self.data_preprocessing(audioFiles, audioLens, audioData)
        print('Shape of one audio file: ', str(audioData[audioFiles[0]].shape))
        print('Data preprocessing done.\t Spend %.4f seconds' % (time.time() - start))
        
        '''All data strcuture is defined in function pack_data)'''
        self.pack_data(audioFiles, audioData, audioLens)

        def repack_data():
            self.pack_data(audioFiles, audioData, audioLens)
        
        start = time.time()
        print('Shape of training data: ', str(self.trainData.shape))
        print('Data preparation done.\t Spend %.4f seconds' % (time.time() - start))

        start = time.time()
        autoencoder = Autoencoder(self.n_hidden_layers, self.feat_size, self.learning_rate, 
                                  self.frames, self.batch_size, 
                                  pretrain_epoch=self.pretrain_epoch, load=self.istesting)
        autoencoder.initPlaceholders()
        autoencoder.initParams()
        autoencoder.model()
        autoencoder.calculateCost()
        
        print('Model declaration done.\t Spend %.4f seconds' % (time.time() - start))
        if not self.istesting:
            autoencoder.training(self.n_epoch, self.trainData, self.devData, repack_data, self.pretrain, self.load_pretrain) 
        else:
            self.extract_representation(audioFiles, audioData, autoencoder)

    def check_input(self):
        assert len(sys.argv) == 2, \
            'ERROR: data folder is the only one argument. Get %s' % (' '.join(sys.argv))

    def get_file_under_folder(self, folder, file_list):
        for (dirpath, dirnames, filenames) in walk(folder):
            file_list.extend([os.path.join(dirpath, fname) for fname in filenames])

    def read_data(self, audioData, audioLens, file_names):
        feat_size = self.feat_size
        for file_name in file_names:
            if '.flac' in file_name or '.wav' in file_name:
                with open(file_name, 'rb') as f:
                    data, samplerate = sf.read(f)
                audio_feat = self.feat_fn(data, samplerate)#, nfilt=self.feat_size)
                audioData[os.path.basename(file_name)] = audio_feat
                audioLens[os.path.basename(file_name)] = audio_feat.shape[0]
    
    def data_preprocessing(self, audioFiles, audioLens, audioData):
        feat_size = self.feat_size
        zero_feat = np.zeros((feat_size, ))
        mean = np.zeros((1, feat_size))
        std = np.zeros((1, feat_size))
        n = 0
        for audioFile in audioFiles:
            if audioLens[audioFile] < self.frames:
                n_lack = self.frames - audioLens[audioFile]
                audioData[audioFile] = np.concatenate([audioData[audioFile], [zero_feat] * n_lack], axis=0)
            mean += np.sum(audioData[audioFile], axis=0, keepdims=True) 
            std += np.sum(audioData[audioFile]**2, axis=0, keepdims=True) 
            n += audioData[audioFile].shape[0]
     
        if not self.normalize:
            return
        mean = mean / n
        std = std / n - mean**2
        if os.path.isfile('./mean') and os.path.isfile('./std'):
            if np.array(json.load(open('./mean', 'r', encoding='utf8'))).shape == mean.shape:
               mean = np.array(json.load(open('./mean', 'r', encoding='utf8')))
            if np.array(json.load(open('./std', 'r', encoding='utf8'))).shape == std.shape:
                std = np.array(json.load(open('./std', 'r', encoding='utf8')))
        else:
            json.dump(mean.tolist(), open('./mean', 'w', encoding='utf8'))
            json.dump(std.tolist(), open('./std', 'w', encoding='utf8'))
        for audioFile in audioFiles:
            audioData[audioFile] = (audioData[audioFile] - (mean + 1e-8))/std
    
    def pack_data(self, audioFiles, audioData, audioLens):
        self.trainData = []
        for audioFile in audioFiles:
            n = audioData[audioFile].shape[0]
            for i in range(0, n, int(self.shift)):
                if i + self.frames > n:
                    break
                self.trainData.append(np.array(audioData[audioFile][i: i + self.frames]))

        n_devData = int(len(self.trainData) * 0.2)
        self.devData = np.array(self.trainData[-n_devData:])
        self.trainData = np.array(self.trainData[:-n_devData])

    def extract_representation(self, audioFiles, audioData, autoencoder):
        encoding_dir = './extract_encoding/'
        if not os.path.isdir(encoding_dir):
            os.mkdir(encoding_dir)
        for audioFile in audioFiles:
            audioEncoding = []
            audioFeatures = []
            n = audioData[audioFile].shape[0]
            assert n != 0, audioData[audioFile].shape
            for i in range(0, n, int(self.shift)):
                if i + self.frames > n:
                    break
                audioFeatures.append(np.array(audioData[audioFile][i: i + self.frames]))
            audioFeatures = np.array(audioFeatures)
            #print(np.array(audioFeatures).shape, end='')
            assert audioFeatures.shape[0] != 0, audioFeatures.shape
            audioEncoding.extend(autoencoder.extract(audioFeatures).tolist())
            print(np.array(audioEncoding).shape)
            json.dump(audioEncoding, open(encoding_dir + audioFile + '.json', 'w', encoding='utf8'))
        print('Extraction of encoding is done')

ace = Ace291()
