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

feat_extractor = speech_feat()

class Ace291():
    def __init__(self):
        self.n_epoch = 1000
        self.n_hidden_layers = [500, 180, 120]
        self.feat_size = 40
        self.learning_rate = 1e-4
        self.frames = 100
        self.batch_size = 50
        self.n_channel = 1
        self.feat_fn = feat_extractor.logfbank
        self.normalize = True
        
        start = time.time()
        self.check_input()
        data_folder = sys.argv[1]
        
        file_names = []
        self.get_file_under_folder(data_folder, file_names)
        
        audioData, audioLens, audioTran = {}, {}, {}
        vocab = defaultdict(list)
        
        self.read_data(audioData, audioLens, audioTran, vocab, file_names)
        
        audioFiles = list(audioData.keys())
        print('Data reading done.\t Spend %.4f seconds' % (time.time() - start))
        
        start = time.time()
        self.data_preprocessing(audioFiles, audioLens, audioData)
        print('Shape of one audio file: ', str(audioData[audioFiles[0]].shape))
        print('Data preprocessing done.\t Spend %.4f seconds' % (time.time() - start))
        
        '''All data strcuture is defined in function pack_data)'''
        self.pack_data(audioFiles, audioData, audioLens, audioTran)

        def repack_data():
            self.pack_data(audioFiles, audioData, audioLens, audioTran)
        
        start = time.time()
        print('Shape of training data: ', str(self.trainData.shape))
        print('Data preparation done.\t Spend %.4f seconds' % (time.time() - start))

        start = time.time()
        autoencoder = Autoencoder(self.n_hidden_layers, self.feat_size, self.learning_rate, 
                                  self.frames, self.batch_size)
        autoencoder.initPlaceholders()
        autoencoder.initParams()
        autoencoder.model()
        autoencoder.calculateCost()
        
        print('Model declaration done.\t Spend %.4f seconds' % (time.time() - start))
        autoencoder.training(self.n_epoch, self.trainData, self.devData, repack_data, True, True) 

    def check_input(self):
        assert len(sys.argv) == 2, \
            'ERROR: data folder is the only one argument. Get %s' % (' '.join(sys.argv))

    def get_file_under_folder(self, folder, file_list):
        for (dirpath, dirnames, filenames) in walk(folder):
            file_list.extend([os.path.join(dirpath, fname) for fname in filenames])

    def read_data(self, audioData, audioLens, audioTran, vocab, file_names):
        feat_size = self.feat_size
        for file_name in file_names:
            if '.txt' in file_name:
                path = os.path.dirname(file_name)
                with open(file_name, 'r') as f:
                    text = f.readlines()
                text = text if text[-1] != '' else text[:-1]
                lines = [line.lower().split() for line in text]
                for line in lines:
                    audioTran[line[0] + '.flac'] = line[1:]
                    for word in line[1:]:
                        vocab[word].append(line[0])
            elif '.flac' in file_name:
                with open(file_name, 'rb') as f:
                    data, samplerate = sf.read(f)
                audio_feat = self.feat_fn(data, samplerate, nfilt=feat_size)
                #delta_feat = feature_extractor(
                audioData[os.path.basename(file_name)] = audio_feat
                audioLens[os.path.basename(file_name)] = audio_feat.shape[0]
        assert sum([1 for audioFile in audioData if audioFile not in audioTran])==0
    
    def data_preprocessing(self, audioFiles, audioLens, audioData):
        feat_size = self.feat_size
        zero_feat = np.zeros((feat_size, ))
        mean = np.zeros((1, feat_size))
        std = np.zeros((1, feat_size))
        n = 0
        for audioFile in audioFiles:
            mean += np.sum(audioData[audioFile], axis=0, keepdims=True) 
            std += np.sum(audioData[audioFile]**2, axis=0, keepdims=True) 
            n += audioData[audioFile].shape[0]
     
        if not self.normalize:
            return
        mean = mean / n
        std = std / n - mean**2
        for audioFile in audioFiles:
            audioData[audioFile] = (audioData[audioFile] - (mean + 1e-8))/std
    
    def pack_data(self, audioFiles, audioData, audioLens, audioTran):
        self.trainData = []
        for audioFile in audioFiles:
            n = audioData[audioFile].shape[0]
            for i in range(0, n, int(self.frames/4)):
                if i + self.frames > n:
                    break
                self.trainData.append(np.array(audioData[audioFile][i: i + self.frames]))

        n_devData = int(len(self.trainData) * 0.2)
        self.devData = np.array(self.trainData[-n_devData:])
        self.trainData = np.array(self.trainData[:-n_devData])

ace = Ace291()
