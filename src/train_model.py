from model import *
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
        self.n_epoch = 20
        self.n_hidden = 100
        self.feat_size = 40
        self.learning_rate = 1e-3
        self.frames = 400
        self.batch_size = 50
        self.n_channel = 1
        self.shift = 5
        self.feat_fn = feat_extractor.logfbank
        self.q_len = 200 # could be set to random between 200~300 later on
        self.max_len = self.frames # equal to 7 seconds
        self.normalize = True
        self.istesting = True
        
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
        #self.data_preprocessing(audioFiles, audioLens, audioData)
        print('Shape of one audio file: ', str(audioData[audioFiles[0]].shape))
        print('Data preprocessing done.\t Spend %.4f seconds' % (time.time() - start))
        
        '''All data strcuture is defined in function pack_data)'''
        self.pack_data(audioFiles, audioData, audioLens, audioTran, self.q_len,self.max_len)
        print(audioData[audioFiles[0]].shape)

        def repack_data():
            self.pack_data(audioFiles, audioData, audioLens, audioTran, self.q_len, self.max_len)
        
        start = time.time()
        print('Shape of training data: ', str(self.trainQue.shape))
        print('Data preparation done.\t Spend %.4f seconds' % (time.time() - start))

        start = time.time()
        autoencoder = Autoencoder(self.n_hidden, self.feat_size, self.learning_rate, 
                                  self.frames, self.batch_size, load=self.istesting)
        autoencoder.initPlaceholders()
        autoencoder.initParams()
        autoencoder.model()
        autoencoder.calculateCost()
        
        print('Model declaration done.\t Spend %.4f seconds' % (time.time() - start))
        if not self.istesting:
            autoencoder.training(self.n_epoch, self.trainData, self.trainLen, 
                                 self.devData, self.devLen, repack_data) 
        else:
            self.extract_representation(audioFiles, audioData, audioLens, autoencoder)

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
            elif '.flac' in file_name or '.wav' in file_name:
                with open(file_name, 'rb') as f:
                    data, samplerate = sf.read(f)
                audio_feat = self.feat_fn(data, samplerate, nfilt=feat_size)
                audioData[os.path.basename(file_name)] = audio_feat
                audioLens[os.path.basename(file_name)] = audio_feat.shape[0]
    
    def data_preprocessing(self, audioFiles, audioLens, audioData):
        max_len = self.max_len
        feat_size = self.feat_size
        zero_feat = np.zeros((feat_size, ))
        mean = np.zeros((1, feat_size))
        std = np.zeros((1, feat_size))
        n = 0
        for audioFile in audioFiles:
            audioLen = audioLens[audioFile]
            #print(audioData[audioFile][:50])
            mean += np.sum(audioData[audioFile], axis=0, keepdims=True) 
            std += np.sum(audioData[audioFile]**2, axis=0, keepdims=True) 
            n += audioData[audioFile].shape[0]
            if audioLen < max_len:
                append_len = max_len - audioLen
                audioData[audioFile] = np.concatenate([audioData[audioFile], 
                                           [zero_feat]*append_len], axis=0)
            else:
                audioData[audioFile] = audioData[audioFile][:max_len]
     
        if not self.normalize:
            return
        mean = mean / n
        std = std / n - mean**2
        for audioFile in audioFiles:
            audioData[audioFile] = (audioData[audioFile] - (mean + 1e-8))/std
    
    def get_indpt_file(self, pos_trans, audioFiles, transcript):
        res = random.choice(audioFiles)
        while len([1 for x in transcript[res] if x in pos_trans]) > 0:
            res = random.choice(audioFiles)
        assert set(pos_trans).isdisjoint(transcript[res]) 
        return res
    
    def pack_data(self, audioFiles, audioData, audioLens, audioTran, q_len, max_len):
        q_len = self.q_len
        self.trainQue = []
        self.trainPos = []
        self.trainNeg = []
        self.trainQueFrames = []
        self.trainPosFrames = []
        self.trainNegFrames = []
        for audioFile in audioFiles:
            if audioLens[audioFile] <= q_len:
                continue
            query_end = np.random.randint(q_len, audioLens[audioFile], (1,))[0]
            query_audio = audioData[audioFile][query_end - q_len: query_end]
     
            indptFile = self.get_indpt_file(audioTran[audioFile], audioFiles, audioTran)
     
            self.trainQue.append(audioData[audioFile])
            self.trainPos.append(audioData[audioFile])
            self.trainNeg.append(audioData[indptFile])
            self.trainQueFrames.append(min(query_end, max_len - 1))
            self.trainPosFrames.append(min(audioLens[audioFile], max_len - 1))
            self.trainNegFrames.append(min(audioLens[indptFile], max_len - 1))

        n_devData = int(len(self.trainPos) * 0.2)
        self.devQue = np.array(self.trainQue[-n_devData:])
        self.devPos = np.array(self.trainPos[-n_devData:])
        self.devNeg = np.array(self.trainNeg[-n_devData:])
        self.devQueFrames = np.array(self.trainQueFrames[-n_devData:])
        self.devPosFrames = np.array(self.trainPosFrames[-n_devData:])
        self.devNegFrames = np.array(self.trainNegFrames[-n_devData:])
        self.trainQue = np.array(self.trainQue[:-n_devData])
        self.trainPos = np.array(self.trainPos[:-n_devData])
        self.trainNeg = np.array(self.trainNeg[:-n_devData])
        self.trainQueFrames = np.array(self.trainQueFrames[:-n_devData])
        self.trainPosFrames = np.array(self.trainPosFrames[:-n_devData])
        self.trainNegFrames = np.array(self.trainNegFrames[:-n_devData])

        self.devData = [self.devQue, self.devPos, self.devNeg]
        self.devLen = [self.devQueFrames, self.devPosFrames, self.devNegFrames]
        self.trainData = [self.trainQue, self.trainPos, self.trainNeg]
        self.trainLen = [self.trainQueFrames, self.trainPosFrames, self.trainNegFrames]

    def extract_representation(self, audioFiles, audioData, audioLens, autoencoder):
        encoding_dir = './extract_encoding_model/'
        if not os.path.isdir(encoding_dir):
            os.mkdir(encoding_dir)
        zero_feat = np.zeros((self.feat_size, ))
        for audioFile in audioFiles:
            audioEncoding = []
            audioFeatures = []
            audioFrameLen = []
            n = audioLens[audioFile]
            assert n != 0, audioData[audioFile].shape
            for i in range(0, n, int(self.shift)):
                end = min(n, i + self.frames)
                feat = np.array(audioData[audioFile])[i: end]
                if end - i < self.max_len:
                    append_len = self.max_len - (end - i)
                    feat = np.concatenate([feat, [zero_feat]*append_len], axis=0)
                print(audioFile, feat.shape)
                audioFeatures.append(feat)
                audioFrameLen.append(end - i)
                if i + self.frames > n:
                    break
            audioFeatures, audioFrameLen = np.array(audioFeatures), np.array(audioFrameLen)
            assert audioFeatures.shape[0] != 0, audioFeatures.shape
            assert len(audioFeatures.shape) == 3 and len(audioFrameLen.shape) == 1
            audioEncoding.extend(autoencoder.extract(audioFeatures, audioFrameLen).tolist())
            print('encoding', np.array(audioEncoding).shape)
            json.dump(audioEncoding, open(encoding_dir + audioFile + '.json', 'w', encoding='utf8'))
        print('Extraction of encoding is done')

ace = Ace291()
