import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import cv2
import torch as th
from torchaudio.transforms import MFCC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Scale
import pdb
import torchaudio

class audio_loader(Dataset):

    def __init__(self, audio_folder, time_frame):
        super(audio_loader, self).__init__()
        self.audio_folder = audio_folder

        self.filenames = []
        self.labels = []
        self.audio_folder = audio_folder
        self.time_frame = time_frame
        for file_name_i in os.listdir(audio_folder):
            self.filenames.append(file_name_i)


    def audio_to_tensor(self, audio_file):
        audios = torchaudio.load(audio_file)
        audios = torchaudio.transforms.Resample(audios[1], 16000)(audios[0])
        audio_data = th.sum(th.as_tensor(audios), dim=0)/2
        if(len(audio_data)/16000<self.time_frame):
            new_audio_data = th.zeros([self.time_frame*16000])
            new_audio_data[0:len(audio_data)] = audio_data
            audio_data = new_audio_data
        else:
            #采样 xxx 
            audio_data = th.as_tensor(audio_data.numpy())[:16000*self.time_frame]
        #audio_data = th.as_tensor(audios[0])
        #audio_feature = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=1024, hop_length=256)(audio_data)
        audio_feature = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=1024, win_length=1024, hop_length=256)(audio_data)
        #print(audio_feature.shape)
        return audio_feature

    def get_audio_data(self, audio_path):
        audio_input =self.audio_to_tensor(audio_path)
        return audio_input

    def __getitem__(self, index):
        audio_name = self.audio_folder + self.filenames[index]
        audio = self.get_audio_data(audio_name)
        return np.array(int(self.filenames[index].split('.')[0])), audio

    def __len__(self):
        return len(self.filenames)
