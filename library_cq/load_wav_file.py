import pickle
import os
import torchaudio
import torch

music_path = "C:/Users/cq/AIST_data/AIST_audio"

def audio_to_tensor(filename):
    time_frame = 20
    audio_file = os.path.join(music_path, filename)
    metadata = torchaudio.info(audio_file)
    print(metadata)
    audios = torchaudio.load(audio_file)
    print(audios)
    audios = torchaudio.transforms.Resample(audios[1], 16000)(audios[0])
    print(audios.size())
    audio_data = torch.sum(torch.as_tensor(audios), dim=0)/2
    print(audio_data.size())

    if(len(audio_data)/16000<time_frame):
        new_audio_data = torch.zeros([time_frame*16000])
        new_audio_data[0:len(audio_data)] = audio_data
        audio_data = new_audio_data
    else:
        #采样 xxx 
        audio_data = torch.as_tensor(audio_data.numpy())[:16000*time_frame]
    #audio_data = th.as_tensor(audios[0])
    #audio_feature = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=1024, hop_length=256)(audio_data)
    audio_feature = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=1024, win_length=1024, hop_length=256)(audio_data)
    print(audio_feature.shape)
    return audio_feature

audio_to_tensor('gBR_sBM_cAll_d05_mBR0_ch01.wav')