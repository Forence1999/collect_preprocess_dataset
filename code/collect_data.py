import os
import sys
import wave
import numpy as np
import soundfile as sf
from pyaudio import PyAudio, paInt16
from lib.audiolib import audiowrite, audioread
from lib.utils import standard_normalizaion, add_prefix_and_suffix_4_basename

RECORD_DEVICE_NAME = "USB Camera-B4.09.24.1"
RECORD_SECONDS = 60  # 180
CHANNELS = 4
SAMPLE_RATE = 16000
RECORD_WIDTH = 2
CHUNK = 1024
Voice_String = []
MICRO_MAPPING = np.array(range(CHANNELS))
DEVICE_INDEX = -1
LOCATION_NUM = 2
DIRECTION = 8
DIRECTION_MAPPING = [0, 45, 90, 135, 180, 225, 270, 315, ]

coordinate_mapping = [[60, 425],
                      [160, 320],
                      [340, 425],
                      [530, 320],
                      [215, 220],
                      [170, 160],
                      [220, 100],
                      [280, 160],
                      [220, 15],
                      [460, 15],
                      [420, 220],
                      [160, 425],
                      [530, 425],
                      [280, 220],
                      [280, 100],
                      [280, 15],
                      [160, 220],
                      [530, 220],
                      [170, 100],
                      [550, 15]
                      ]


def get_device_index():
    device_index = -1
    
    # scan to get usb device
    p = PyAudio()
    print('num_device:', p.get_device_count())
    for index in range(p.get_device_count()):
        info = p.get_device_info_by_index(index)
        device_name = info.get("name")
        print("device_name: ", device_name)
        
        # find mic usb device
        if device_name.find(RECORD_DEVICE_NAME) != -1:
            device_index = index
            break
    
    if device_index != -1:
        print('-' * 20 + 'Find the device' + '-' * 20 + '\n', p.get_device_info_by_index(device_index), '\n')
        del p
    else:
        print('-' * 20 + 'Cannot find the device' + '-' * 20 + '\n')
        exit()
    
    return device_index


def init_micro_mapping():
    print('Please tap each microphone clockwise from the upper left corner ~ ')
    mapping = [None, ] * 4
    while True:
        for i in range(CHANNELS):
            while True:
                ratio = monitor_audio_and_return_amplitude_ratio(mapping_flag=False)
                idx = np.where(ratio > 0.5)[0]
                if len(idx) == 1 and (idx[0] not in mapping):
                    mapping[i] = idx[0]
                    print(' '.join(['Logical channel', str(i), 'has been set as physical channel', str(mapping[i]),
                                    'Amplitude**2 ratio: ', str(ratio)]))
                    break
        print('Final mapping: ')
        print('Logical channel: ', list(range(CHANNELS)))
        print('Physical channel: ', mapping)
        # break
        
        confirm_info = input('Confirm or Reset the mapping? Press [y]/n :')
        if confirm_info in ['y', '', 'yes', 'Yes']:
            break
        else:
            print('The system will reset the mapping')
            continue
    return np.array(mapping)


def monitor_audio_and_return_amplitude_ratio(mapping_flag):
    frames = monitor_from_4mics(record_seconds=1)
    audio = split_channels_from_frames(frames=frames, num_channel=CHANNELS, mapping_flag=mapping_flag)
    amp2_sum = np.sum(standard_normalizaion(audio) ** 2, axis=1).reshape(-1)
    amp2_ratio = amp2_sum / amp2_sum.sum()
    
    return amp2_ratio


def monitor_from_4mics(record_seconds=RECORD_SECONDS):
    print('-' * 20 + "Start monitoring ...")
    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(RECORD_WIDTH),
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=DEVICE_INDEX)
    # 16 data
    frames = []
    
    for i in range(int(SAMPLE_RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('-' * 20 + "End monitoring ...\n")
    
    return frames


def save_wav_from_frames(filename, frames):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(RECORD_WIDTH)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def save_multi_channel_audio(standard_path, audio, fs=SAMPLE_RATE, norm=True, ):
    for i in range(len(audio)):
        file_path = add_prefix_and_suffix_4_basename(standard_path, suffix='_mic%d' % i)
        audiowrite(file_path, audio[i], sample_rate=fs, norm=norm, target_level=-25, clipping_threshold=0.99)
        # sf.write(file_path, audio[i], samplerate=fs)


def split_channels_from_frames(frames, num_channel=CHANNELS, mapping_flag=True):
    audio = np.frombuffer(b''.join(frames), dtype=np.short)
    audio = np.reshape(audio, (-1, num_channel)).T
    if mapping_flag:
        audio = audio[MICRO_MAPPING]
    return audio


def encode_dir(dataset_dir, walker, source, direction):
    save_dir = os.path.join(dataset_dir, 'initial', 'src_' + str(coordinate_mapping[source - 1][0]) + '_' +
                            str(coordinate_mapping[source - 1][1]),
                            'walker_' + str(coordinate_mapping[walker - 1][0]) + '_1_' +
                            str(coordinate_mapping[walker - 1][1]), str(direction))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'record.wav')
    
    return save_dir, save_path


if __name__ == '__main__':
    DEVICE_INDEX = get_device_index()
    MICRO_MAPPING = init_micro_mapping()
    dataset_dir = './dataset'
    ### collect data
    while True:
        walker = -1
        source = -1
        direction = -1
        while True:
            try:
                walker = int(input('walker number   : '))
                source = int(input('source number   : '))
                direction = int(input('direction number: '))
            except:
                continue
            else:
                break
        # print('walker: ', walker, '\tsource: ', source, '\tDOA: ', DOA, )
        
        confirm_info = input('Confirm to collect data? Press [y]/n :')
        if confirm_info in ['n']:
            continue
        else:
            print('\n' + '-' * 20 + 'Collecting data...')
        save_dir, save_path = encode_dir(dataset_dir, walker, source, direction)
        frames = monitor_from_4mics()
        save_wav_from_frames(save_path, frames)
        
        audio = split_channels_from_frames(frames, mapping_flag=True)
        save_multi_channel_audio(save_path, audio, fs=SAMPLE_RATE, norm=False, )
