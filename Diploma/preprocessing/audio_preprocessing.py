import os
from tqdm import tqdm as tqdm
import librosa
import scipy
import numpy as np
from argparse import ArgumentParser
from skvideo.io import ffprobe
import concurrent.futures
from glob import glob

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0))
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate)
    while True:
        if offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n
        else:
            yield Frame(audio[offset:], timestamp, (len(audio) - offset) / float(sample_rate))
            break


def extract_melspec(task):
    src_wav, src_vid, dst_fold, target_time_shape, n_fft_coef, n_mels, skip_frames_counter = task
    output_shapes = []
    sampling_rate = librosa.get_samplerate(src_wav)
    audio_wave, _ = librosa.load(src_wav, sr=sampling_rate)
    videometadata = ffprobe(src_vid)
    frame_rate, denom = map(float, videometadata['video']['@avg_frame_rate'].split("/"))
    frame_rate = frame_rate // denom
    if np.all(audio_wave == .0):
        return -1, None
    # audio_wave_wiener = scipy.signal.wiener(audio_wave)
    audio_frames = frame_generator(1000 * skip_frames_counter / frame_rate, audio_wave, sampling_rate)
    i = skip_frames_counter
    os.makedirs(dst_fold, exist_ok=True)
    for audio_frame in audio_frames:
        output_frame_path = os.path.join(dst_fold, "{:05d}.npy".format(i))
        i += skip_frames_counter
        audio_slice = audio_frame.bytes
        hop_length = max(len(audio_slice) // target_time_shape, 1)
        n_fft = hop_length * n_fft_coef
        M = librosa.feature.melspectrogram(audio_slice, sr=sampling_rate, hop_length=hop_length, n_fft=n_fft,
                                           n_mels=n_mels)
        M = librosa.core.power_to_db(M)
        if output_shapes and M.shape != output_shapes[0]:
            continue
        output_shapes.append(M.shape)
        with open(output_frame_path, "wb") as fp:
            np.save(fp, M, allow_pickle=False)
    return 0, output_shapes


def convert_dump_audio_folder(input_path, output_path, video_path, target_time_shape=64, n_fft_coef=4, n_mels=64, skip_frames_counter=5):
    tasks = []
    for root, _, files in os.walk(input_path):
        for file in tqdm(files):
            output_folder_path = os.path.join(output_path, os.path.splitext(file)[0])
            video_file = glob(os.path.join(video_path, file[:-4] + "*"))[0]
            file = os.path.join(root, file)
            if os.path.splitext(file)[1] != ".wav":
                continue
            tasks.append((file, video_file, output_folder_path, target_time_shape, n_fft_coef, n_mels, skip_frames_counter))
    n_complete, total = 0, len(tasks)
    overall_output_shapes = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        for task, result in zip(tasks, executor.map(extract_melspec, tasks)):
            n_complete += 1
            if result[0] < 0:
                print('Finished {}, result: {}, progress: {}/{}'.format(task[0], result[0], n_complete, total))
            elif result[0] == 0:
                print('Finished {}, result: {}, progress: {}/{}'.format(task[0], result[0], n_complete, total))
                overall_output_shapes.extend(result[1])
    print(np.unique(overall_output_shapes, return_counts=True))



if __name__ == "__main__":
    parser = ArgumentParser("Python3 Script to make mel-histograms from .wav files slices")
    parser.add_argument("--input_path", "-i", type=str, help="Path to folder with .wav files")
    parser.add_argument("--output_path", "-o", type=str, help="Path to output folder")
    parser.add_argument("--video_path", "-v", type=str, help="Path to video folder")
    parser.add_argument("--skip_frames_counter", "-sf", type=int, default=5, help="Slice .wav into segments with 1000 * skip_frames_counter / 30 (FPS) ms length to convert them into Mel")
    parser.add_argument("--target_time_shape", "-ts", type=int, default=63, help="Target time shape, used to calculate hop_length")
    parser.add_argument("--n_fft_coef", type=int, default=4, help="We apply n_fft equal to hop_length * n_fft_coef")
    parser.add_argument("--n_mels", type=int, default=64, help="Number of mel filters")
    args = parser.parse_args()
    convert_dump_audio_folder(**vars(args))
    # for root, dirs, _ in os.walk(args.output_path):
    #     output_shapes = []
    #     mels = []
    #     for dir in tqdm(dirs):
    #         files = os.listdir(os.path.join(root, dir))
    #         for file in files:
    #             mel = np.load(os.path.join(root, dir, file))
    #             output_shapes.append(mel.shape)
    #             mels.append(mel)
    #     output_shapes = np.array(output_shapes)
    #     print(np.unique(output_shapes, return_counts=True))

