import os

from tqdm import tqdm as tqdm
import librosa
import numpy as np
from argparse import ArgumentParser
from skvideo.io import ffprobe
import librosa.display
import concurrent.futures
from glob import glob


def extract_melspec(task):
    src_wav, src_vid, dst_fold, target_time_shape, n_fft_coef, n_mels, skip_frames_counter, window_duration = task
    sampling_rate = librosa.get_samplerate(src_wav)
    audio_wave, _ = librosa.load(src_wav, sr=sampling_rate)
    output_shapes = []
    videometadata = ffprobe(src_vid)
    video_length = int(videometadata["video"]["@nb_frames"])
    frame_rate, denom = map(float, videometadata['video']['@avg_frame_rate'].split("/"))
    frame_rate = frame_rate / denom
    if np.all(audio_wave == .0):
        return -1, None
    samples_per_video_frame = sampling_rate // frame_rate
    audio_frame_width = samples_per_video_frame // 2
    os.makedirs(dst_fold, exist_ok=True)
    for i in [i - 1 if i != 0 else i for i in range(0, video_length, skip_frames_counter)]:
        output_frame_path = os.path.join(dst_fold, "{:05d}.npy".format(i + 1))
        current_frame_center = (i * samples_per_video_frame + audio_frame_width)
        start_slice_pos = int(max(current_frame_center - (window_duration // 2) * sampling_rate , 0))
        finish_slice_pos = int(min(current_frame_center + (window_duration // 2) * sampling_rate, len(audio_wave)))
        audio_frame = audio_wave[start_slice_pos:finish_slice_pos]
        if len(audio_frame) == 0:
            print("Zero slice length, oops")
            continue
        hop_length = max(len(audio_frame) // target_time_shape, 1)
        n_fft = hop_length * n_fft_coef
        M = librosa.feature.melspectrogram(audio_frame, sr=sampling_rate, hop_length=hop_length, n_fft=n_fft,
                                           n_mels=n_mels)
        M = librosa.core.power_to_db(M)
        # fig, ax = plt.subplots()
        # img = librosa.display.specshow(M, y_axis='mel', x_axis='time', ax=ax, sr=sampling_rate)
        # fig.colorbar(img, format="%+2.f dB")
        # os.makedirs(os.path.join(dst_fold, "vis"), exist_ok=True)
        # plt.savefig(os.path.join(dst_fold, "vis", "{:05d}.png".format(i + 1)))
        if output_shapes and M.shape != output_shapes[0]:
            print("Oops")
            continue
        output_shapes.append(M.shape)
        with open(output_frame_path, "wb") as fp:
            np.save(fp, M, allow_pickle=False)
    return 0, output_shapes

def convert_dump_audio_folder(input_path, output_path, video_path, target_time_shape=64, n_fft_coef=4, n_mels=64, skip_frames_counter=5, window_duration=2):
    tasks = []
    for root, _, files in os.walk(input_path):
        for file in tqdm(files):
            output_folder_path = os.path.join(output_path, os.path.splitext(file)[0])
            video_file = glob(os.path.join(video_path, file[:-4] + "*"))[0]
            file = os.path.join(root, file)
            if os.path.splitext(file)[1] != ".wav":
                continue
            tasks.append((file, video_file, output_folder_path, target_time_shape - 1, n_fft_coef, n_mels, skip_frames_counter, window_duration))
    n_complete, total = 0, len(tasks)
    overall_output_shapes = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
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
    parser.add_argument("--skip_frames_counter", "-sf", type=int, default=5, help="We take every skip_frames_counter from the dataset (for CNN)")
    parser.add_argument("--target_time_shape", "-ts", type=int, default=122, help="Target time shape, used to calculate hop_length")
    parser.add_argument("--n_fft_coef", type=int, default=4, help="We apply n_fft equal to hop_length * n_fft_coef")
    parser.add_argument("--n_mels", type=int, default=122, help="Number of mel filters")
    parser.add_argument("--window_duration", type=int, default=2, help="Seconds of audio to crop for each anchor frame")
    args = parser.parse_args()
    convert_dump_audio_folder(**vars(args))
    # for root, dirs, _ in os.walk(args.output_path):
    #     output_shapes = []
    #     mels = []
    #     for dir in tqdm(dirs):
    #         files = os.listdir(os.path.join(root, dir))
    #         for file in files[::2]:
    #             mel = np.load(os.path.join(root, dir, file))
    #             output_shapes.append(mel.shape)
    #             mels.append(mel)
    #     output_shapes = np.array(output_shapes)
    #     print(np.unique(output_shapes, return_counts=True))
    #     print(np.mean(mels))
    #     print(np.std(mels))

