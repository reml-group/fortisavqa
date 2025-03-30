import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip
import multiprocessing

'''
Function: Extract audio files(.wav) from videos with multiprocessing.
'''


def get_audio_wav(name, save_pth, audio_name):
    video = VideoFileClip(name)
    audio = video.audio
    audio.write_audiofile(os.path.join(save_pth, audio_name), fps=16000)


def aud_process(params):
    video_pth = params['video_pth']
    save_pth = params['save_pth']

    sound_list = os.listdir(video_pth)
    sound_list.sort()
    for audio_id in sound_list:
        name = os.path.join(video_pth, audio_id)
        audio_name = audio_id[:-4] + '.wav'
        exist_lis = os.listdir(save_pth)
        if audio_name in exist_lis:
            print("already exist!")
            continue
        try:
            get_audio_wav(name, save_pth, audio_name)
            print("finish video id: " + audio_name)
        except:
            print("cannot load ", name)


def pool_process(video_pth, save_pth, pool=None):
    params = {'video_pth': video_pth, 'save_pth': save_pth}
    request_param = []

    if pool is None:
        aud_process(params)
    else:
        request_param.append(params)

    if pool is not None:
        pool.map(aud_process, request_param)
        pool.close()
        pool.join()


if __name__ == "__main__":
    video_pth =  "/home/majie/datahub/MUSIC_AVQA_R/video/"
    save_pth =  "/home/majie/datahub/MUSIC_AVQA_R/audio/"

    # multiprocessing
    cpu_count = multiprocessing.cpu_count()  # cpu nums, 获取CPU核数
    process_count = cpu_count * 2 - 1  # thread nums, 获取最大线程数
    pool = multiprocessing.Pool(process_count)

    pool_process(video_pth, save_pth, pool)




