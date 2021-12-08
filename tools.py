import os
from moviepy.editor import VideoFileClip
from .gaze_tracking import GazeTracking
import numpy as np
import pandas as pd

class Tools(object):

    def __init__(self, videoname, PreEyebrowHeight, PreEyebrowPitch, PreMouthHeight, PreMouthPitch, PreEyesPitch):
        self.videoname = videoname
        self.pre_EyebrowHeight = PreEyebrowHeight
        self.pre_EyebrowPitch = PreEyebrowPitch
        self.pre_MouthHeight = PreMouthHeight
        self.pre_MouthPitch = PreMouthPitch
        self.pre_EyesPitch = PreEyesPitch
        self.eyebrow_height_list = []
        self.eyebrow_pitch_list = []
        self.mouth_height_list = []
        self.mouth_pitch_list = []
        self.eyes_pitch_list = []


    def doAll(self):
        FileSplit = self.videoname.split('.')
        ResultFile = FileSplit[0]+'_fast'+'.mp4' 

        ResultfilePath, Duration = self.fastenvideo(ResultFile)
        GazeModel = GazeTracking(ResultfilePath)

        DistractRate = self.CollectPredata(GazeModel)

        eyebrow_height_mean = np.mean(self.eyebrow_height_list)
        eyebrow_pitch_mean = np.mean(self.eyebrow_pitch_list)
        mouth_height_mean = np.mean(self.mouth_height_list)
        mouth_pitch_mean = np.mean(self.mouth_pitch_list)

        #是否保存csv檔
        # savecsv()

        # winktime計算
        avgnum = np.mean(self.eyes_pitch_list)
        stdnum = np.std(self.eyes_pitch_list)
        lowbound = avgnum - 2*stdnum
        winktime = 0

        for i in range(0,len(self.eyes_pitch_list)):
            if self.eyes_pitch_list[i] < lowbound and self.eyes_pitch_list[i-1] > lowbound and self.eyes_pitch_list[i-2] > lowbound:
                winktime = winktime+1
        
        return eyebrow_height_mean, eyebrow_pitch_mean, mouth_height_mean, mouth_pitch_mean, winktime , DistractRate


    def fastenvideo(self,resultfile):
        #設定路徑
        FilePath = os.path.join('./media/video/',self.videoname)
        ResultfilePath = os.path.join('./media/video/fast/',resultfile)
        #選取影片+設定時間
        video = VideoFileClip(FilePath)
        duration = video.duration
        #影片加速
        video = video.time_transform(lambda t: 2 * t, apply_to=['mask', 'video', 'audio']).with_duration(duration / 2)
        video.write_videofile(ResultfilePath)

        return ResultfilePath,duration/2


    def delete3std(self,listbefore):
        listafter = listbefore
        var = np.std(listafter)
        mean = np.mean(listafter)
        j = 0
        time = len(listafter)
        
        while True:
            if listafter[j] >= (mean+3*var) or listafter[j] <= (mean-3*var):
                del listafter[j]
                time -= 1
                if j == time:
                    break
                continue
            j += 1
            if j == time:
                break

        return listafter


    def CollectPredata(self,gazemodel):
        eyebrow_height, eyebrow_pitch, mouth_height, mouth_pitch, eyes_pitch, outright, outleft, outcenter = gazemodel.learning_face()

        self.eyebrow_height_list = self.delete3std(eyebrow_height)
        self.eyebrow_height_list = [i/float(self.pre_EyebrowHeight) for i in self.eyebrow_height_list]
        # self.eyebrow_height_list = self.eyebrow_height_list/self.pre_EyebrowHeight

        self.eyebrow_pitch_list = self.delete3std(eyebrow_pitch)
        self.eyebrow_pitch_list = [i/float(self.pre_EyebrowPitch) for i in self.eyebrow_pitch_list]

        # self.eyebrow_pitch_list = self.eyebrow_pitch_list/self.pre_EyebrowPitch

        self.mouth_height_list = self.delete3std(mouth_height)
        self.mouth_height_list = [i/float(self.pre_MouthHeight) for i in self.mouth_height_list]

        # self.mouth_height_list = self.mouth_height_list/self.pre_MouthHeight

        self.mouth_pitch_list = self.delete3std(mouth_pitch)
        self.mouth_pitch_list = [i/float(self.pre_MouthPitch) for i in self.mouth_pitch_list]
        # self.mouth_pitch_list = self.mouth_pitch_list/self.pre_MouthPitch

        self.eyes_pitch_list = [i/float(self.pre_EyesPitch) for i in eyes_pitch]
        # self.eyes_pitch_list = eyes_pitch/self.pre_EyesPitch

        sumnum = outright+outleft+outcenter
        maxnum = max(outright, outleft, outcenter)
        distractrate = maxnum/sumnum
        # print(sumnum, maxnum, distractrate)

        return distractrate
    
    # def savecsv(self):
    #     csv = self.videoname.split('.')
    #     csvName_reverse = csv[0]+'.csv'

    #     d = {'eyebrow_height':self.eyebrow_height_list,'eyebrow_pitch':self.eyebrow_pitch_list,'mouth_height':self.mouth_height_list,'mouth_pitch':self.mouth_pitch_list,'eyes_pitch':self.eyes_pitch_list}
    #     df = pd.DataFrame.from_dict(data = d, orient='index')
    #     df_T = df.T

    #     df_T.to_csv(csvName_reverse)