import streamlit as st

import pyaudio
import wave

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob
import os
import gc
import pandas as pdru
import os
import sys
from PIL import Image
from keras.models import load_model


_ = '''def record_wav(F"input.wav", Record_Seconds=5):
    rec_time = 5  # 録音時間[s]
    file_path = "/wavfiles/"  # 音声を保存するファイル名
    fmt = pyaudio.paInt16  # 音声のフォーマット
    ch = 1  # チャンネル1(モノラル)
    sampling_rate = 44100  # サンプリング周波数
    chunk = 2 ** 11  # チャンク（データ点数）
    audio = pyaudio.PyAudio()
    index = 1  # 録音デバイスのインデックス番号（デフォルト1）

    stream = audio.open(format=fmt, channels=ch, rate=sampling_rate, input=True,
                        input_device_index=index,
                        frames_per_buffer=chunk)
    # 録音処理
    frames = []
    for i in range(0, int(sampling_rate / chunk * rec_time)):
        data = stream.read(chunk)
        frames.append(data)

    wav = wave.open(file_path, 'wb')
    wav.setnchannels(ch)
    wav.setsampwidth(audio.get_sample_size(fmt))
    wav.setframerate(sampling_rate)
    wav.writeframes(b''.join(frames))
    wav.close()


def analyze_audio():
    dataset_list = "D:/nonnon/データセット/VOI-See_test/テスト用データ/*.wav"  # DBのBLOBのなかのWAVファイルをdataset_listという名前を付けて呼び出す。
    audio_path = glob.glob(dataset_list)  # globを使ってdataset_listの中身をaudio_pathという名前のリストとして扱わせる。
    # print(audio_path) #リストの中身を確認するとき用。

    # リストを100個ごとに分割
    length = len(audio_path)
    n = 0
    s = 100
    for i in audio_path:
        s_audio_path = audio_path[n:n + s:1]
        n += s
        # print(s_audio_path)

        for filename in s_audio_path:  # s_audio_pathをfilenameという変数に代入
            y, sr = librosa.load(filename, sr=None)  # librosaで音声ファイルを㏛(サンプリングレート)＝○○kHzとして読み込む.

            # メル周波数スペクトログラム変換の処理(メル周波数スペクトログラムに関するいろんな数値の設定)
            win_length = 2048  # win_length = 音声をどれぐらいの長さずつ区切るか。
            hop_length = win_length // 4  # hop_length = 一行上で区切ったものを、どれぐらい分ずらして繋げていくか。
            n_fft = win_length * 8  # n_fft = 周波数の幅。
            window = "hann"  # 適用する窓関数の種類。
            n_mels = 128
            mel_power = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                       win_length=win_length, window=window, center=True, n_mels=n_mels)
            mel_power_in_db = librosa.power_to_db(mel_power, ref=np.max)
            # print(f'mel_power.shape: {mel_power.shape}')

            # 変換結果の描画と画像化
            fig = plt.figure(figsize=(6.4, 4.8 / 2))
            ax = fig.add_subplot()
            librosa.display.specshow(mel_power_in_db, x_axis="time", y_axis="mel", sr=sr)  # x軸＝時間、y軸＝メル尺度
            plt.colorbar(format="%+2.0f dB")  # 描画色の設定。
            ax.set_title("mel-frequency power spectrogram")
            # plt.show()　#スペクトログラムを見るとき用。

            # スペクトログラム画像の保存
            new_dir_path = "D:/nonnon/データセット/VOI-See_test/入力データ/"
            os.makedirs(new_dir_path, exist_ok=True)  # 生成した画像の保存先フォルダの作成
            new_filename = os.path.basename(filename)  # 元の音声データのファイル名を取得して、ファイル名のみと拡張子に分割。
            new_filename_no_extension = os.path.splitext(filename)[0]  # 元の音声データから拡張子を除いたファイル名を取得。
            new_filename_solo = new_filename_no_extension.split("\\")[-1]
            # print(new_filename_solo) #ファイル名を取得できているか確認するとき用
            imgname = new_dir_path + new_filename_solo + ".png"  # 保存先フォルダのパスとフォルダ名と画像用の拡張子をimgnameという変数にまとめる。
            plt.savefig(imgname)  # 描画された画像を指定したフォルダに、元ファイルのファイル名を引き継いで保存。

            # pltの描画を閉じる
            plt.close()
            plt.clf()
            plt.cla()
            # gc.collect()

            voice_data = imgname
            image = Image.open(voice_data)
            image = image.resize((64, 64))
            image = image.convert("L").convert("RGB")
            image.show

            np_image = np.array(image)
            np_image = np_image / 255
            np_image = np_image[np.newaxis, :, :, :]

            # model_age = load_model("model.h5")
            # age_result = model_age.predict(np_image)
            # print(age_result)

            model_cute = load_model("cute.h5")
            cute_result = model_cute.predict(np_image)
            print(cute_result)

            model_cool = load_model("cool.h5")
            cool_result = model_cool.predict(np_image)
            print(cool_result)

        if n >= length:
            break
'''

def main():
    st.title("VOI-See β版")
    st.header("ようこそ【VOI-See】へ。あなたの声を聞かせてください")
    st.write("VOI-Seeとは、ディープラーニングの技術を用いて、あなたの声のユニークさを、目に見え手に取れる形へと変化させることを目標にサービス展開するプロジェクトです。")
    st.write("このベータ版では、あなたの声をwebブラウザから録音し、AIに分析させることで、CUTEさとCOOLさを数値化します。")
    radio = st.radio(label='あなたに当てはまるものを選択してください。(必須)',
                      options=('(未選択)', '39歳までの男性', '39歳までの女性', '40歳以上の男性', '40歳以上の女性'),
                      index=0,
                      horizontal=False,)
    if radio == '(未選択)':
        st.warning('選択は必須ですm(__)m')
        # 条件を満たないときは処理を停止する
        st.stop()
    else:
        st.write("録音スタートボタンを押して、「ボイシー、私の声を聴いて。」と語りかけてください。")
        if st.button("録音スタート"):
            record_comment = st.empty()
            record_comment.write("録音しています…")
            #record_wav(Record_Seconds = 5)
            record_comment.write("録音完了しました！")
            if st.button("分析開始"):
                analyze_comment = st.empty()
                analyze_comment.write("分析中…")
                #analyze_audio()
                analyze_comment.write("分析結果はこちらです！")
            elif st.button("録音をやり直す"):
                st.stop()



if __name__ == "__main__":
    main()