# -*- coding: utf-8 -*- 
import requests
from google.cloud import speech
import io
import os, sys
import sounddevice as sd
import soundfile as sf
import numpy as np
# import librosa
# import vlc

from scipy.io.wavfile import write
sr = 16000  # Sample rate
seconds = 3  # Duration of recording
filename = 'myfile.wav' #recording of my speech
vaccinations = requests.get('https://api.corona-zahlen.org/vaccinations')
import pickle
pickle.dump(vaccinations.json(), open( "vaccination_data.pkl", "wb" ) )
vaccinations = pickle.load( open( "/Users/shushanamanakhimova/S_Dialog/vaccination_data.pkl", "rb" ) )

phrases = {'hello':'Willkommen bei der Corona Impfauskunft. Fragen Sie!', 
    'continue':'Weiter!', 
    'goodbye':'Vielen Dank für Ihren Besuch!', 
    'done':'fertig', 
    'done':'tschüss'}




states_d = {'schleswig':'SH', 'hamburg':'HH', 'berlin':'BE', 'bayern':'BY', 
            'niedersachsen': 'NI', 'bremen': 'HB', 
            'nordrhein':'NW', 'hessen':'HE', 'rheinland':'RP', 'baden':'BW', 
            'saarland': 'SL', 'brandenburg':'BB', 'mecklenburg':'MV', 'sachsen':'SN',
            'anhalt':'ST', 'thüringen':'TH', 'deutschland':'DE', 'hier':'DE'}
state_names = {'SH':'Schleswig-Hostein', 'HH':'Hamburg', 'BE':'Berlin', 'BY':'Bayern', 
            'NI':'Niedersachsen', 'HB':'Bremen', 
            'NW': 'Nordrhein Westphalen', 'HE':'Hessen', 'RP':'Rheinland Pfalz', 'BW':'Baden Würthenberg', 
            'SL':'Saarland', 'BB':'Brandenburg', 'MV':'Mecklenburg Vorpommern', 
            'SN': 'Sachsen', 'ST':'Sachsen-Anhalt', 'TH':'Thüringen', 'DE':'Deutschland'}
vaccines_d = {'biontech':'biontech', 'biontec':'biontech', 
              'moderna':'moderna', 
              'janssen':'janssen', 'jansen':'janssen',
              'delta':'delta',
              'astraZeneca':'astraZeneca', 'astra':'astraZeneca', 'zeneca':'astraZeneca'}
vaccine_names = {'biontech':'Biontech', 'moderna':'Moderna', 'janssen':'Janssen', 'delta':'Delta',
              'astraZeneca':'Astra Zeneca'}

def init_google():
    credentials='/Users/shushanamanakhimova/S_Dialog/true-sprite-320717-fbd9b9414a32.json'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=credentials

def transcribe(): #transcribing my speech into text
    client = speech.SpeechClient()
    with io.open(filename, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content = content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="de-DE",
    )
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        for index, alternative in enumerate(result.alternatives):
            print("Transcript {}: {}".format(index, alternative.transcript))
            return alternative.transcript
    
# Playback the file
# def play_file(filename, sr):
#     data, fs = sf.read(filename)  
#     sd.play(data, sr)
#     status = sd.wait()  # Wait until file is done playing
# def play_mp3(filename):
#     p = vlc.MediaPlayer(filename)
#     p.play()

# from pydub import AudioSegment
# from pydub.utils import mediainfo

# def mp3_to_wav(infile, outfile):
#     sr = mediainfo(infile)['sample_rate']
#     sound = AudioSegment.from_mp3(infile)
#     sound.export(outfile, format="wav")
#     return sr

# from gtts import gTTS
# def tts(text):
#     voice = gTTS(text, lang='de')
#     audio_file = './her.mp3'
#     voice.save(audio_file)
#     #from pygame import mixer 
#     #wozu braucht man mixer funktion aus pygame, wenn wir schon vlc benutzen?
#     #import vlc
#     #p = vlc.MediaPlayer(audio_file)
#     #p.play()
#     sr = mp3_to_wav(audio_file, 'her.wav')
#     play_file('her.wav', int(sr))
import pyttsx3
def tts(text):
    engine = pyttsx3.init()
    engine.setProperty('voice', 'german')
    engine.setProperty('rate', 200)
    engine.say(text)
    engine.runAndWait()

def speech_input():
    record_file()
    text = transcribe()
    return text

def do_input():
    return speech_input()



def record_file():

    sr = 44100
    duration = 3
    data = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()  
    sf.write(filename, data, sr)
    print(data)
    
    # Convert `data` to 16 bit integers:
    y = (np.iinfo(np.int16).max * (data/np.abs(data).max())).astype(np.int16) 
    
    write(filename, sr, y)
    # y = librosa.load(filename, sr=16000, duration=3.0)
    # librosa.output.write_wav(filename, y, 16000)
    # sf.write(filename, x, sr, subtype='PCM_16')

def normalize(in_s):
    return in_s.lower()


def semantic(input_s):
    semantics = {'state':'', 'vaccine':''}
    for key in states_d.keys():
        if key in input_s:
            semantics['state'] = states_d[key]
    for key in  vaccines_d.keys():
        if key in input_s:
            semantics['vaccine'] =  vaccines_d[key]
    return semantics

# expects semantics: semantics[0] == bundesland, semantics[1] == impfstoff 
# vaccinations = requests.get('https://api.corona-zahlen.org/vaccinations')
def data(semantics, vaccinations):
    s = semantics['state']
    v = semantics['vaccine']
    if s: # state given
        if s != 'DE':
            if v: # and vaccine given
                vacc_number = vaccinations["data"]["states"][s]['vaccination'][v]
            else: # all vaccines for state
                vacc_number = vaccinations["data"]["states"][s]['vaccinated']
        else:
            if v: # and vaccine given
                vacc_number = vaccinations["data"]['vaccination'][v]
            else: # all vaccines for Germany
                vacc_number = vaccinations['data']['vaccinated']
    else: # no state
        if v: # but vaccine
            vacc_number = vaccinations["data"]['vaccination'][v]
        else: # nothing given
            vacc_number = None
    return vacc_number

def output(semantics, results, inputs, eliza):
    ret = ''
    s = semantics['state']
    v = semantics['vaccine']
    if s: # state given
        s = state_names[semantics['state']]
        if v: # and vaccine given
            v = vaccine_names[semantics['vaccine']]
            ret = 'Die Impfungen für {} mit {} sind {}'.format(s, v, results)
        else: # all vaccines for state
            ret = 'Die Impfungen für {} sind {}'.format(s, results)
    else: # no state
        if v: # but vaccine
            v = vaccine_names[semantics['vaccine']]
            ret = 'Die Impfungen in Deutschland mit {} sind {}'.format(v, results)
        else: # nothing given
            ret = eliza.respond(inputs)
    return ret

from eliza import eliza


root = r'/Users/shushanamanakhimova/S_Dialog/'
elz = eliza.Eliza()
elz.load(root+'/eliza/deutsch.txt')



def main():
    init_google()
    dialogmanager(elz)

if __name__ == "__main__":
    main()

