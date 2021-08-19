from genericpath import isfile
import opensmile
import os
import audformat
from sklearn import svm
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import pickle

class EmoRec():
    root = './emodb/'
    clf = None
    filename = 'emorec.wav'
    sr = 16000
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        if not os.path.isdir(self.root):
            self.download_emodb()
        db = audformat.Database.load(self.root)
        db.map_files(lambda x: os.path.join(self.root, x))
        self.df_emo = db.tables['emotion'].df
        self.df_files = db.tables['files'].df
        if not self.clf:
            if not  os.path.isfile("svc_emodb_gemaps.pkl"):
                self.train_model()
            else:
                self.clf = pickle.load( open( "svc_emodb_gemaps.pkl", "rb" ) )
    def download_emodb(self):
        os.system('wget -c https://tubcloud.tu-berlin.de/s/LzPWz83Fjneb6SP/download')
        os.system('mv download emodb_audformat.zip')
        os.system('unzip emodb_audformat.zip')
        os.system('rm emodb_audformat.zip')

    def train_model(self):
        print('extracting features...')
        df_feats = self.smile.process_files(self.df_emo.index)
        train_labels = self.df_emo.emotion
        train_feats =  df_feats
        self.clf = svm.SVC(kernel='linear', C=.001)
        print('training a model...')
        self.clf.fit(train_feats, train_labels)
        pickle.dump(self.clf, open( "svc_emodb_gemaps.pkl", "wb" ) )
        print('done')
        

    def classify(self, wavefile):
        test_feats = self.smile.process_file(wavefile)
        return self.clf.predict(test_feats)

    def classify_from_micro(self, seconds):
        self.record(seconds)
        return self.classify(self.filename)[0]

    def record(self, seconds):
        data = sd.rec(int(seconds * self.sr), samplerate=self.sr, channels=1)
        sd.wait()  
        write(self.filename, self.sr, data)

def main():
    test = EmoRec()
    print(test.classify_from_micro(3))

if __name__ == "__main__":
    main()

