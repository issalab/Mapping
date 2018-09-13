import pickle
import pandas as pd


class ReadMeta:

    def __init__(self, neuralfeaturesdir):
        self.neuralfeaturesdir = neuralfeaturesdir

    def get_times(self):
        file = open(self.neuralfeaturesdir + 'Data_imagesAndtimes.pickle', 'rb')
        data_times = pickle.load(file)
        file.close()

        times = data_times[1]
        return times

    def get_DF_neu(self):
        DF_neu = pd.read_csv(self.neuralfeaturesdir + 'DataFrame_Neural.csv', sep=",", index_col=False)
        print(self.neuralfeaturesdir )
        return DF_neu

    def get_DF_img(self):
        DF_img = pd.read_csv(self.neuralfeaturesdir + 'DataFrame_Images.csv', sep=",", index_col=False)
        return DF_img


