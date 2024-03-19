import pandas as pd
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# class IEMOCAPDataset(Dataset):

#     def __init__(self, path, roberta_path, split):
#         self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
#         self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
#         self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

#         self.speakers, self.labels, \
#         self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
#         self.sentences, self.trainIds, self.testIds, self.validIds \
#         = pickle.load(open(roberta_path, 'rb'), encoding='latin1')
#         if split == 'train':
#             self.keys = [x for x in self.trainIds]
#         elif split == 'test':
#             self.keys = [x for x in self.testIds]
#         elif split == 'valid':
#             self.keys = [x for x in self.validIds]
        
#         self.len = len(self.keys)

#     def __getitem__(self, index):
#         vid = self.keys[index]
#         sentence=self.sentences[vid]
#         a=torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]])
#         label=torch.LongTensor(self.videoLabels[vid]),
#         return torch.FloatTensor(self.videoText[vid]),\
#                torch.FloatTensor(self.videoVisual[vid]),\
#                torch.FloatTensor(self.videoAudio[vid]),\
#                torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]]),\
#                torch.FloatTensor([1]*len(self.videoLabels[vid])),\
#                torch.LongTensor(self.videoLabels[vid]),\
#                vid
               

#     def __len__(self):
#         return self.len

#     def collate_fn(self, data):
#         dat = pd.DataFrame(data)

#         return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class IEMOCAPDataset(Dataset):

    def __init__(self, path, roberta_path, split):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open(roberta_path, 'rb'), encoding='latin1')
        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        sentence=self.sentences[vid]
        label=torch.LongTensor(self.videoLabels[vid])
        speaker=torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]])
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid
               

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

# class MELDDataset(Dataset):

#     def __init__(self, path=None, train=True):
#         self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
#         self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
#         self.testVid,self.aaa = pickle.load(open(path, 'rb'),encoding='latin1')
#         self.keys = [x for x in (self.trainVid if train else self.testVid)]

#         self.len = len(self.keys)

#     def __getitem__(self, index):
#         vid = self.keys[index]
#         return torch.FloatTensor(self.videoText[vid]),\
#                torch.FloatTensor(self.videoVisual[vid]),\
#                torch.FloatTensor(self.videoAudio[vid]),\
#                torch.FloatTensor(self.videoSpeakers[vid]),\
#                torch.FloatTensor([1]*len(self.videoLabels[vid])),\
#                torch.LongTensor(self.videoLabels[vid]),\
#                vid

#     def __len__(self):
#         return self.len

#     def return_labels(self):
#         return_label = []
#         for key in self.keys:
#             return_label+=self.videoLabels[key]
#         return return_label

#     def collate_fn(self, data):
#         dat = pd.DataFrame(data)
#         return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
class MELDDataset(Dataset):

    def __init__(self, path, roberta_path, split):
        
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid,self.aaa = pickle.load(open(path, 'rb'),encoding='latin1')
        

        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open(roberta_path, 'rb'), encoding='latin1')

        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        sentens=self.sentences[vid]
      
        a=torch.FloatTensor(self.videoSpeakers[vid])
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               self.videoIDs[vid]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i].tolist()) if i<7 else pad_sequence(dat[i].tolist(), True) if i<9 else dat[i].tolist() for i in dat]



class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):
        
        self.Speakers, self.Features, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return  torch.FloatTensor(self.Features[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.EmotionLabels[conv])), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                conv

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]
