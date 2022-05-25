import config
import torch

class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER
    
    def __len__(self):
        return(len(self.tweet))

    def __getitem(self, item):
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())
        
        len_selected_text = len(selected_text)
        idx0 = -1
        idx1 = -1
        for ind in (i for i, e in enumerate(tweet) if e==selected_text):
            if tweet[ind: ind+len_selected_text] == selected_text:
                idx0 = ind
                idx1 = ind + len_selected_text-1
                break
        