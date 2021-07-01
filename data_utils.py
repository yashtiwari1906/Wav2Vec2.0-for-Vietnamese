
import re
import json
import os
import torch 
import string
#import datasets 
from tqdm import tqdm
import soundfile
import transformers
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import json


def remove_punctuation_and_lower(texts):
    punctuation = re.sub(r"\'", r"", string.punctuation)
    for i in range(len(texts)):
        texts[i] = texts[i].translate(str.maketrans("", "", punctuation)).upper()
    return texts

def create_vocabulary_file(texts):
    vocab_list = list(set(" ".join(texts)))    
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict.pop(" ")
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open("./vocab.json", "w") as f:
        json.dump(vocab_dict, f)

def process_timit_dataset(read_limit=2500):

    path = r"/content/drive/MyDrive/vivos/train/waves"

    filenames, text_labels = [], []

    for dirname, _, filenames_ in (os.walk(path)):
        for filename in filenames_:
            #filename = filenames[i]
            text_labels.append(filename.split(".")[0])
            #print(os.path.join(dirname, filename))
            filenames.append(os.path.join(dirname, filename))
    #print(filenames[0])
    print("filenames and labels are ready to use!!!")
    print(filenames)
    with open("/content/text_file.json", "r") as file_open:
        file = json.load(file_open)
    labels = []
    for lab in text_labels:
        labels.append(file[lab])
    train_files, train_text = filenames[:read_limit], labels[:read_limit]
    test_files, test_text = filenames[:read_limit], labels[:read_limit]
    train_text = remove_punctuation_and_lower(train_text)
    test_text = remove_punctuation_and_lower(test_text)
    create_vocabulary_file(train_text + test_text)
    print(train_files)
    return {"file": train_files, "text": train_text}, {"file": test_files, "text": test_text}

def processor():
    
    tokenizer = Wav2Vec2CTCTokenizer("/content/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor_ = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor_

class TimitDataloader:

    def __init__(self, data, batch_size):

        tokenizer = Wav2Vec2CTCTokenizer("/content/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.processor = processor#transformers.Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.files, self.text = data["file"], data["text"]
        self.batch_size = batch_size
        self.ptr = 0

    def __len__(self):
        return len(self.files) // self.batch_size 

    def flow(self):
        speech, text = [], []
        for _ in range(self.batch_size):
            signal, sr = soundfile.read(self.files[self.ptr], dtype="float32")
            speech.append(signal)
            text.append(self.text[self.ptr])
            self.ptr += 1
            if self.ptr >= len(self.files):
                self.ptr = 0

        inputs = self.processor(speech, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
        input_data, input_attention = inputs["input_values"], inputs["attention_mask"]
        with self.processor.as_target_processor():
            labels = self.processor(text, padding=True, return_tensors="pt")
            targets, attention_mask = labels["input_ids"], labels["attention_mask"]
            targets = targets.masked_fill(attention_mask.ne(1), -100)
        return input_data, input_attention, targets


def get_dataloaders(batch_size, read_limit=2500):
    train_data, test_data = process_timit_dataset(read_limit=read_limit)
    train_loader = TimitDataloader(train_data, batch_size)
    test_loader = TimitDataloader(test_data, batch_size)
    return train_loader, test_loader

    
        
if __name__ == "__main__":

    train_data, test_data = process_timit_dataset()
    train_loader = TimitDataloader(train_data, batch_size=4)

    inputs, input_attention, targets = train_loader.flow()
    print(inputs)
    print(input_attention)
    print(targets)