import logging
import pandas as pd
import os
from argparse import ArgumentParser
import zipfile
import datetime
from tqdm import tqdm
import numpy as np
import sys
import traceback
from pytorch_pretrained_bert import BertModel, BertTokenizer
from tools import get_feature
from torch.utils.data import DataLoader, TensorDataset
import torch
from collections import namedtuple

pd.options.mode.chained_assignment = None
__max_position_embeddings = 512

##LOGGING PROPERTY
LOG_FILE = 'logfile'
CONSOLE_LEVEL = logging.INFO
LOGFILE_LEVEL = logging.DEBUG
def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode", metavar="MODE", default="preprocess",
                        help="Choose Mode : preprocess,     Default:preprocess")
    parser.add_argument("--bert_option", dest="bert_option", metavar="bert_option", default='bert-base-uncased',
                        help="Please choose usage : corpus, bert     Default:corpus")
    parser.add_argument("--data", dest="data", metavar="data",
                        default="D:\\DT_project\\python_project\\news")
    parser.add_argument("--target", dest="target", metavar="target",
                        default="target")
    parser.add_argument("--max_length", dest="max_length", metavar="max_length",
                        default=512,
                        help="Choose data extention that you want to analyze : zip, txt, csv  Default:zip")
    parser.add_argument("--device", dest="device", metavar="device",
                        default='cuda',
                        help="Please put the minimum value that filter document(int)  Default:None")
    return parser


class myCluster(object):
    def __init__(self, config):

        self.file_path = config.data
        self.target_path = config.target

        ##check file and folder
        assert os.path.isdir(config.data), "There is no folder holding news data [" + str(config.data) + "] please modify your options"
        if not os.path.isdir(config.target):
            os.mkdir(config.target)

        ##read news list(Should be zip file)
        self.news_file_list = os.listdir(config.data)

        ## init BERT element
        self.bert_option = config.bert_option
        #self.bert = BertModel.from_pretrained(config.bert_option)
        #self.tokenizer = BertTokenizer.from_pretrained(config.bert_option, do_lower_case=True, do_basic_tokenize=True)

        self.bert = BertModel.from_pretrained('bert-base-uncased').to(config.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)

        self.max_length = config.max_length
        self.device = config.device

        ## init memory
        self.info_memory = []
        self.vector_memory = []

        ##set log file
        if not logging.getLogger() == None:
            for handler in logging.getLogger().handlers[:]:  # make a copy of the list
                logging.getLogger().removeHandler(handler)
        logging.basicConfig(filename=LOG_FILE, level=LOGFILE_LEVEL,
                            filemode='w')  # logging의 config 변경
        console = logging.StreamHandler()  # logging을 콘솔화면에 출력
        console.setLevel(CONSOLE_LEVEL)  # log level 설정
        logging.getLogger().addHandler(console)  # logger 인스턴스에 콘솔창의 결과를 핸들러에 추가한다.

        self.subset = namedtuple('info', ('file_index', 'doc_index', 'sent_index'))



    def run(self):

        """
        iterator = load_data(self.file_path, file_name)
        for source, index, doc in iterator:
            self.info_memory.append([source, index])
        """
        progressbar = tqdm(self.news_file_list, desc='preprocess file: ', unit="txt")
        for file_name in progressbar:
            df = pd.read_csv(os.path.join(self.file_path, file_name), sep='|', encoding='utf-8')
            indexs = df['Index'].values.tolist()
            docs = df['Story Body'].values.tolist()
            for index, doc in zip(indexs, docs):
                sent_length = len(doc.split("     "))
                print(sent_length)

                features = get_feature(doc.split("     "), self.tokenizer, self.max_length)
                doc_vectors = self.convert_ids_to_vector(self.bert, features)
                sent_indexs = [ i for i in range(sent_length)]
                doc_indexs = [index]*sent_length
                file_indexs = [file_name] * sent_length
                temp = [self.subset(file_index, doc_index, sent_index)for file_index, doc_index, sent_index in zip(file_indexs, doc_indexs, sent_indexs)]
                self.info_memory.extend(temp)
                self.vector_memory.extend(doc_vectors)
        print("end")
        print(len(self.info_memory))
        print(len(self.vector_memory))

    def convert_ids_to_vector(self, bert, data):
        bert.eval()
        with torch.no_grad():
            data = tuple(t.to(self.device) for t in data)
            input_ids, input_mask, segment_ids = data
            _, pooled_output = bert(input_ids=input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask,
                                    output_all_encoded_layers=False)
            data = tuple(t.to('cpu') for t in data)
            pooled_output = pooled_output.cpu()
        return pooled_output.tolist()

    def load_data(self, file_name):
        return pd.read_csv(os.path.join(self.file_path, file_name), sep='|', encoding='utf-8')

def doc_to_vector(tokenizer, model):





    return

"""
def load_data(file_path, file_name, tokenizer, model):
    df = pd.read_csv(os.path.join(file_path, file_name), sep='|', encoding='utf-8')
    data_list = df[['Index', 'Story Body']].values.tolist()
    for item in data_list:
        index = item[0]
        doc = item[1]
        texts = doc.split("     ")
        for text in texts:



            yield file_name, index, vectors
"""

def main():
    parser = build_parser()
    config = parser.parse_args()
    config.device = torch.device("cuda" if torch.cuda.is_available() and config.device in ['gpu', 'cuda'] else "cpu")
    agent = myCluster(config)
    agent.run()


if __name__ == "__main__":
    main()