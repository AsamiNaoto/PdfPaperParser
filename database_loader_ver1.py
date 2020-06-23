from pymongo import MongoClient
import pymongo
import datetime
import pprint
import re
from pathlib import Path


from pdf_parser_ver2 import PdfParser, DirectoryPdfParser, PdfParserCount


class PaperDataBaseLoadPattern():
        def __init__(self, collection, find_patterns, content_name, find_max=100):
            self.collection = collection
            self.find_patterns = find_patterns
            self.content_name = content_name
            self.find_max = find_max
        
        def load(self):
            """
            検索結果からfind_max分だけ検索．呼んだ回数だけ検索結果が進む, 検索はパターンのorで行う．
            """
            result = self.collection.find(filter={"$or":[{"content."+self.content_name:i} for i in self.find_patterns]}).sort("date", pymongo.DESCENDING)  # dateはないけど
            out_list = []
            iter_counter = 0
            for paper in result:
                out_list.append(paper)
                iter_counter += 1
                if iter_counter >= self.find_max-1:
                    yield out_list
                    out_list = []  # リストの初期化
                    iter_counter = 0
            yield out_list #全て終わったときに返す
        
        def load_and(self):
            """
            検索結果からfind_max分だけ検索．呼んだ回数だけ検索結果が進む，検索はパターンのand行う
            """
            result = self.collection.find(filter={"$and":[{"content."+self.content_name:i} for i in self.find_patterns]}).sort("date", pymongo.DESCENDING)  # dateはないけど
            out_list = []
            iter_counter = 0
            for paper in result:
                out_list.append(paper)
                iter_counter += 1
                if iter_counter >= self.find_max-1:
                    yield out_list
                    out_list = []  # リストの初期化
                    iter_counter = 0
            yield out_list #全て終わったときに返す
            
if __name__ == "__main__":
    client = MongoClient(host='133.91.72.15', port=27017)
    db = client["db"]
    collection = db["papers"] 
    
    count_patterns = [re.compile("ディープラーニング|深層学習"),
                    re.compile("CNN|ニューラルネットワーク"),
                    re.compile("VAE|変分オートエンコーダ"),
                    re.compile("GAN"),
                    ]
    
    database_loader = PaperDataBaseLoadPattern(collection=collection,
                                            content_name="序論",
                                            find_patterns=count_patterns,
                                            find_max=50
                                            )
    database_loader_generator = database_loader.load()
    
    dict_list = next(database_loader_generator)
    pprint.pprint(dict_list)