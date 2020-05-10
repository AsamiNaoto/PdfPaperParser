from pymongo import MongoClient
import datetime
import pprint
import re
from pathlib import Path
import pprint
import json

from pdf_parser_ver2 import PdfParser, DirectoryPdfParser

client = MongoClient('localhost', 27017)
db = client["db"]
collection = db["papers"]

start_patterns = {"序論":re.compile("[1-9]*( |　)*(背景|はじめに|Abstract|序論|概要|Introduction)"), 
                "結論":re.compile("[2-9]+( |　)*(おわりに|結論|まとめ|むすび|Conclusion)")}  # これが当てはまらないものも多い
end_patterns = {"序論":re.compile("[2-9]+( |　)*(関連研究|提案手法|従来手法|従来研究)"),
                "結論":re.compile("参考文献|謝辞")}  # これが当てはまらないものも多い
max_iter = 20
parse_page_numbers = None

with open('paper_directory.json', encoding="utf-8") as f:
    paper_directory_information = json.load(f)
    
dir_path_list = [Path(str_path) for str_path in paper_directory_information["directories"]]
conference_name_list = paper_directory_information["conference_names"]
title_position_number_list = paper_directory_information["title_position_numbers"]


for dir_path, conference_name,  title_position_number in zip(dir_path_list, conference_name_list, title_position_number_list):
    
    pdf_parser = PdfParser(conference_name=conference_name,
                        start_patterns=start_patterns,
                        end_patterns=end_patterns,
                        title_position_number=title_position_number,
                        parse_page_numbers=parse_page_numbers,
                        )



    dir_pdf_parser = DirectoryPdfParser(dir_path=dir_path,
                                        pdf_parser=pdf_parser,
                                        max_iter = max_iter
                                        )
    try:
        for dict_list in dir_pdf_parser.parse_dict_list():
            result = collection.insert_many(dict_list)
            post_ids = result.inserted_ids
            print(post_ids)
    except:  # キーが重複した場合
        print("key duplicate")
        continue

