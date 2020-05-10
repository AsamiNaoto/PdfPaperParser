from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTContainer, LTTextBox
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

from pathlib import Path
import re
from collections import OrderedDict
import tqdm
import pprint
import json
import abc


def find_textboxes_recursively(layout):
    """
    再帰的にテキストボックス（LTTextBox）を探して、テキストボックスのリストを取得する。
    """
    # LTTextBoxを継承するオブジェクトの場合は1要素のリストを返す。
    if isinstance(layout, LTTextBox):
        text_boxes = [layout]
        return text_boxes  # 返すのはリスト

    # LTContainerを継承するオブジェクトは子要素を含むので、再帰的に探す。
    if isinstance(layout, LTContainer):
        text_boxes = []
        for child in layout:
            text_boxes.extend(find_textboxes_recursively(child))  # 再帰的にリストをextend
            
        return text_boxes

    return []  # 何も取得できなかった場合は空リストを返す


class SortTextbox2Column():
    """
    2段組み用のソート関数，始めのソートは左側と右側
    """
    def __init__(self, layout_x0, layout_x1):
        self.half_x = (layout_x0 + layout_x1)/2
    
    def __call__(self, text_box):
        if text_box.x0 < self.half_x:
            left_or_right = -1  # it mean left
            
        else:
            left_or_right = 1  # it mean right
            
        return (left_or_right, -text_box.y1)


class SortTextbox():
    """
    一段組のソート関数．textboxの左下の座標でソート
    """
    def __init__(self,*args):
        """
        2段組み用のソートクラスとの対応のため
        """
        pass
    def __call__(self, text_box):
        return (-text_boxt.y1, text_box.x0)


class PaperBase(metaclass=abc.ABCMeta):
    """
    論文のデータクラスとPaser用のストラテジーを一つにした抽象基底クラス.正直一つにする意味はない.
    ただ変更するクラスをまとめただけ
    
    """
    @abc.abstractmethod
    def toDict(self):
        pass
    
    @classmethod
    def parse_by_textboxes(cls, text_boxes, parse_info):
        """
        text_boxesからパースする
        """
        paper_title, parse_text_dict = cls.str_from_textboxes(text_boxes, parse_info)  # スタティクメソッド
        paper = cls.parse_by_text_dict(paper_title=paper_title, 
                                       parse_text_dict=parse_text_dict, 
                                       parse_info=parse_info)  # クラスメソッド
        
        return paper
    
    @classmethod
    def parse_by_text_dict(cls, paper_title, parse_text_dict, parse_info):
        """
        parse_by_textboxes(pdfのパース)内で利用する．サブクラスはこのメソッドをオーバーライドする．
        """
        raise NotImplementedError("Implement parse_by_text_dict")
        
        
    @classmethod
    def parse_by_dict(cls, content):
        """
        データベース読み込み用のメソッド．サブクラスはこのメソッドをオーバーライドする
        """
        raise NotImplementedError("Implement parse_by_content")
        
    @staticmethod
    def str_from_textboxes(text_boxes, parse_info):
        """
        共通するテキスト取得プログラム．オーバーライドはしない
        """
        parse_text_flag = False  # このフラッグがTrueである部分を保存する        
                
        patterns_keys = parse_info["start_patterns"].keys()  # キーのリスト(のようなもの)
        patterns_key_iter = iter(patterns_keys)  # 長さの違うfor文内で回すので，キーをイテレーター化
        pattern_key = next(patterns_key_iter)  # 最初のキーを取得
        
        parse_text_dict = {i:"" for i in patterns_keys}
        
        for i,box in enumerate(text_boxes):
            text = box.get_text().strip()  # 末尾の文字を削除
            if i == parse_info["title_position_number"]:
                paper_title = text

            if parse_text_flag:  # flagがTrueのうちは，parse_textにtextを加え続ける
                parse_text_dict[pattern_key] += text

            if parse_info["start_patterns"][pattern_key].search(text):  # マッチしたらフラッグをTrueに
                parse_text_flag = True
            
            if parse_info["end_patterns"][pattern_key] is not None:  # Noneだったら，最後までflagはTrue
                if parse_info["end_patterns"][pattern_key].search(text):
                    try:
                        parse_text_flag = False
                        pattern_key = next(patterns_key_iter)  # end_patternがマッチしたらpatterns_key_iterをイテレーション
                    except StopIteration:
                        break  # 次のpattern_keyがなくなってStopIterationエラーが出たら終了
        
        return paper_title, parse_text_dict


class PaperForSave(PaperBase):
    """
    論文をテキストデータとして，保存するための論文データクラス
    """
    def __init__(self, 
                 conf_name=None, 
                 pdf_name=None, 
                 paper_title=None, 
                 pdf_content=None,
                 
                ):
        """
        一つのデータで
        Parameters
        ----------
        conf_name: str
            学会や論文集を表す文字列
        pdf_name: str
            対応するpdfファイルの名前を表す文字列
        paper_title: str
            論文のタイトル
        pdf_content: dict
            保存するテキストのdictionaly
        """
        self.conf_name = conf_name
        self.pdf_name = pdf_name
        self.paper_title = paper_title
        self.pdf_content = pdf_content
        
    def toDict(self):
        out_dict = {"conf_name": self.conf_name,
                    "pdf_name": self.pdf_name,
                    "paper_title": self.paper_title,
                    "content":self.pdf_content
                   }
        return out_dict
    
    @classmethod
    def parse_by_text_dict(cls, paper_title, parse_text_dict, parse_info):
        #from IPython.core.debugger import Pdb; Pdb().set_trace()  # PaperForSave
        paper_conf_name = parse_info["conf_name"]
        paper_pdf_name = parse_info["pdf_name"]
        
        # Paperへのデータの付与
        paper = cls(conf_name=paper_conf_name,
                    paper_title=paper_title,
                    pdf_name=paper_pdf_name,
                    pdf_content=parse_text_dict
                   )

        return paper


class PaperForCount(PaperBase):
    """
    カウント用のPaperオブジェクト
    """
    def __init__(self, pdf_name=None, count_patterns=[], paper_title=None):
        """
        countersは保存する文字列あるいはパターンのリスト
        Parameters
        ----------
        count_patterns: list of patterns
            検索するパターンのリスト
        """
        self.pdf_name = pdf_name
        self.paper_title = paper_title
        self.counters = OrderedDict()
        for i in count_patterns:
            self.counters[i] = 0  # パターンオブジェクトはhashableでキーにできる．まず，0に初期化
    
    def toDict(self):
        counters = {i.pattern:self.counters[i] for i in self.counters.keys()}  # キーを文字列へ
        
        out_dict = {"pdf_name":self.pdf_name,
                    "paper_title":self.paper_title,
                    "counters":counters
                   }
        return out_dict
    
    @classmethod
    def parse_by_text_dict(cls, paper_title, parse_text_dict, parse_info):
        """
        Parameters
        ----------
        parse_info: dict 
            パースの時に必要な情報
        """
        #from IPython.core.debugger import Pdb; Pdb().set_trace()  # PaperForCount
        
        paper_pdf_name = parse_info["pdf_name"]
                
        # 以下Paperへのデータの付与
        count_patterns = parse_info["count_patterns"]
        
        paper = cls(pdf_name=paper_pdf_name,
                    paper_title=paper_title,
                    count_patterns=count_patterns
                   )
        
        for pattern in count_patterns:
            for text in parse_text_dict.values():
                m = pattern.findall(text)
                paper.counters[pattern] += len(m)
                
        return paper
    
    @classmethod
    def parse_by_dict(cls, paper_dict, parse_info):
        """
        データベースからパースするとき用
        """
        #from IPython.core.debugger import Pdb; Pdb().set_trace()  # PaperForCount
        paper_title = paper_dict["paper_title"]
        paper_pdf_name = paper_dict["pdf_name"]
        parse_text_dict = paper_dict["contents"]
        
        count_patterns = parse_infonfo["count_patterns"]
        
        paper = cls(pdf_name=paper_pdf_name,
                    paper_title=paper_title,
                    count_patterns=count_patterns
                   )
        
        for pattern in count_patterns:
            for text in parse_text_dict.values():
                m = pattern.findall(text)
                paper.counters[pattern] += len(m)
                
        return paper
    
    def is_counted(self):
        """
        検索結果が一つも含まれていないかを判定する
        """
        if set(self.counters.values()) == {0}:
            return False
        else:
            return True
        
    def __repr__(self):
        str_pdf_name = str(self.pdf_name)
        str_paper_title = str(self.paper_title)
        str_counters = str(self.counters)
        return str_pdf_name+"\n"+str_paper_title+"\n"+str_counters


class PdfParser():
    """
    PaperForSaveに対応し，全てのPdfParserはこのクラスを継承する
    """
    def __init__(self, 
                 conference_name,
                 start_patterns={"all":re.compile(".*")},
                 end_patterns={"all":None},
                 title_position_number=2,
                 parse_page_numbers=[0],
                 column_number=2,
                 paper_data_class=PaperForSave()
                ):
        """
        Parameters
        ----------
        conference_name: str
            学会や論文集の名前
        start_patterns: dict of patterns
            Paperオブジェクトに保持するテキストの開始位置の辞書
        end_patterns: dict of pattrens
            Paperオブジェクトに保持するテキストの終了位置の辞書，Noneは最後まで
        title_position_number: int
            titleが与えられるtextboxのインデックス(ソート後)
        parse_page_numbers: list of int
            パースするページのリスト，Noneは最後まで
        paper_data_class: Paper class
            ペーパークラスのオブジェクトをストラテジーとして直接与える．
        """
        
        self.conference_name = conference_name
        
        if set(start_patterns.keys()) != set(end_patterns.keys()):
            raise ValueError("start patterns and eend patterns are not correspondding")
        
        self.title_position_number = title_position_number
        self.parse_page_numbers = parse_page_numbers  
        self.column_number = column_number
        
        self.paper_data_class = paper_data_class
        
        self.start_patterns = start_patterns
        self.end_patterns = end_patterns
        
        # パースに必要なクラスの作成
        # Layout Analysisのパラメーターを設定。縦書きの検出を有効にする。
        laparams = LAParams(detect_vertical=True)

        # 共有のリソースを管理するリソースマネージャーを作成。
        resource_manager = PDFResourceManager(caching=False)

        # ページを集めるPageAggregatorオブジェクトを作成。
        self.device = PDFPageAggregator(resource_manager, laparams=laparams)

        # Interpreterオブジェクトを作成。
        self.interpreter = PDFPageInterpreter(resource_manager, self.device)
        
        if column_number==1:
            self.SortFuncClass = SortTextbox  # クラスを変数として保持
        elif column_number==2:
            self.SortFuncClass = SortTextbox2Column
        else:
            raise ValueError("The column rather than two is not defined")
        
    def parse(self, pdf_file_path):
        """
        オーバーライドは原則禁止
        """
        self.pdf_file_name = str(pdf_file_path.stem)  # 内部メソッドからの参照用
        
        with open(pdf_file_path, "rb") as f:

            parse_text = ""
            parse_text_flag = False  # このフラッグがTrueである部分を序論とする

            for page in PDFPage.get_pages(f, pagenos=self.parse_page_numbers):
                self.interpreter.process_page(page)  # ページを処理する。
                layout = self.device.get_result()  # LTPageオブジェクトを取得。
                text_boxes = find_textboxes_recursively(layout)      

                # text_boxの座標値毎にソート，複数キーのソート
                # 少なくともこのページは全て読み込む必要があるため，非効率
                sort_func= self.SortFuncClass(layout_x0=layout.x0, layout_x1=layout.x1)
                text_boxes.sort(key=sort_func)
                
                info_dict = self.parse_info()
                paper = self.paper_data_class.parse_by_textboxes(text_boxes, info_dict)

        return paper
    
    def parse_info(self):
        """
        Paperオブジェクトによって要オーバーライド
        """
        info_dict = {}
        info_dict["conf_name"] = self.conference_name
        info_dict["pdf_name"] = self.pdf_file_name
        info_dict["start_patterns"] = self.start_patterns
        info_dict["end_patterns"] = self.end_patterns
        info_dict["title_position_number"] = self.title_position_number
        return info_dict


class PdfParserCount(PdfParser):
    """
    カウント用のPdfParser
    """
    def __init__(self, count_patterns,**kwargs):
        """
        Parameters
        ----------
        count_patterns: list of pattern
            検索したいパターンのリスト
        conference_name: str
            学会や論文集の名前
        start_patterns: dict of patterns
            Paperオブジェクトに保持するテキストの開始位置の辞書
        end_patterns: dict of pattrens
            Paperオブジェクトに保持するテキストの終了位置の辞書，Noneは最後まで
        title_position_number: int
            titleが与えられるtextboxのインデックス(ソート後)
        parse_page_numbers: list of int
            パースするページのリスト，Noneは最後まで
        paper_data_class: Paper class
            ペーパークラスのオブジェクトをストラテジーとして直接与える．
        """
        kwargs["paper_data_class"] = PaperForCount()  # カウント用のPaperクラス
        super(PdfParserCount, self).__init__(**kwargs)  # 引数展開
        self.count_patterns = count_patterns
        
    def parse_info(self):
        info_dict = super(PdfParserCount, self).parse_info()
        info_dict["count_patterns"] = self.count_patterns
        
        return info_dict


class DirectoryPdfParserJson():
    """
    あるディレクトリを与えたpdf_parserによってパースする．
    """
    def __init__(self, 
                 dir_path,
                 pdf_parser
                ):
        
        self.dir_path = Path(dir_path)
        self.pdf_list = list(self.dir_path.glob("./*.pdf"))  # 複数回パースする必要があるため、リスト化
        
        self.pdf_parser = pdf_parser
        
    def parse(self):
        """
        Paperオブジェクトのリストを返す．
        """
        paper_list = []
        for i in tqdm.tqdm(self.pdf_list):
            paper = self.pdf_parser.parse(i)
            paper_list.append(paper)
        
        return paper_list
    
    def parse_dict(self):
        """
        ディクショナリに変換して返す．
        """
        save_dict = {self.pdf_parser.conference_name:{}}
        paper_list = self.parse()
            
        # 全てメモリで展開するので，非効率
        for paper in paper_list:
            save_dict[self.pdf_parser.conference_name][paper.pdf_name] = paper.toDict()
            
        return save_dict

    def parse_dict_list(self):
        """
        ディクショナリのリストに変換して返す．
        """
        out_list = [paper.toDict() for paper in self.parse()]
        return out_list

if __name__ == "__main__":
    
    ##########################
    ### 共通項目
    ##########################
    start_patterns = {"序論":re.compile("[1-9]*\s*\S*(背景|はじめに|Abstract|序論|概要|Introduction)")}  # これが当てはまらないものも多い
    end_patterns = {"序論":re.compile("(関連研究|提案手法|従来手法|従来研究)")}  # これが当てはまらないものも多い
    #end_patterns = {"序論":None}
    conference_name = "SSII2019"
    title_position_number = 2
    parse_page_numbers = [0]  # 正直これが一番重要(1枚目まで確認)

    ##########################
    ### PdfParserのテスト
    ##########################
    pdf_paper_parser = PdfParser(
                                conference_name=conference_name,
                                start_patterns=start_patterns,
                                end_patterns=end_patterns,
                                title_position_number=title_position_number,
                                parse_page_numbers=parse_page_numbers,
                                )

    paper = pdf_paper_parser.parse(Path("../sample_pdf/IS1-02.pdf"))
    print(paper.pdf_content)
    print("\n\n\n")

    ##########################
    ### PdfParserCountのテスト
    ##########################
    count_patterns = [re.compile("ディープラーニング|深層学習"),
                    re.compile("CNN|ニューラルネットワーク"),
                    re.compile("VAE|変分オートエンコーダ"),
                    re.compile("GAN")
                    ]
    pdf_paper_parser = PdfParserCount(count_patterns=count_patterns,
                                  conference_name=conference_name,
                                  start_patterns=start_patterns,
                                  end_patterns=end_patterns,
                                  title_position_number=title_position_number,
                                  parse_page_numbers=parse_page_numbers,
                                  )

    paper = pdf_paper_parser.parse(Path("../sample_pdf/IS1-02.pdf"))
    pprint.pprint(paper.toDict())
    print("\n\n\n")

    ##########################
    ### DirectoryPdfParserJsonのテスト
    ##########################

    dir_path = Path("../sample_pdf")
    pdf_parser = PdfParser(conference_name=conference_name,
                        start_patterns=start_patterns,
                        end_patterns=end_patterns,
                        title_position_number=title_position_number,
                        parse_page_numbers=parse_page_numbers,
                        )

    dir_pdf_parser = DirectoryPdfParserJson(dir_path=dir_path,
                                            pdf_parser=pdf_parser
                                        )
    paper_dict_list = dir_pdf_parser.parse_dict_list()
    pprint.pprint(paper_dict_list)


    