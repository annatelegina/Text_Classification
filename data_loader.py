from lxml import etree
import os

class TextDataLoader(object):
    def __init__(self, path, name='news_eval_', valid_symbols=['-', "+", "0"]):
        self.path = path
        self.name = name
        self.valid_symbols = valid_symbols
        
        self.train_dict = self.parse_xml(mode='train')
        self.test_dict = self.parse_xml(mode='test')
    
    def parse_xml(self, mode='train'):
        filename = os.path.join(self.path, mode,self.name+mode + '.xml')
        
        tree = etree.parse(filename)
        root = tree.getroot()

        residents = []

        for member in root.findall('sentence'):
            resident = {}
            speech = member.find('speech').text.strip()
            resident.update({"speech": speech})
            evaluation = member.find('evaluation').text.strip()
            if evaluation not in self.valid_symbols:
                continue
            resident.update({"evaluation": evaluation})
            residents.append(resident)

        sentences = residents
        return sentences
        
        
    def get_data(self, mode='train'):
        sentences = []
        targets = []
        our_list = self.train_dict if mode == 'train' else self.test_dict
        
        for i in our_list:
            sent = i["speech"]
            a = "".join(c for c in sent if c not in (';','(', ')','!',':', '-', ',', '?', '"', '«','»', '%',"@" )\
                                        and c not in "1234567890")
            sentences.append(a)
            if i["evaluation"] == "+":
                k = 1
            elif i["evaluation"] == "-":
                k = -1
            else:
                k = 0
            targets.append(k)
            
        return sentences, targets
