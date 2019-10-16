from pathlib import Path
from pretreatment.parser import line_parser
import numpy as np
from pretreatment.tool import pad_array
import random
import re
split_re = re.compile(r"([^a-zA-Z0-9])")

def resource(dir_name):
    res = [ [open(f, 'rb').readlines(), 0 if "white_lee" in str(f) else 1,str(f)] for f in Path(dir_name).iterdir()]
    # for f in Path(dir_name).iterdir():
        # print(f)
        # exit()
    random.shuffle(res)
    return res

def spliter(code):
    """
    :param code: 一篇代码
    :return: 分割后的代码
    """
    # print("+++++++++")
    # print(code)
    code = re.sub(r"^b'\s.*/\*.*$", "", str(code))
    code = re.sub(r"^b'\s.*\*.*$", "", str(code))
    code = re.sub(r"b'.*\*/$", "", str(code))
    code = re.sub(r"^b'\s.*\*/.*$", "", str(code))
    code = re.sub(r"//.*$", "", str(code))
    code = re.sub(r"<!--.*$", "", str(code))
    code = re.sub(r"<%/\*.*$", "", str(code))
    code = re.sub(r"<%--.*$", "", str(code))
    code = re.sub(r"^b'\s.*\*.*$", "", str(code))
    # print("==========")
    # print(code)
    # exit()
    code = str(code).strip("b'").strip(r"\\r\\n").strip(r"\r\n")
    return [w for w in split_re.split(code) if w and w != " "]


@line_parser(name="vector", max_length=20)
def word_vector(resource, embedding):
    max_sentence = 400
    matrix = []
    for line in resource[0]:
        word_ids = [embedding.get_index(c) for c in spliter(line)[0:20]]
        pad_vector = pad_array(np.array(word_ids), 20, embedding.padding_word)
        matrix.append(pad_vector)
        if len(matrix) == max_sentence:
            break
    for _ in range(400 - len(matrix)):
        matrix.append(np.ones(20) * embedding.padding_word)
    #print(np.array(matrix).shape)
    #print(matrix)
    return np.array(matrix)

@line_parser(name="label", max_length=0)
def label(resource, embedding):
    return np.array([resource[1]])
@line_parser(name="filename", max_length=100)
def file(resource,embedding):
    return np.array([resource[2]])
