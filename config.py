from pretreatment.context import ContextManager
from pretreatment.document import ItertorDocument
from tool import spliter, word_vector, label, resource,file
from pathlib import Path
manager = ContextManager()

with manager.new("train") as context:
    context.set_batch_size(25)
    context.set_sentences_files(*list(Path("F:\Lee\论文\我的项目\CNN_word2wec_sentence\webshell_train").iterdir())) # 用于训练词向量的文件
    context.set_resource_type(ItertorDocument)
    context.set_word2dev_model_file("word2vec.m")
    context.set_resource(resource("F:\Lee\论文\我的项目\CNN_word2wec_sentence\webshell_train"))
    context.set_splitter(spliter)  # 将文章分割为字符
    context.set_line_parser(word_vector, label,file)  # 将样本分割为词向量


with manager.extend_as("train", "test") as context:
    context.set_resource(resource("F:\Lee\论文\我的项目\CNN_word2wec_sentence\webshell_test"))
with manager.extend_as("train", "mytest") as context:
    context.set_resource(resource("F:\Lee\论文\我的项目\CNN_word2wec_sentence\mytest"))


if __name__ == "__main__":
    for data in manager("base").document.load():
        vector = data["vector"]
        label = data["label"]
        filename = data["filename"]
        # print(vector)
        # print(label)
        # exit()
        for i in range(10):
            for x in vector[1, i, :]:
                word = manager("base").embedding.get_word(int(x))
                word and print(word)
        exit()