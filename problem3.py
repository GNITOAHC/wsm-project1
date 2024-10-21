import os
from util.vectorspace import SearchType, VectorSpace, DefaultParser

import jieba

# Setup path
CWD = os.getcwd()
os.environ["JIEBA_DATA"] = f"{CWD}/jieba_cache"

jieba.dt.tmp_dir = os.environ["JIEBA_DATA"]  # type: ignore
if not os.path.exists(jieba.dt.tmp_dir):  # type: ignore
    os.mkdir(jieba.dt.tmp_dir)  # type: ignore


class ChineseParser(DefaultParser):
    def __init__(self):
        self.stopwords = []

        # Initialize jieba
        jieba.initialize()

    def clean(self, s: str) -> str:
        """remove any nasty grammar tokens from string"""
        return s.replace("。", "").replace("，", "").replace(r"\s+", " ").lower()

    def remove_stopwords(self, words: list[str]) -> list[str]:
        """Remove common words which have no search value"""
        # return [word for word in words if word not in self.stopwords]
        return words

    def tokenise(self, string):
        """Break string up into tokens and stem words"""
        words = jieba.cut(self.clean(string))
        return list(words)

    def text_preprocess(self, text: str) -> list[str]:
        """Tokenise text and remove stopwords"""
        words = self.tokenise(text)
        return self.remove_stopwords(words)


def run(query: str):
    # e.g. documents["1"] = "document text"
    documents: dict[str, str] = {}
    for file in os.listdir("ChineseNews"):
        if file.endswith(".txt"):
            id = file.split(".")[0]
            id = id[4:]
            with open("ChineseNews/" + file, "r", encoding="utf-8") as f:
                documents[id] = f.read()

    # for d in documents:
    #     print(d, documents[d])
    # return

    vs = VectorSpace(documents, parser=ChineseParser())

    print("\nTF Cosnine")
    result_tf_cos_ids, result_tf_cos_scores = vs.search(query, 10, SearchType.TF_COS)
    print("NewsID \t Score")
    for i in zip(result_tf_cos_ids, result_tf_cos_scores):
        print(f"News{i[0]} \t {i[1]}")

    print("\nTF-IDF Cosnine")
    result_tfidf_cos_ids, result_tfidf_cos_scores = vs.search(
        query, 10, SearchType.TFIDF_COS
    )
    print("NewsID \t Score")
    for i in zip(result_tfidf_cos_ids, result_tfidf_cos_scores):
        print(f"News{i[0]} \t {i[1]}")


def main():
    query = input("Enter query: ")
    run(query)


if __name__ == "__main__":
    main()
