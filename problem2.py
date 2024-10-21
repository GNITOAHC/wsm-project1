import os
from util.vectorspace import VectorSpace, SearchType, SimStrategy, DefaultParser
import nltk

# Setup path
CWD = os.getcwd()
os.environ["NLTK_DATA"] = f"{CWD}/nltk_data"
nltk.data.path.append(os.environ["NLTK_DATA"])


class RelFeedbackParser(DefaultParser):
    def __init__(self):
        # Install the nltk packages
        nltk.download("punkt_tab", download_dir="./nltk_data")
        nltk.download("averaged_perceptron_tagger_eng", download_dir="./nltk_data")

        self.stopwords = open("util/english.stop", "r").read().split()

    def tokenise(self, string):
        """Break string up into tokens and stem words"""
        corpus: list[str] = []

        words = self.clean(string)
        tokens = nltk.word_tokenize(words)
        tagged = nltk.pos_tag(tokens)
        for word, tag in tagged:
            if tag.startswith("N") or tag.startswith("V"):
                corpus.append(word)

        return corpus


def run(query: str):
    # e.g. documents["1"] = "document text"
    documents: dict[str, str] = {}
    for file in os.listdir("EnglishNews"):
        if file.endswith(".txt"):
            id = file.split(".")[0]
            id = id[4:]
            with open("EnglishNews/" + file, "r", encoding="utf-8") as f:
                documents[id] = f.read()

    # for d in documents:
    #     print(d, documents[d])
    # return

    vs = VectorSpace(documents, parser=RelFeedbackParser())

    print("\nTF-IDF Cosnine")
    result_tfidf_cos_ids, result_tfidf_cos_scores = vs.search(
        query, 10, SearchType.TFIDF_COS
    )
    print("NewsID \t Score")
    for i in zip(result_tfidf_cos_ids, result_tfidf_cos_scores):
        print(f"News{i[0]} \t {i[1]}")

    print("\nRelevance Feedback (Re-ranked)")
    result_ids, result_scores = vs.search_rel_feedback(query, 10, SimStrategy.COSINE)
    print("NewsID \t Score")
    for i in zip(result_ids, result_scores):
        print(f"News{i[0]} \t {i[1]}")


def main():
    query = input("Enter query: ")
    run(query)


if __name__ == "__main__":
    main()
