import os
from util.vectorspace import VectorSpace, SearchType


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

    vs = VectorSpace(documents)

    print("TF Cosnine")
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

    print("\nTF Euclidean")
    result_tf_euc_ids, result_tf_euc_scores = vs.search(
        query, 10, SearchType.TF_EUCLIDEAN_DIST
    )
    print("NewsID \t Score")
    for i in zip(result_tf_euc_ids, result_tf_euc_scores):
        print(f"News{i[0]} \t {i[1]}")

    print("\nTF-IDF Euclidean")
    result_tfidf_euc_ids, result_tfidf_euc_scores = vs.search(
        query, 10, SearchType.TFIDF_EUCLIDEAN_DIST
    )
    print("NewsID \t Score")
    for i in zip(result_tfidf_euc_ids, result_tfidf_euc_scores):
        print(f"News{i[0]} \t {i[1]}")


def main():
    query = input("Enter query: ")
    run(query)


if __name__ == "__main__":
    main()
