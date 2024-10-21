import os
from util.vectorspace import VectorSpace, SearchType


COLL = "smaller_dataset/collections"
QURY = "smaller_dataset/queries"


def parse_rel(path: str) -> dict[str, list[str]]:
    result = {}

    with open(path, "r") as file:
        for line in file:
            key, values = line.strip().split("\t")  # Split the line into key and values
            result[key] = [
                int(num) for num in values.strip("[]").split(",")
            ]  # Parse the values

    return result


def run():
    # e.g. documents["1"] = "document text"
    documents: dict[str, str] = {}
    for file in os.listdir(COLL):
        if file.endswith(".txt"):
            id = file.split(".")[0]
            id = id[1:]
            with open(COLL + "/" + file, "r", encoding="utf-8") as f:
                documents[id] = f.read()

    # vs = VectorSpace()
    vs = VectorSpace(documents)

    # e.g. rel["q1"] = ["1", "2", ...]
    rel: dict[str, list[str]] = parse_rel("smaller_dataset/rel.tsv")

    # e.g. all_queries["q1"] = "query text"
    all_queries: dict[str, str] = {}
    for file in os.listdir(QURY):
        if file.split(".")[0] in rel.keys():
            with open(QURY + "/" + file, "r", encoding="utf-8") as f:
                all_queries[file.split(".")[0]] = f.read()

    # e.g. all_results["q1"] = ["1", "2", ...]
    all_results: dict[str, list[str]] = {}
    for qname, query in all_queries.items():
        all_results[qname] = vs.search(query, 10)[0]

    print()

    # Calculate MRR
    mrr = 0
    for id, result in all_results.items():
        for i, doc in enumerate(result):
            if int(doc) in rel[id]:
                mrr += 1 / (i + 1)
                break

    print(f"MRR: {mrr/len(all_queries):.5f}")

    # Calculate MAP
    map_score = 0
    for id, result in all_results.items():
        avp = 0
        correct = 0
        for i, doc in enumerate(result):
            if int(doc) in rel[id]:
                correct += 1
                avp += correct / (i + 1)  # Precision at i + 1
        if correct == 0:
            continue
        avp /= correct  # Average precision for this query
        map_score += avp

    print(f"MAP: {map_score/len(all_queries):.5f}")

    # Calculate Recall
    recall = 0
    for id, result in all_results.items():
        correct = 0
        for i, doc in enumerate(result):
            if int(doc) in rel[id]:
                correct += 1
        recall += correct / 10  # Assuming a fixed recall denominator of 10

    print(f"Recall: {recall/len(all_queries):.5f}")


def main():
    run()


if __name__ == "__main__":
    main()
