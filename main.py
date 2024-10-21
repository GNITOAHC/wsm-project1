import problem1
import problem2
import problem3
import problem4
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Eng_query", type=str, default="Typhoon Taiwan war")
    parser.add_argument("--Chi_query", type=str, default="資安 遊戲")
    args = parser.parse_args()

    problem1.run(args.Eng_query)
    print("\n-------------------------")
    problem2.run(args.Eng_query)
    print("\n-------------------------")
    problem3.run(args.Chi_query)
    print("\n-------------------------")
    problem4.run()
