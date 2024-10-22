# 2023 Web Search & Mining: Project 1

## Dependencies

```
nltk = "^3.9.1"
jieba = "^0.42.1"
numpy = "^2.1.2"
textblob = "^0.18.0.post0"
```

**Setting up via `poetry`:**

```
pip install poetry
poetry install
```

**Setting up via `pip`**

```
pip install nltk jieba numpy textblob
```

## Execution

Default values are:

-   ENG_QUERY: "Typhoon Taiwan war"
-   CHI_QUERY: "資安 遊戲"

**To use default values**

```
python main.py
```

**Usage**

```
usage: main.py [-h] [--Eng_query ENG_QUERY] [--Chi_query CHI_QUERY]

options:
  -h, --help            show this help message and exit
  --Eng_query ENG_QUERY
  --Chi_query CHI_QUERY
```
