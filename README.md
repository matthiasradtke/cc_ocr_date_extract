# OCR, Date Extraction and Classification

## Overview
This project is an example of a production-ready pipeline that performed OCR on a collection of PDF documents
(e.g. invoices) and extracts a searched date (e.g. invoice date). 
A model is trained to classify all parsed dates in order to identify the searched date.
The classification is primarily based on text-based features associated with the search date. 

## How to Install and Run
Install [python-poetry](https://python-poetry.org/docs/), a tool for dependency management and packaging.

Create a python environment with
```
poetry install
```

Run the project with
```
kedro run
```


## Technologies

- [Poetry](https://python-poetry.org)
- [Kedro](https://kedro.readthedocs.io)
- [DVC](https://dvc.org)
- [tesseract](https://github.com/tesseract-ocr/tesseract)
- [scikit-learn](https://scikit-learn.org)
- [spaCy](https://spacy.io)
