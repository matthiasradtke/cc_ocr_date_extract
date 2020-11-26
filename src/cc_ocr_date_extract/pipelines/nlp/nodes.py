from typing import Dict, Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.matcher import Matcher

from spacy_langdetect import LanguageDetector

import pandas as pd
import numpy as np

import json
from datetime import datetime


def create_spacy_nlp_object(parameters: Dict[str, Any]) -> Language:
    nlp = spacy.load(parameters['spacy_lang'])
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    # Modify tokenizer
    suffixes = list(nlp.Defaults.suffixes)
    # remove dot as suffix
    suffixes.append(r"\.")
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search

    # modify tokenizer infix patterns
    infixes = (LIST_ELLIPSES + LIST_ICONS + [
        # EDIT: Removed hyphen \- : r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[0-9])[+\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ])
    infix_re = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer

    return nlp


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    # remove consecutive whitespaces
    import re
    df['pdf_text'] = df['pdf_text'].apply(lambda s: re.sub(r'\s+\n', ' ', s))
    return df


def get_date_matches(nlp: Language, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    # TODO: regex to configfile
    # pattern1 = [  # "%m/%d/%Y", "%m/%d/%y" also without zero-padding
    #     {"TEXT": {"REGEX": r"^(?:(1[0-2]|0?[1-9])[.\-\/]{1}(3[01]|[12][0-9]|0?[1-9]))[.\-\/]{1}(?:[0-9]{2})?[0-9]{2}$"}}
    # ]
    # pattern2 = [  # "%d/%m/%Y", "%d/%m/%y" also without zero-padding
    #     {"TEXT": {"REGEX": r"^(?:(3[01]|[12][0-9]|0?[1-9])[.\-\/]{1}(1[0-2]|0?[1-9]))[.\-\/]{1}(?:[0-9]{2})?[0-9]{2}$"}}
    # ]

    pattern1_2 = [  # "%m/%d/%Y", "%m/%d/%y" or "%d/%m/%Y", "%d/%m/%y"
        {"TEXT": {
            "REGEX": r"^(?:(1[0-2]|0?[1-9])[.\-\/]{1}(3[01]|[12][0-9]|0?[1-9]))[.\-\/]{1}(?:[0-9]{2})?[0-9]{2}$|^(?:(3[01]|[12][0-9]|0?[1-9])[.\-\/]{1}(1[0-2]|0?[1-9]))[.\-\/]{1}(?:[0-9]{2})?[0-9]{2}$"}}
    ]

    pattern3 = [  # "%Y/%m/%d"
        {"TEXT": {"REGEX": r"^(?:[1-9]{1}[0-9]{3})[.\-\/]{1}(?:(1[0-2]|0?[1-9])[.\-\/]{1}(3[01]|[12][0-9]|0?[1-9]))$"}}]

    months = r"(Jan(uar(y)?)?|Feb(ruar(y)?)?|Mar(ch)?|Mär(z)?|Apr(il)?|Ma(y|i)|Jun(e|i)?|Jul(y|i)?|Aug(ust)?|Sep(tember)?|O(c|k)t(ober)?|Nov(ember)?|De(c|z)(ember)?)"
    pattern4 = [  # "%d-%B-%Y", "d-%b-%Y" (20-Jun-2020, 20-June-2020)
        {"TEXT": {
            "REGEX": fr"^(?:(3[01]|[12][0-9]|0?[1-9])[.\-\/]{{1}}({months}))[.\-\/]{{1}}(?:[0-9]{{2}})?[0-9]{{2}}$"}}]

    matcher = Matcher(nlp.vocab)

    # matcher.add("Date: (mm/dd/yyyy m/d/yy)", None, pattern1)
    # matcher.add("Date: (dd/mm/yyyy d/m/yy)", None, pattern2)
    matcher.add("Date: (__/__/yyyy _/_/yy)", None, pattern1_2)
    matcher.add("Date: (yyyy/mm/dd)", None, pattern3)
    matcher.add("Date: (dd-Mon-yyyy)", None, pattern4)

    def parse_date(string: str, lang: str) -> datetime:
        from dateparser import parse
        # date_formats = ["%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%d-%B-%Y", "d-%b-%Y"]
        date = parse(string, languages=[lang])
        if not date:
            date = parse(string)
        return date

    def get_date_matches_from_text(doc: Doc) -> str:
        result = []
        all_dates = []
        for i, (match_id, start, end) in enumerate(matcher(doc)):
            match_id_str = nlp.vocab.strings[match_id]
            match_string = doc[start:end].text
            match_date = parse_date(match_string, doc._.language['language'])
            text_left = doc[max(0, start - parameters['n_lefts']):max(0, end - 1)].text
            text_right = doc[end:min(len(doc), end + parameters['n_rights'])].text
            result.append({
                'date_position': i,
                'match_id': match_id_str,
                'match_string': match_string,
                'match_date': match_date.strftime('%Y-%m-%d'),
                'text_left': text_left,
                'text_right': text_right,
            })
            all_dates.append(match_date)

        # get order of dates
        for i, idx in enumerate(np.argsort(all_dates)):
            result[idx]['date_order'] = i
            # also add total number of found dates
            result[idx]['n_match_dates'] = len(all_dates)

        return json.dumps(result)

    # find matching date strings
    matches = []
    language = []
    for d in nlp.pipe(df['pdf_text'], disable=['tagger', 'ner']):
        matches.append(get_date_matches_from_text(d))
        language.append(d._.language)
    df['matches'] = pd.Series(matches)
    df['language'] = pd.Series(language)

    return df
