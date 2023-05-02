from distutils import text_file
import pandas as pd
import string
import emoji
import re

CONTRACTIONS = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "cannot": "can not",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "that'll": "that will",
}


def regex_or(*items):
    """
    Format regular expression (or statement)

    Args:
        items (regex): Regular Expression

    Returns:
        capturing regex
    """
    return "(?:" + "|".join(items) + ")"


## Text Normalization

## Squeeze Whitespace
Whitespace = re.compile(
    "[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE
)


def squeezeWhitespace(input):
    """
    "foo   bar " => "foo bar"
    """
    return Whitespace.sub(" ", input).strip()


# 全角转半角
def full_to_half(text: str):  # 输入为一个句子
    _text = ""
    for char in text:
        inside_code = ord(char)  # 以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        if char == "’":
            _text += "'"
        else:
            _text += chr(inside_code)
    return _text


import html


def normalize_text(text):
    """
    Twitter text comes HTML-escaped, so unescape it.
    We also first unescape &amp;'s, in case the text has been buggily double-escaped.
    """
    text = squeezeWhitespace(text)
    text = full_to_half(text)
    text = text.replace("&amp;", "&")
    text = html.unescape(text)

    return text


# expand
## Flattening List of Lists
def flatten(l):
    """
    Flatten a list of lists by one level.

    Args:
        l (list of lists): List of lists

    Returns:
        flattened_list (list): Flattened list
    """
    flattened_list = [item for sublist in l for item in sublist]
    return flattened_list


def expand_contractions(tokens):
    """
    Expand English contractions.

    Args:
        tokens (list of str): Token list

    Returns:
        tokens (list of str): Tokens, now with expanded contractions.
    """
    if not isinstance(tokens, list):
        tokens = tokens.split()
    tokens = flatten(
        list(
            map(
                lambda t: CONTRACTIONS[t.lower()].split()
                if t.lower() in CONTRACTIONS
                else [t],
                tokens,
            )
        )
    )
    return " ".join(tokens)


def strip_user_mentions(tokens):
    """
    Remove tokens mentioning a username (Reddit or Twitter).

    Args:
        tokens (list of str): Token list

    Returns:
        tokens (list of str): Tokens, now without usernames.
    """
    if not isinstance(tokens, list):
        tokens = tokens.split()
    tokens = list(
        filter(lambda t: not (t.startswith("u/") or t.startswith("@")), tokens)
    )
    return " ".join(tokens)


def strip_url_simple(text):
    # 去除url
    text = re.sub(r"^(https:\S+)", " ", text)
    text = re.sub(r"[a-zA-Z]+://[^\s]*", " ", text)
    return text


def strip_punctuation(text):
    """
    Remove standalone punctuation from the token list

    Args:
        text

    Returns:
        text: text, now without standalone punctuation.
    """
    str_punctuation = r"""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~"""
    # = string.punctuation - '-'
    # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    text = text.translate(str.maketrans(str_punctuation, " " * len(str_punctuation)))
    return squeezeWhitespace(text)

def strip_emojis(text):
    return emoji.replace_emoji(text, "")


def process_for_filtering(str_original, flag_print=False):
    str_original = str(str_original)
    str_original = str_original.lower()
    str_processed = normalize_text(str_original)
    if flag_print:
        print(str_processed)
    str_processed = expand_contractions(str_processed)
    if flag_print:
        print(str_processed)
    str_processed = strip_user_mentions(str_processed)
    if flag_print:
        print(str_processed)
    str_processed = strip_url_simple(str_processed)
    if flag_print:
        print(str_processed)
    str_processed = strip_emojis(str_processed)
    if flag_print:
        print(str_processed)
    str_processed = strip_punctuation(str_processed)
    if flag_print:
        print(str_processed)
    return str_processed

def process_for_modeling(str_original, flag_print=False):
    str_original = str(str_original)
    text_preprocessed = normalize_text(str_original)
    if flag_print:
        print(text_preprocessed)
    text_preprocessed = expand_contractions(text_preprocessed)
    if flag_print:
        print(text_preprocessed)
    text_preprocessed = strip_user_mentions(text_preprocessed)
    if flag_print:
        print(text_preprocessed)
    text_preprocessed = strip_url_simple(text_preprocessed)
    if flag_print:
        print(text_preprocessed)
    text_preprocessed = emoji.demojize(text_preprocessed)
    if flag_print:
        print(text_preprocessed)
    # text_preprocessed = strip_punctuation(text_preprocessed)
    # if flag_print:
    #     print(text_preprocessed)
    return text_preprocessed
