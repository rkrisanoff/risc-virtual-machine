#!/usr/bin/python3
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

import re
import sys

from isa import write_bin_code, write_json_code


def preprocess(raw: str) -> str:
    lines = []
    for line in raw.split("\n"):
        # remove comments
        comment_idx = line.find(";")
        if comment_idx != -1:
            line = line[:comment_idx]
        # remove leading spaces
        line = line.strip()

        lines.append(line)

    text = " ".join(lines)
    text = re.sub("[ ][ ]*", " ", text)

    return text


def tokenize(text):
    text = re.sub(
        r"'.*'",
        lambda match: f'{",".join(map(lambda char:str(ord(char)),match.group()[1:-1]))}',
        text
    )
    data_section_index = text.find("section data:")
    text_section_index = text.find("section text:")

    data_tokens = re.split(
        "[, ]", text[data_section_index + len("section data:"): text_section_index])
    data_tokens = list(filter(lambda token: token, data_tokens))
    data_tokens = list(
        map(lambda token: (token[:-1],) if token[-1] == ':' else token, data_tokens))

    text_tokens = re.split(
        "[, ]", text[text_section_index + len("section text:"):])
    text_tokens = list(filter(lambda token: token, text_tokens))
    text_tokens = list(
        map(lambda token: (token[:-1],) if token[-1] == ':' else token, text_tokens))
    return data_tokens, text_tokens


def allocate(tokens):
    data = []
    labels = {}
    for token in tokens:
        if isinstance(token, tuple):
            labels[token[0]] = len(data)
        elif token.isdigit():
            data.append(token)
    labels["INPUT"] = len(data)
    labels["OUTPUT"] = len(data) + 1
    return data, labels


def parse(tokens):
    labels = {}
    code = []
    args_count = 0
    for token in tokens:
        if isinstance(token, tuple):
            labels[token[0]] = len(code)
        else:
            token_upper = token.upper()
            if args_count > 0:
                if args_count != 0 and token[0] == 'x' and token[1:].isdigit():
                    token = token[1:]
                code[-1]["args"].append(token)
                args_count -= 1
            else:
                code.append({"opcode": token_upper, "args": []})
                if token_upper == 'HALT':
                    args_count = 0
                elif token_upper == 'JMP':
                    args_count = 1
                elif token_upper in ["SW", "SWI", "LW", "LWI"]:
                    args_count = 2
                else:
                    args_count = 3
    return code, labels


def translate(text):
    processed_text = preprocess(text)
    data_tokens, text_tokens = tokenize(processed_text)
    labels = {}
    data, data_labels = allocate(data_tokens)
    code, code_labels = parse(text_tokens)
    labels = data_labels.copy()
    for key, value in code_labels.items():
        labels[key] = value
    for word_idx, word in enumerate(code):
        if isinstance(word, dict):
            for arg_idx, arg in enumerate(word["args"]):
                if arg in labels:
                    code[word_idx]["args"][arg_idx] = labels[arg]

    return data, code


def main(args):
    assert len(args) == 3, \
        "Wrong arguments: translator.py <input_file> <target_json_file> <target_bin_file>"

    source, target_json, target_bin = args

    with open(source, "rt", encoding="utf-8") as file:
        source = file.read()

    data, code = translate(source)

    write_json_code(target_json, data, code)
    byte_count = write_bin_code(target_bin, data, code)

    print(
        f"source LoC: {len(source.split())} code instr: {len(code)} code bytes: {byte_count}")


if __name__ == '__main__':
    main(sys.argv[1:])
