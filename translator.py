#!/usr/bin/python3
# pylint: disable=missing-function-docstring  # чтобы не быть Капитаном Очевидностью
# pylint: disable=missing-class-docstring  # чтобы не быть Капитаном Очевидностью
# pylint: disable=invalid-name                # сохраним традиционные наименования сигналов
# pylint: disable=consider-using-f-string     # избыточный синтаксис
# pylint: disable=redefined-builtin
# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=missing-module-docstring

import re
import sys

from isa import write_bin_code, write_json_code, ops_args_count


def pre_process(raw: str) -> str:
    lines = []
    for line in raw.split("\n"):
        # remove comments
        comment_idx = line.find("#")
        if comment_idx != -1:
            line = line[:comment_idx]
        # remove leading spaces
        line = line.strip()

        lines.append(line)

    text = " ".join(lines)
    # text = raw.replace("\n", " ")
    # избавляется от лишних пробелов и символов перехода строки
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
            if token_upper in ops_args_count:
                code.append({"opcode": token_upper, "args": []})
                args_count = ops_args_count[token_upper]
            elif args_count > 0:
                if args_count != 0 and token[0] == 'x' and token[1:].isdigit():
                    token = token[1:]
                code[-1]["args"].append(token)
                args_count -= 1
    return code, labels


def translate(text):
    processed_text = pre_process(text)
    data_tokens, text_tokens = tokenize(processed_text)
    labels = {}
    data, data_labels = allocate(data_tokens)
    code, code_labels = parse(text_tokens)
    labels = data_labels.copy()
    code_start_label = len(data)
    for key, value in code_labels.items():
        labels[key] = value + code_start_label
    program = data + code
    for word_idx, word in enumerate(program):
        if isinstance(word, dict):
            for arg_idx, arg in enumerate(word["args"]):
                if arg in labels:
                    program[word_idx]["args"][arg_idx] = labels[arg]

    return program


def main(args):
    assert len(args) == 3, \
        "Wrong arguments: translator.py <input_file> <target_json_file> <target_bin_file>"

    source, target_json, target_bin = args

    with open(source, "rt", encoding="utf-8") as f:
        source = f.read()

    program = translate(source)

    print("source LoC:", len(source.split()), "code instr:",
          len(program))

    write_json_code(target_json, program)
    write_bin_code(target_bin, program)


if __name__ == '__main__':
    main(sys.argv[1:])
