#!/usr/bin/python3
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string
# pylint: disable=redefined-builtin
# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=missing-module-docstring

import re
import sys
import unittest

from isa import normalize, encode_instr, decode_instr


class TestBinarizator(unittest.TestCase):

    def test_normalize(self):
        code = [
            "0", "1",
            {"opcode": "ADDI", "args": ["3", "0", 696]},
            {"opcode": "LW", "args": ["1", "3"]},
            {"opcode": "SW", "args": ["2", "1"]},
            {"opcode": "BEQ", "args": ["1", "0", 38]},
            {"opcode": "HALT", "args": []}]
        benchmark = [
            '0', '1',
            {'opcode': 'ADDI', 'args': [3, 0, 696]},
            {'opcode': 'LW', 'args': [1, 3, 0]},
            {'opcode': 'SW', 'args': [0, 2, 1]},
            {'opcode': 'BEQ', 'args': [1, 0, 38]},
            {"opcode": "HALT", "args": [0]}
        ]
        normalized_code = normalize(code)
        print(normalized_code)

        for word_idx, word in enumerate(normalized_code):
            if isinstance(word, dict):
                self.assertEqual(
                    word['opcode'], normalized_code[word_idx]['opcode'])
                self.assertListEqual(word["args"], benchmark[word_idx]["args"])
            else:
                self.assertEqual(word, benchmark[word_idx])

    def test_register_instruction(self):
        init = [
            {"opcode": "ADD", "args": [1, 2, 3]},
            {"opcode": "SUB", "args": [1, 1, 2]},
            {"opcode": "REM", "args": [1, 2, 0]},
            {"opcode": "DIV", "args": [4, 1, 3]},
            {"opcode": "MUL", "args": [3, 2, 3]},
        ]
        for instr in init:
            encoded = encode_instr(instr)
            opcode, rd, rs1, rs2, imm = decode_instr(
            int.from_bytes(encoded, "little"))

            self.assertEqual(opcode.name, instr["opcode"])
            self.assertListEqual([rd,rs1,rs2], instr["args"])
            self.assertEqual(imm, 0)

    def test_immediate_instruction(self):
        init = [
            {"opcode": "ADDI", "args": [1, 2, 3]},
            {"opcode": "SUBI", "args": [1, 1, 2]},
            {"opcode": "REMI", "args": [1, 2, 0]},
            {"opcode": "DIVI", "args": [4, 1, 3]},
            {"opcode": "MULI", "args": [3, 2, 3]},
        ]
        for instr in init:
            encoded = encode_instr(instr)
            opcode, rd, rs1, rs2, imm = decode_instr(
            int.from_bytes(encoded, "little"))

            self.assertEqual(opcode.name, instr["opcode"])
            self.assertListEqual([rd,rs1,imm], instr["args"])
            self.assertEqual(rs2, 0)
    def test_branch_instruction(self):
        init = [
            {"opcode": "BEQ", "args": [1, 2, 3]},
            {"opcode": "BNE", "args": [1, 1, 2]},
            {"opcode": "BLT", "args": [1, 2, 0]},
            {"opcode": "BNL", "args": [4, 1, 3]},
            {"opcode": "BGT", "args": [3, 2, 3]},
            {"opcode": "BNG", "args": [3, 2, 3]},

        ]
        for instr in init:
            encoded = encode_instr(instr)
            opcode, rd, rs1, rs2, imm = decode_instr(
            int.from_bytes(encoded, "little"))

            self.assertEqual(opcode.name, instr["opcode"])
            self.assertListEqual([rs1,rs2,imm], instr["args"])
            self.assertEqual(rd, 0)
