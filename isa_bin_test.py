#!/usr/bin/python3
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=missing-module-docstring

import unittest

from isa import Instruction, decode_opcode, normalize, encode_instr


class TestBinarizator(unittest.TestCase):

    def test_normalize(self):
        code = [
            {"opcode": "ADDI", "args": ["3", "0", 969]},
            {"opcode": "LW", "args": ["1", "3"]},
            {"opcode": "SW", "args": ["2", "1"]},
            {"opcode": "BEQ", "args": ["1", "0", 99]},
            {"opcode": "HALT", "args": []}]
        benchmark = [
            {'opcode': 'ADDI', 'args': [3, 0, 969]},
            {'opcode': 'LW', 'args': [1, 3, 0]},
            {'opcode': 'SW', 'args': [0, 2, 1]},
            {'opcode': 'BEQ', 'args': [1, 0, 99]},
            {"opcode": "HALT", "args": []}
        ]
        normalized_code = normalize(code)
        print(normalized_code)

        for word_idx, word in enumerate(normalized_code):
            self.assertEqual(
                word['opcode'], benchmark[word_idx]['opcode'])
            self.assertListEqual(word["args"], benchmark[word_idx]["args"])

    def test_register_instruction(self):
        init = [
            {"opcode": "ADD", "args": [1, 2, 3]},
            {"opcode": "SUB", "args": [1, 1, 2]},
            {"opcode": "REM", "args": [1, 2, 0]},
            {"opcode": "DIV", "args": [3, 1, 3]},
            {"opcode": "MUL", "args": [3, 2, 3]},
        ]
        for instr in init:
            encoded = encode_instr(instr)
            instr32 = int.from_bytes(encoded, "little")
            opcode = decode_opcode(Instruction.fetch_opcode(instr32))

            rd = Instruction.fetch_rd(instr32)
            rs1 = Instruction.fetch_rs1(instr32)
            rs2 = Instruction.fetch_rs2(instr32)

            self.assertEqual(opcode.name, instr["opcode"])
            self.assertListEqual([rd, rs1, rs2], instr["args"])

    def test_immediate_instruction(self):
        init = [
            {"opcode": "ADDI", "args": [1, 2, 3]},
            {"opcode": "SUBI", "args": [1, 1, 2]},
            {"opcode": "REMI", "args": [1, 2, 0]},
            {"opcode": "DIVI", "args": [2, 1, 3]},
            {"opcode": "MULI", "args": [3, 2, 3]},
        ]
        for instr in init:
            encoded = encode_instr(instr)
            instr32 = int.from_bytes(encoded, "little")
            opcode = decode_opcode(Instruction.fetch_opcode(instr32))

            rd = Instruction.fetch_rd(instr32)
            rs1 = Instruction.fetch_rs1(instr32)
            imm = opcode.instruction_type.fetch_imm(instr32)

            self.assertEqual(opcode.name, instr["opcode"])
            self.assertListEqual([rd, rs1, imm], instr["args"])

    def test_branch_instruction(self):
        init = [
            {"opcode": "BEQ", "args": [1, 2, 3]},
            {"opcode": "BNE", "args": [1, 1, 2]},
            {"opcode": "BLT", "args": [1, 2, 0]},
            {"opcode": "BNL", "args": [2, 1, 3]},
            {"opcode": "BGT", "args": [3, 2, 3]},
            {"opcode": "BNG", "args": [3, 2, 3]},

        ]
        for instr in init:
            encoded = encode_instr(instr)
            instr32 = int.from_bytes(encoded, "little")
            opcode = decode_opcode(Instruction.fetch_opcode(instr32))
            rs1 = Instruction.fetch_rs1(instr32)
            rs2 = Instruction.fetch_rs2(instr32)
            imm = opcode.instruction_type.fetch_imm(instr32)

            self.assertEqual(opcode.name, instr["opcode"])
            self.assertListEqual([rs1, rs2, imm], instr["args"])
