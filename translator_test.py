#!/usr/bin/python3
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string
# pylint: disable=line-too-long
import unittest

from translator import translate, parse, allocate, pre_process, tokenize
from isa import read_json_code


class TestTranslatatorHello(unittest.TestCase):

    def test_pre_process(self):
        raw = '''section data:
hello: 'Hello, World!',0
lenght: 13

section text:
    _start:
        addi x2,x0,hello
        addi x3,x0,969 # 969 - OUTPUT
    write:
        lw x1,x2
        beq x1,x0,end # f
        sw x3,x1
        addi x2,x2,1
        jmp write
    end:
        halt'''
        post_processed = pre_process(raw)

        self.assertEqual(post_processed, "section data: hello: 'Hello, World!',0 lenght: 13 "
                         "section text: _start: addi x2,x0,hello addi x3,x0,969 write: lw x1,x2 beq x1,x0,end sw x3,x1 addi x2,x2,1 jmp write end: halt",
                         'failed preprocessing')

    def test_tokenize(self):
        data_tokens, code_tokens = tokenize(
            "section data: hello: 'Hello, World!',0 lenght: 13\
             section text: _start: addi x2,x0,hello addi x3,x0,969 write: lw x1,x2 beq x1,x0,end sw x3,x1 addi x2,x2,1 jmp write end: halt"
        )

        self.assertEqual(data_tokens, [('hello',), '72', '101', '108', '108', '111',
                         '44', '32', '87', '111', '114', '108', '100', '33', '0', ('lenght',), '13'])
        self.assertEqual(code_tokens, [('_start',), 'addi', 'x2', 'x0', 'hello', 'addi', 'x3', 'x0', '969', ('write',), 'lw',
                         'x1', 'x2', 'beq', 'x1', 'x0', 'end', 'sw', 'x3', 'x1', 'addi', 'x2', 'x2', '1', 'jmp', 'write', ('end',), 'halt'])

    def test_allocate(self):
        tokens = [('hello',), '72', '101', '108', '108', '111',
                  '44', '32', '87', '111', '114', '108', '100', '33', '0', ('lenght',), '13']
        data, labels = allocate(tokens)
        self.assertEqual(data, ['72', '101', '108', '108', '111', '44',
                         '32', '87', '111', '114', '108', '100', '33', '0', '13'])
        self.assertDictEqual(labels, {'hello': 0, 'lenght': 14})

    def test_parse(self):
        tokens = [('_start',), 'addi', 'x2', 'x0', 'hello', 'addi', 'x3', 'x0', '969', ('write',), 'lw', 'x1', 'x2',
                  'beq', 'x1', 'x0', 'end', 'sw', 'x3', 'x1', 'addi', 'x2', 'x2', '1', 'jmp', 'write', ('end',), 'halt']
        code, labels = parse(tokens)

        benchmark = [
            {'opcode': 'ADDI', 'args': ['2', '0', 'hello']},
            {'opcode': 'ADDI', 'args': ['3', '0', '969']},
            {'opcode': 'LW', 'args': ['1', '2']},
            {'opcode': 'BEQ', 'args': ['1', '0', 'end']},
            {'opcode': 'SW', 'args': ['3', '1']},
            {'opcode': 'ADDI', 'args': ['2', '2', '1']},
            {'opcode': 'JMP', 'args': ['write']},
            {'opcode': 'HALT', 'args': []}
        ]
        self.assertDictEqual(labels, {'_start': 0, 'write': 2, 'end': 7})
        for instr_idx, instr in enumerate(benchmark):
            self.assertEqual(code[instr_idx]["opcode"], instr["opcode"])
            self.assertEqual(code[instr_idx]["args"], instr["args"])

    def test_translate(self):
        source = '''section data:
hello: 'Hello, World!',0
lenght: 13

section text:
    _start:
        addi x2,x0,hello
        addi x3,x0,969 # 969 - OUTPUT
    write:
        lw x1,x2
        beq x1,x0,end # f
        sw x3,x1
        addi x2,x2,1
        jmp write
    end:
        halt'''
        translated = translate(source)
        cat = read_json_code("./tests/correct/hello.json")
        for instr_idx, instr in enumerate(cat):
            if isinstance(instr, dict):
                self.assertEqual(
                    translated[instr_idx]["opcode"], instr["opcode"])
                self.assertEqual(translated[instr_idx]["args"], instr["args"])
            else:
                self.assertEqual(translated[instr_idx], instr)


class TestTranslatatorCat(unittest.TestCase):

    def test_pre_process(self):
        raw = '''section data:
buffer: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

section text:
    _start:
        addi x2,x0,buffer
        addi x3,x0,696 # STDIN
    read:
        lw x1,x3
        sw x2,x1
        beq x1,x0,finish_read
        addi x2,x2,1
        jmp read
    finish_read:
        addi x2,x0,buffer
        addi x3,x0,969 # STDOUT

    write:
        lw x1,x2
        sw x3,x1
        beq x1,x0,end # f
        addi x2,x2,1
        jmp write
    end:
        halt'''
        post_processed = pre_process(raw)

        self.assertEqual(post_processed, "section data: buffer: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 "
                         "section text: _start: addi x2,x0,buffer addi x3,x0,696 read: lw x1,x3 sw x2,x1 beq x1,x0,finish_read addi x2,x2,1 jmp read "
                         "finish_read: addi x2,x0,buffer addi x3,x0,969 write: lw x1,x2 sw x3,x1 beq x1,x0,end addi x2,x2,1 jmp write end: halt",
                         'failed preprocessing')

    def test_tokenize(self):
        data_tokens, code_tokens = tokenize(
            "section data: buffer: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 "
            "section text: _start: addi 2,0,buffer addi 3,0,696 read: lw 1,3 sw 2,1 beq 1,0,finish_read addi 2,2,1 jmp read "
            "finish_read: addi 2,0,buffer addi 3,0,969 write: lw 1,2 sw 3,1 beq 1,0,end addi 2,2,1 jmp write end: halt"
        )

        self.assertEqual(data_tokens, [('buffer',), '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
                         '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'])
        self.assertEqual(code_tokens, [
            ('_start',), 'addi', '2', '0', 'buffer', 'addi', '3', '0', '696',
            ('read',), 'lw', '1', '3', 'sw', '2', '1', 'beq', '1', '0', 'finish_read', 'addi', '2', '2', '1', 'jmp', 'read',
            ('finish_read',), 'addi', '2', '0', 'buffer', 'addi', '3', '0', '969',
            ('write',), 'lw', '1', '2', 'sw', '3', '1', 'beq', '1', '0', 'end', 'addi', '2', '2', '1', 'jmp', 'write',
            ('end',), 'halt'
        ])

    def test_allocate(self):
        tokens = [('buffer',), '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
                  '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
        data, labels = allocate(tokens)
        self.assertEqual(data, ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
                         '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'])
        self.assertDictEqual(labels, {'buffer': 0})

    def test_parse(self):
        tokens = [
            ('_start',), 'addi', '2', '0', 'buffer', 'addi', '3', '0', '696',
            ('read',), 'lw', '1', '3', 'sw', '2', '1', 'beq', '1', '0', 'finish_read', 'addi', '2', '2', '1', 'jmp', 'read',
            ('finish_read',), 'addi', '2', '0', 'buffer', 'addi', '3', '0', '969',
            ('write',), 'lw', '1', '2', 'sw', '3', '1', 'beq', '1', '0', 'end', 'addi', '2', '2', '1', 'jmp', 'write',
            ('end',), 'halt'
        ]
        code, labels = parse(tokens)

        benchmark = [
            {'opcode': 'ADDI', 'args': ['2', '0', 'buffer']},
            {'opcode': 'ADDI', 'args': ['3', '0', '696']},
            {'opcode': 'LW', 'args': ['1', '3']},
            {'opcode': 'SW', 'args': ['2', '1']},
            {'opcode': 'BEQ', 'args': ['1', '0', 'finish_read']},
            {'opcode': 'ADDI', 'args': ['2', '2', '1']},
            {'opcode': 'JMP', 'args': ['read']},
            {'opcode': 'ADDI', 'args': ['2', '0', 'buffer']},
            {'opcode': 'ADDI', 'args': ['3', '0', '969']},
            {'opcode': 'LW', 'args': ['1', '2']},
            {'opcode': 'SW', 'args': ['3', '1']},
            {'opcode': 'BEQ', 'args': ['1', '0', 'end']},
            {'opcode': 'ADDI', 'args': ['2', '2', '1']},
            {'opcode': 'JMP', 'args': ['write']},
            {'opcode': 'HALT', 'args': []}]
        self.assertDictEqual(
            labels, {'_start': 0, 'read': 2, 'finish_read': 7, 'write': 9, 'end': 14})
        for instr_idx, instr in enumerate(benchmark):
            self.assertEqual(code[instr_idx]["opcode"], instr["opcode"])
            self.assertEqual(code[instr_idx]["args"], instr["args"])

    def test_translate(self):
        source = '''section data:
buffer: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

section text:
    _start:
        addi 2,0,buffer
        addi 3,0,696 # STDIN
    read:
        lw 1,3
        sw 2,1
        beq 1,0,finish_read
        addi 2,2,1
        jmp read
    finish_read:
        addi 2,0,buffer
        addi 3,0,969 # STDOUT

    write:
        lw 1,2
        sw 3,1
        beq 1,0,end # f
        addi 2,2,1
        jmp write
    end:
        halt'''
        translated = translate(source)
        cat = read_json_code("./tests/correct/cat.json")
        for instr_idx, instr in enumerate(cat):
            if isinstance(instr, dict):
                self.assertEqual(
                    translated[instr_idx]["opcode"], instr["opcode"])
                self.assertEqual(translated[instr_idx]["args"], instr["args"])
            else:
                self.assertEqual(translated[instr_idx], instr)
