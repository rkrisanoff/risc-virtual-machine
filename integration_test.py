# pylint: disable=missing-class-docstring     # чтобы не быть Капитаном Очевидностью
# pylint: disable=missing-function-docstring  # чтобы не быть Капитаном Очевидностью
# pylint: disable=line-too-long               # строки с ожидаемым выводом
"""Интеграционные тесты транслятора и машины
"""

import contextlib
import io
import os
import tempfile
import unittest

import machine
import translator


class TestCases(unittest.TestCase):

    def test_cat(self):
        # Создаём временную папку для скомпилированного файла. Удаляется автоматически.
        with tempfile.TemporaryDirectory() as tmpdirname:
            source = "./tests/sources/cat.asm"
            target = os.path.join(tmpdirname, "machine_code.out")
            input_stream = "./tests/inputs/input2.txt"

            # Собираем весь стандартный вывод в переменную stdout.
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                # with self.assertLogs('', level='INFO') as logs:
                translator.main([source, target, target])
                machine.main([target, input_stream])
                self.assertEqual(stdout.getvalue(),
                                 'source LoC: 40 code instr: 15 code bytes: 188\n'
                                 'Output is `Arhitecture of Computer\x00`\n'
                                 'instr_counter: 240 ticks: 1201\n'
                                 )

    def test_hello(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            source = "./tests/sources/hello.asm"
            target = os.path.join(tmpdirname, "machine_code.out")
            input_stream = "./tests/inputs/input.txt"

            # Собираем весь стандартный вывод в переменную stdout.
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                translator.main([source, target, target])
                machine.main([target, input_stream])
            self.assertEqual(stdout.getvalue(),
                             'source LoC: 25 code instr: 8 code bytes: 92\n'
                             'Output is `Hello, World!`\n'
                             'instr_counter: 69 ticks: 346\n'
                             )

    def test_prop5(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            source = "./tests/sources/prob5.asm"
            target = os.path.join(tmpdirname, "machine_code.out")
            input_stream = "./tests/inputs/input.txt"

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                # Собираем журнал событий по уровню INFO в переменную logs.
                # with self.assertLogs('', level='INFO') as logs:
                translator.main([source, target, target])
                machine.main([target, input_stream])
                self.assertEqual(stdout.getvalue(),
                                 'source LoC: 242 code instr: 89 code bytes: 496\n'
                                 'Output is `232792560`\n'
                                 'instr_counter: 3778 ticks: 18891\n')
