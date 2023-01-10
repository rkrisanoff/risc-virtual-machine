# pylint: disable=missing-class-docstring     # чтобы не быть Капитаном Очевидностью
# pylint: disable=missing-function-docstring  # чтобы не быть Капитаном Очевидностью
# pylint: disable=line-too-long               # строки с ожидаемым выводом
# pylint: disable=redefined-builtin
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
                             'source LoC: 46 code instr: 46 code bytes: 188\n'
                             'Output is `Arhitecture of Computer\x00`\n'
                             'instr_counter: 240 ticks: 581\n'
                             )
#             self.assertEqual(logs.output[2],
#                              '''INFO:root:Memory map is
# [       ] [{   35}    [0000100011]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   34}    [0000100010]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   33}    [0000100001]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   32}    [0000100000]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   31}    [0000011111]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   30}    [0000011110]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   29}    [0000011101]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   28}    [0000011100]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   27}    [0000011011]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   26}    [0000011010]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   25}    [0000011001]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   24}    [0000011000]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   23}    [0000010111]  -> [00000000000000000000000000000000] = (         0)
# [       ] [{   22}    [0000010110]  -> [00000000000000000000000001110010] = (       114)
# [       ] [{   21}    [0000010101]  -> [00000000000000000000000001100101] = (       101)
# [       ] [{   20}    [0000010100]  -> [00000000000000000000000001110100] = (       116)
# [       ] [{   19}    [0000010011]  -> [00000000000000000000000001110101] = (       117)
# [       ] [{   18}    [0000010010]  -> [00000000000000000000000001110000] = (       112)
# [       ] [{   17}    [0000010001]  -> [00000000000000000000000001101101] = (       109)
# [       ] [{   16}    [0000010000]  -> [00000000000000000000000001101111] = (       111)
# [       ] [{   15}    [0000001111]  -> [00000000000000000000000001000011] = (        67)
# [       ] [{   14}    [0000001110]  -> [00000000000000000000000000100000] = (        32)
# [       ] [{   13}    [0000001101]  -> [00000000000000000000000001100110] = (       102)
# [       ] [{   12}    [0000001100]  -> [00000000000000000000000001101111] = (       111)
# [       ] [{   11}    [0000001011]  -> [00000000000000000000000000100000] = (        32)
# [       ] [{   10}    [0000001010]  -> [00000000000000000000000001100101] = (       101)
# [       ] [{    9}    [0000001001]  -> [00000000000000000000000001110010] = (       114)
# [       ] [{    8}    [0000001000]  -> [00000000000000000000000001110101] = (       117)
# [       ] [{    7}    [0000000111]  -> [00000000000000000000000001110100] = (       116)
# [       ] [{    6}    [0000000110]  -> [00000000000000000000000001100011] = (        99)
# [       ] [{    5}    [0000000101]  -> [00000000000000000000000001100101] = (       101)
# [       ] [{    4}    [0000000100]  -> [00000000000000000000000001110100] = (       116)
# [       ] [{    3}    [0000000011]  -> [00000000000000000000000001101001] = (       105)
# [       ] [{    2}    [0000000010]  -> [00000000000000000000000001101000] = (       104)
# [       ] [{    1}    [0000000001]  -> [00000000000000000000000001110010] = (       114)
# [BUFFERS] [{    0}    [0000000000]  -> [00000000000000000000000001000001] = (        65)
# ''')

    def test_hello(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            source = "./tests/sources/hello.asm"
            target = os.path.join(tmpdirname, "machine_code.out")
            input_stream = "./tests/inputs/input.txt"

            # Собираем весь стандартный вывод в переменную stdout.
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                translator.main([source, target, target])
                machine.main([target, input_stream])
            print(stdout.getvalue())
            self.assertEqual(stdout.getvalue(),
                             'source LoC: 33 code instr: 23 code bytes: 96\n'
                             'Output is `Hello, World!`\n'
                             'instr_counter: 69 ticks: 169\n'
                             )

    def test_prop5(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            source = "./tests/sources/prob5.asm"
            target = os.path.join(tmpdirname, "machine_code.out")
            input_stream = "./tests/inputs/input3.txt"

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                # Собираем журнал событий по уровню INFO в переменную logs.
                # with self.assertLogs('', level='INFO') as logs:
                translator.main([source, target, target])
                machine.main([target, input_stream])
                self.assertEqual(stdout.getvalue(),
                                 'source LoC: 242 code instr: 112 code bytes: 452\n'
                                 'Output is `232792560`\n'
                                 'instr_counter: 3777 ticks: 10399\n')
#                     self.assertEqual(logs.output[2],
#                                      '''INFO:root:Memory map is
# [              ] [{   35}    [0000100011]  -> [00000000000000000000000000000000] = (         0)
# [              ] [{   34}    [0000100010]  -> [00000000000000000000000000110000] = (        48)
# [              ] [{   33}    [0000100001]  -> [00000000000000000000000000110110] = (        54)
# [              ] [{   32}    [0000100000]  -> [00000000000000000000000000110101] = (        53)
# [              ] [{   31}    [0000011111]  -> [00000000000000000000000000110010] = (        50)
# [              ] [{   30}    [0000011110]  -> [00000000000000000000000000111001] = (        57)
# [              ] [{   29}    [0000011101]  -> [00000000000000000000000000110111] = (        55)
# [              ] [{   28}    [0000011100]  -> [00000000000000000000000000110010] = (        50)
# [              ] [{   27}    [0000011011]  -> [00000000000000000000000000110011] = (        51)
# [              ] [{   26}    [0000011010]  -> [00000000000000000000000000110010] = (        50)
# [              ] [{   25}    [0000011001]  -> [00000000000000000000000000000000] = (         0)
# [DIVISOR       ] [{   24}    [0000011000]  -> [00000000000000000000000000010011] = (        19)
# [RESULT        ] [{   23}    [0000010111]  -> [00001101111000000010000111110000] = ( 232792560)
# [DIVISOR_IDX   ] [{   22}    [0000010110]  -> [00000000000000000000000000001100] = (        12)
# [CURRENT_NUMBER] [{   21}    [0000010101]  -> [00000000000000000000000000000001] = (         1)
# [N             ] [{   20}    [0000010100]  -> [00000000000000000000000000010101] = (        21)
# [              ] [{   19}    [0000010011]  -> [00000000000000000000000000000000] = (         0)
# [              ] [{   18}    [0000010010]  -> [00000000000000000000000000000000] = (         0)
# [              ] [{   17}    [0000010001]  -> [00000000000000000000000000000000] = (         0)
# [              ] [{   16}    [0000010000]  -> [00000000000000000000000000000000] = (         0)
# [              ] [{   15}    [0000001111]  -> [00000000000000000000000000000000] = (         0)
# [              ] [{   14}    [0000001110]  -> [00000000000000000000000000000000] = (         0)
# [              ] [{   13}    [0000001101]  -> [00000000000000000000000000000000] = (         0)
# [              ] [{   12}    [0000001100]  -> [00000000000000000000000000000000] = (         0)
# [              ] [{   11}    [0000001011]  -> [00000000000000000000000000010011] = (        19)
# [              ] [{   10}    [0000001010]  -> [00000000000000000000000000010001] = (        17)
# [              ] [{    9}    [0000001001]  -> [00000000000000000000000000000010] = (         2)
# [              ] [{    8}    [0000001000]  -> [00000000000000000000000000001101] = (        13)
# [              ] [{    7}    [0000000111]  -> [00000000000000000000000000001011] = (        11)
# [              ] [{    6}    [0000000110]  -> [00000000000000000000000000000011] = (         3)
# [              ] [{    5}    [0000000101]  -> [00000000000000000000000000000010] = (         2)
# [              ] [{    4}    [0000000100]  -> [00000000000000000000000000000111] = (         7)
# [              ] [{    3}    [0000000011]  -> [00000000000000000000000000000101] = (         5)
# [              ] [{    2}    [0000000010]  -> [00000000000000000000000000000010] = (         2)
# [              ] [{    1}    [0000000001]  -> [00000000000000000000000000000011] = (         3)
# [DIVISORS      ] [{    0}    [0000000000]  -> [00000000000000000000000000000010] = (         2)
# ''')
