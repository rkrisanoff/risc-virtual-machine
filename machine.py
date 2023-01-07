#!/usr/bin/python3
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string

"""Модель процессора, позволяющая выполнить странслированные программы на языке DrukharyLisp.
"""
import logging
from collections import deque
import sys

from isa import encode_instruct, read_bin_code, Opcode,\
    ops_gr, STDIN, STDOUT


class RegisterUnit:
    registers: list[int]
    reg_d: int
    reg_a: int
    reg_b: int

    def __init__(self, registers_count: int, stack_vertex: int) -> None:
        self.registers = [0] * registers_count
        self.registers[registers_count - 1] = stack_vertex
        self.reg_d = 0
        self.reg_a = 0
        self.reg_b = 0

    def latch_sel_tar_reg(self, number):
        self.reg_d = number

    def latch_sel_a_reg(self, number):
        self.reg_a = number

    def latch_sel_b_reg(self, number):
        self.reg_b = number

    def get_a_data(self):
        return self.registers[self.reg_a]

    def get_b_data(self):
        return self.registers[self.reg_b]

    def set_dest_data(self, data):

        if self.reg_d != 0:
            self.registers[self.reg_d] = int(data)


class ALU:
    output: int
    a: int
    b: int
    _operations_ = {
        Opcode.ADD: lambda a, b: a + b,
        Opcode.ADDI: lambda a, b: a + b,
        Opcode.SUB: lambda a, b: a - b,
        Opcode.SUBI: lambda a, b: a - b,
        Opcode.MUL: lambda a, b: a * b,
        Opcode.MULI: lambda a, b: a * b,
        Opcode.DIV: lambda a, b: a // b,
        Opcode.DIVI: lambda a, b: a // b,
        Opcode.REM: lambda a, b: a % b,
        Opcode.REMI: lambda a, b: a % b
    }

    def __init__(self) -> None:
        self.output = 0
        self.a = 0
        self.b = 0

    def load(self, a, b):
        self.a = a
        self.b = b

    def compute(self, operation) -> int:
        self.output = int(self._operations_[operation](self.a, self.b))
        return self.output


class BranchComparator:
    a: int
    b: int

    def __init__(self) -> None:
        self.a = 0
        self.b = 0

    def load(self, a, b):
        self.a = a
        self.b = b

    def compare(self) -> tuple[bool, bool]:
        return self.a == self.b,\
            self.a < self.b


class IO:
    input_buffer: deque

    def __init__(self, input_tokens: list) -> None:
        self.input_buffer = deque(input_tokens)
        self.output_buffer = deque()

    def eof(self):
        return not self.input_buffer

    def input(self):
        return self.input_buffer.popleft()

    def output(self, character):
        self.output_buffer.append(character)


class DataPath():
    memory: list[dict | str]
    program_counter: int
    data_address: int
    data_memory_size: int
    memory: list[int]
    ru: RegisterUnit
    alu: ALU
    bc: BranchComparator
    io: IO

    immediately_generator: int
    current_instruction: dict
    args: deque[int]

    def __init__(self, program: list, data_memory_size: int, input_buffer: list):
        self.program_counter = program[0]
        self.data_memory_size = data_memory_size
        self.memory = program[1:] + [0] * (data_memory_size - len(program))
        self.data_address = 0
        self.io = IO([ord(token) for token in input_buffer])
        self.immediately_generator = 0
        self.current_instruction = Opcode.HALT
        self.current_data = 0
        self.ru = RegisterUnit(5, stack_vertex=len(self.memory) - 1)
        self.alu = ALU()
        self.bc = BranchComparator()

    def select_instruction(self) -> Opcode:
        self.current_instruction = self.memory[self.program_counter]
        self.program_counter += 1
        opcode, rd, rs1, rs2, imm = encode_instruct(self.current_instruction)
        self.ru.reg_d = rd
        self.ru.reg_a = rs1
        self.ru.reg_b = rs2
        self.immediately_generator = imm
        return opcode

    def latch_a_reg_to_alu(self):
        self.alu.a = self.ru.get_a_data()

    def latch_b_reg_to_alu(self):
        self.alu.b = self.ru.get_b_data()

    def latch_imm_to_alu(self):
        """Загружает непосредственное значение в ALU"""
        self.alu.b = self.immediately_generator

    def compute_ALU(self, opcode: Opcode):
        self.alu.compute(opcode)

    def latch_address_to_memory(self):
        """Загружает целевой адрес в память"""

        if self.ru.get_a_data() == STDIN:
            if self.io.eof():
                raise EOFError
            self.current_data = self.io.input()
        else:
            self.data_address = self.ru.get_a_data()
            self.current_data = self.memory[self.data_address]

    def store_data_to_memory_from_reg(self):
        """Загружает данные в память"""
        if self.ru.get_a_data() == STDOUT:
            self.io.output(chr(self.ru.get_b_data()))
        else:
            self.memory[self.ru.get_a_data()
                        ] = self.ru.get_b_data()

    def store_data_to_memory_from_imm(self):
        """Загружает данные в память"""
        if self.ru.get_a_data() == STDOUT:
            self.io.output(chr(self.immediately_generator))
        else:
            self.memory[self.ru.get_a_data(
            )] = self.immediately_generator

    def latch_address_to_memory_from_imm(self):
        if self.immediately_generator == STDIN:
            if self.io.eof():
                raise EOFError
            self.current_data = self.io.input()
        else:
            self.data_address = self.immediately_generator
            self.current_data = self.memory[self.data_address]

    def latch_reg_from_memory(self):
        """Значение из памяти перезаписывает регистр"""
        self.ru.set_dest_data(self.current_data)

    def latch_reg_from_alu(self):
        """ALU перезаписывает регистр"""
        self.ru.set_dest_data(self.alu.output)

    def latch_program_counter(self):
        """Перезаписывает значение PC из ImmGen"""
        self.program_counter = self.immediately_generator

    # target_reg because such branch comparator was organized
    def latch_regs_to_bc(self):
        """Загружает регистры в Branch Comparator."""
        self.bc.a, self.bc.b =\
            self.ru.get_a_data(), self.ru.get_b_data()
        return self.bc.compare()


class ControlUnit():
    data_path: DataPath

    def __init__(self, data_path):
        self.data_path = data_path
        self._tick = 0

    def tick(self):
        """Счётчик тактов процессора. Вызывается при переходе на следующий такт."""
        logging.debug('%s', self)
        self._tick += 1

    def current_tick(self):
        """Возвращает текущий такт."""
        return self._tick

    def decode_and_execute_instruction(self):
        opcode = self.data_path.select_instruction()
        dp = self.data_path
        self.tick()

        if opcode is Opcode.JMP:
            dp.latch_program_counter()
        elif opcode in ops_gr["branch"]:
            equals, less = dp.latch_regs_to_bc()
            if any([
                opcode is Opcode.BEQ and equals,
                opcode is Opcode.BNE and not equals,
                opcode is Opcode.BLT and less,
                opcode is Opcode.BNL and not less,
                opcode is Opcode.BGT and not less and not equals,
                opcode is Opcode.BNG and (less or equals)
            ]):
                self.tick()
                dp.latch_program_counter()
        elif opcode is Opcode.LWI:
            dp.latch_address_to_memory_from_imm()
            self.tick()
            dp.latch_reg_from_memory()
        elif opcode is Opcode.LW:
            dp.latch_address_to_memory()
            self.tick()
            dp.latch_reg_from_memory()
        elif opcode is Opcode.SW:
            dp.store_data_to_memory_from_reg()
        elif opcode is Opcode.SWI:
            dp.store_data_to_memory_from_imm()
        elif opcode in ops_gr["arith"]:
            if opcode in ops_gr["imm"]:
                dp.latch_imm_to_alu()
            else:
                dp.latch_b_reg_to_alu()
            dp.latch_a_reg_to_alu()
            dp.compute_ALU(opcode=opcode)
            self.tick()
            dp.latch_reg_from_alu()

        elif opcode is Opcode.HALT:
            raise StopIteration()

        self.tick()

    def __repr__(self):
        state = "{{TICK: {}, PC: {}, ADDR: {}, OUT: }}".format(
            self._tick,
            self.data_path.program_counter,
            self.data_path.data_address
            # self.data_path.output_buffer[0]
        )

        registers = "{{[T: {}, L: {}, R: {}, IM: {}]  Regs {} }}".format(
            self.data_path.ru.reg_d,
            self.data_path.ru.reg_a,
            self.data_path.ru.reg_b,
            self.data_path.immediately_generator,
            f"[{' '.join([str(reg) for reg in self.data_path.ru.registers])}]"
        )

        opcode, rd, rs1, rs2, imm = encode_instruct(
            self.data_path.current_instruction)
        action = "{} {}".format(
            opcode.name, f"[{' '.join([str(arg) for arg in [rd, rs1, rs2, imm]])}]")
        alu = "ALU [a:{} b:{} output:{}]".format(
            self.data_path.alu.a, self.data_path.alu.b, self.data_path.alu.output)

        return "{} {} {} {} ".format(state, registers, alu, action)


def show_memory(memory):
    data_memory_state = ""
    for address, cell in enumerate(reversed(memory)):
        address = len(memory) - address - 1
        address_br = bin(address)[2:]
        address_br = (10 - len(address_br)) * "0" + address_br
        if isinstance(cell, (int, str)):
            cell = int(cell)
            # binary representation == br
            cell_br = bin(cell)[2:]
            cell_br = (32 - len(cell_br)) * "0" + cell_br
            data_memory_state += f"({address:5})\
        [{address_br:10}]  -> [{cell_br:32}] = ({cell:10})\n"
        elif isinstance(cell, dict):
            data_memory_state += f"({address:5})\
        [{address_br:10}]  -> [{'0'*32}] = ({cell['opcode']}\
             {','.join([str(arg) for arg in cell['args']])})\n"
    return data_memory_state


def simulation(program: list[dict | str], input_tokens, data_memory_size, limit):
    """Запуск симуляции процессора.

    Длительность моделирования ограничена количеством выполненных инструкций.
    """
    logging.info("{ INPUT MESSAGE } [ `%s` ]", "".join(input_tokens))
    logging.info("{ INPUT TOKENS  } [ %s ]", ",".join(
        [str(ord(token)) for token in input_tokens]))

    data_path = DataPath(program, data_memory_size, input_tokens)
    control_unit = ControlUnit(data_path)
    instr_counter = 0

    try:
        while True:
            if not limit > instr_counter:
                print("too long execution, increase limit!")
                break
            control_unit.decode_and_execute_instruction()
            instr_counter += 1
    except EOFError:
        logging.warning('Input buffer is empty!')
    except StopIteration:
        pass

    return ''.join(data_path.io.output_buffer), instr_counter,\
        control_unit.current_tick(), show_memory(data_path.memory)


def main(args):

    assert len(args) == 2,\
        "Wrong arguments: machine.py <code.json> <input>"
    code_file, input_file = args

    program = read_bin_code(code_file)
    with open(input_file, encoding="utf-8") as file:
        input_text = file.read()
        input_token = []
        for char in input_text:
            input_token.append(char)
    input_token.append(chr(0))

    output, instr_counter, ticks, data_memory_state = "", "", "", ""
    output, instr_counter, ticks, data_memory_state = simulation(
        program,
        input_tokens=input_token,
        data_memory_size=200,
        limit=15000
    )
    logging.info("%s", f"Memory map is\n{data_memory_state}")

    print(f"Output is `{''.join(output)}`")
    print(f"instr_counter: {instr_counter} ticks: {ticks}")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main(sys.argv[1:])
