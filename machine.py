#!/usr/bin/python3
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes

"""Модель процессора, позволяющая выполнить странслированные программы на языке DrukharyLisp.
"""
from enum import Enum
import logging
from collections import deque
import sys
from typing import Callable

from isa import Register, decode_opcode, read_bin_code, format_instr, Opcode,\
    INPUT, OUTPUT, Immediate, Instruction


class DataPath():
    memory: list[int]
    program_counter: int
    data_address: int
    data_memory_size: int

    imm_gen: int
    instruction: int
    registers: list[int]

    def __init__(self, data: list[int], code: list[int], input_buffer: list):
        self.program_counter = 0
        self.imem = code
        self.dmem = data
        self.data_address = 0
        self.imm_gen = 0
        self.current_data = 0
        self.registers = [0] * 4
        self.raw_imm = 0
        self.rd = 0
        self.rs1 = 0
        self.rs2 = 0
        self.input_buffer = deque([ord(token) for token in input_buffer] + [0])
        self.output_buffer = deque()
        self.a, self.b = 0, 0
        self.computed = 0

    def select_instruction(self) -> int:
        self.instruction = self.imem[self.program_counter]
        self.program_counter += 1
        self.rd = Instruction.fetch_rd(self.instruction)
        self.rs1 = Instruction.fetch_rs1(self.instruction)
        self.rs2 = Instruction.fetch_rs2(self.instruction)
        self.raw_imm = self.instruction
        return self.instruction

    def generate_immediate(self, fetch_imm: Callable[[int], int]):
        self.imm_gen = fetch_imm(self.raw_imm)

    def latch_rs1_to_alu(self):
        self.a = self.registers[self.rs1]

    def latch_rs2_to_alu(self):
        self.b = self.registers[self.rs2]

    def latch_imm_to_alu(self):
        """Загружает непосредственное значение в ALU"""
        self.b = self.imm_gen

    def compute_ALU(self, opcode: Opcode):
        if opcode in (Opcode.ADD, Opcode.ADDI):
            self.computed = self.a + self.b
        elif opcode in (Opcode.SUB, Opcode.SUBI):
            self.computed = self.a - self.b
        elif opcode in (Opcode.MUL, Opcode.MULI):
            self.computed = self.a * self.b
        elif opcode in (Opcode.DIV, Opcode.DIVI):
            self.computed = self.a // self.b
        elif opcode in (Opcode.REM, Opcode.REMI):
            self.computed = self.a % self.b
        self.computed = int(self.computed)

    def latch_address_to_memory(self):
        """Загружает целевой адрес в память"""

        if self.registers[self.rs1] == INPUT:
            if not self.input_buffer:
                raise EOFError
            self.current_data = self.input_buffer.popleft()
        else:
            self.data_address = self.registers[self.rs1]
            self.current_data = self.dmem[self.data_address]

    def store_data_to_memory_from_reg(self):
        """Загружает данные в память"""
        if self.registers[self.rs1] == OUTPUT:
            self.output_buffer.append(chr(self.registers[self.rs2]))

        else:
            self.dmem[self.registers[self.rs1]
                      ] = self.registers[self.rs2]

    def store_data_to_memory_from_imm(self):
        """Загружает данные в память"""
        if self.registers[self.rs1] == OUTPUT:
            self.output_buffer.append(chr(self.imm_gen))
        else:
            self.dmem[self.registers[self.rs1]] = self.imm_gen

    def latch_address_to_memory_from_imm(self):
        if self.imm_gen == INPUT:
            if not self.input_buffer:
                raise EOFError
            self.current_data = self.input_buffer.popleft()
        else:
            self.data_address = self.imm_gen
            self.current_data = self.dmem[self.data_address]

    def latch_rd_from_memory(self):
        """Значение из памяти перезаписывает регистр"""
        self.registers[self.rd] = self.current_data

    def latch_rd_from_alu(self):
        """ALU перезаписывает регистр"""
        self.registers[self.rd] = self.computed

    def latch_program_counter(self):
        """Перезаписывает значение PC из ImmGen"""
        self.program_counter = self.imm_gen

    def latch_compare_flags(self):
        """Загружает регистры в Branch Comparator."""
        return self.registers[self.rs1] == self.registers[self.rs2],\
            self.registers[self.rs1] < self.registers[self.rs2]


class InstructionStage(Enum):
    FETCH_INSTRUCTION = 0
    DECODE_INSTRUCTION = 1
    EXECUTE = 2
    MEMORY_ACCESS = 3
    WRITE_BACK = 4


class ControlUnit():
    data_path: DataPath
    stage: InstructionStage
    opcode: Opcode

    def __init__(self, data_path):
        self.data_path = data_path
        self._tick_ = 0
        self.stage = InstructionStage.FETCH_INSTRUCTION
        self.instr = 0

        self.equals, self.less = False, False

    def complete_stage(self):
        if self.stage is InstructionStage.FETCH_INSTRUCTION:
            self.fetch_instruction()
            logging.debug(" <== %s ==> [%s]", format_instr(self.instr),bin(self.instr))
        elif self.stage is InstructionStage.DECODE_INSTRUCTION:
            self.decoding()
        elif self.stage is InstructionStage.EXECUTE:
            self.execute()
        elif self.stage is InstructionStage.MEMORY_ACCESS:
            self.memory_access()
        elif self.stage is InstructionStage.WRITE_BACK:
            self.write_back()

        self.tick()
        self.stage = InstructionStage((self.stage.value + 1) % 5)

    def tick(self):
        """Счётчик тактов процессора. Вызывается при переходе на следующий такт."""
        logging.debug('%s', self)
        self._tick_ += 1

    def current_tick(self):
        """Возвращает текущий такт."""
        return self._tick_

    def fetch_instruction(self):
        """Извлекает инструкцию из памяти"""
        self.instr = self.data_path.select_instruction()

    def decoding(self):
        """Декодирует инструкцию"""
        self.opcode = decode_opcode(self.instr)

        if self.opcode is Opcode.HALT:
            raise StopIteration()

        self.data_path.generate_immediate(
            self.opcode.instruction_type.fetch_imm)

    def execute(self):
        self.equals, self.less = self.data_path.latch_compare_flags()

        self.data_path.latch_rs1_to_alu()

        if self.opcode.instruction_type is Register:
            self.data_path.latch_rs2_to_alu()
        elif self.opcode.instruction_type is Immediate:
            self.data_path.latch_imm_to_alu()

        self.data_path.compute_ALU(opcode=self.opcode)

    def memory_access(self):
        if self.opcode is Opcode.LWI:
            self.data_path.latch_address_to_memory_from_imm()
        elif self.opcode is Opcode.LW:
            self.data_path.latch_address_to_memory()
        elif self.opcode is Opcode.SW:
            self.data_path.store_data_to_memory_from_reg()
        elif self.opcode is Opcode.SWI:
            self.data_path.store_data_to_memory_from_imm()

    def write_back(self):
        if self.opcode in (Opcode.LW, Opcode.LWI):
            self.data_path.latch_rd_from_memory()
        elif self.opcode.instruction_type in (Immediate, Register)\
                and self.opcode not in (Opcode.LW, Opcode.SW):
            self.data_path.latch_rd_from_alu()
        elif any([
            self.opcode is Opcode.JMP,
            self.opcode is Opcode.BEQ and self.equals,
            self.opcode is Opcode.BNE and not self.equals,
            self.opcode is Opcode.BLT and self.less,
            self.opcode is Opcode.BNL and not self.less,
            self.opcode is Opcode.BGT and not self.less and not self.equals,
            self.opcode is Opcode.BNG and (self.less or self.equals)
        ]):
            self.data_path.latch_program_counter()

    def __repr__(self):
        state = f"{{TICK: {self._tick_}"\
                f" PC: {self.data_path.program_counter}"\
                f" ADDR: {self.data_path.data_address}}}"

        registers = f"{{[rd: {self.data_path.rd}"\
            f", rs1: {self.data_path.rs1}"\
            f", rs2: {self.data_path.rs2}"\
            f", imm: {self.data_path.imm_gen}]"\
            f" Regs [{' '.join([str(reg) for reg in self.data_path.registers])}] }}"

        alu = f"ALU [a:{self.data_path.a} b:{self.data_path.b} computed:{self.data_path.computed}]"

        return f"{state} {registers} {alu}"


def show_data_memory(memory):
    data_memory_state = ""
    for address, cell in enumerate(reversed(memory)):
        address = len(memory) - address - 1
        address_br = bin(address)[2:]
        address_br = (10 - len(address_br)) * "0" + address_br
        cell = int(cell)
        cell_br = bin(cell)[2:]
        cell_br = (32 - len(cell_br)) * "0" + cell_br
        data_memory_state += f"({address:5})\
        [{address_br:10}]  -> [{cell_br:32}] = ({cell:10})"
        data_memory_state += "\n"
    return data_memory_state


def show_instr_memory(memory):
    instr_memory_state = ""
    for address, cell in enumerate(reversed(memory)):
        address = len(memory) - address - 1
        address_br = bin(address)[2:]
        address_br = (10 - len(address_br)) * "0" + address_br
        cell = int(cell)
        cell_br = bin(cell)[2:]
        cell_br = (32 - len(cell_br)) * "0" + cell_br
        instr_memory_state += f"({address:5})\
        [{address_br:10}]  -> [{cell_br:32}]" f" ~ {format_instr(cell):20}\n"

    return instr_memory_state


def simulation(data: list[int], code: list[int], input_tokens, limit):
    """Запуск симуляции процессора.

    Длительность моделирования ограничена количеством выполненных инструкций.
    """
    logging.info("{ INPUT MESSAGE } [ `%s` ]", "".join(list(input_tokens)))
    logging.info("{ INPUT TOKENS  } [ %s ]", ",".join(
        [str(ord(token)) for token in input_tokens]))
    dmem, imem = show_data_memory(data), show_instr_memory(code)
    logging.debug("%s", f"Instruction memory map is\n{imem}")
    logging.debug("%s", f"Data memory map is\n{dmem}")
    data_path = DataPath(data, code, input_tokens)
    control_unit = ControlUnit(data_path)
    tick_counter = 0
    try:
        while True:
            if not limit > tick_counter:
                print("too long execution, increase limit!")
                break
            control_unit.complete_stage()
            tick_counter += 1

    except EOFError:
        logging.warning('Input buffer is empty!')
    except StopIteration:
        pass

    finally:
        dmem = show_data_memory(data_path.dmem)
        logging.debug("%s", f"Data memory map is\n{dmem}")

    return ''.join(data_path.output_buffer), tick_counter // 5,\
        control_unit.current_tick()


def main(args):

    assert len(args) == 2,\
        "Wrong arguments: machine.py <code.bin> <input>"
    code_file, input_file = args

    data, code = read_bin_code(code_file)

    with open(input_file, encoding="utf-8") as file:
        input_tokens = file.read()

    output, instr_counter, ticks = simulation(
        data, code,
        input_tokens=input_tokens,
        limit=20000
    )

    print(f"Output is `{''.join(output)}`")
    print(f"instr_counter: {instr_counter} ticks: {ticks}")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    main(sys.argv[1:])
