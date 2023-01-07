# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

from abc import ABC, abstractmethod

from enum import Enum
import json
from typing import NamedTuple, Tuple

OPCODE_SIZE = 7
REG_SIZE = 5
IMM_SIZE = 10
op_m, op_offs = 0b00000_00000_00000_0000000000_1111111, 0
rd_m = 0b11111_00000_00000_0000000000_0000000
rd_offs = REG_SIZE * 2 + IMM_SIZE + OPCODE_SIZE
rs1_m, rs1_offs = 0b00000_11111_00000_0000000000_0000000, REG_SIZE + \
    IMM_SIZE + OPCODE_SIZE
rs2_m, rs2_offs = 0b00000_00000_11111_0000000000_0000000, IMM_SIZE + OPCODE_SIZE
imm_m_j, imm_j_offs = 0b11111_11111_11111_1111111111_0000000, OPCODE_SIZE
imm_m_l, imm_l_offs = 0b00000_11111_11111_1111111111_0000000, OPCODE_SIZE
imm_m, imm_offs = 0b00000_00000_00000_1111111111_0000000, OPCODE_SIZE


class Instruction(ABC):
    @staticmethod
    @abstractmethod
    def code(opcode: int, args: list[int]) -> int:
        pass

    @staticmethod
    @abstractmethod
    def decode(instruct: int):
        pass

    @staticmethod
    def decode_opcode(instr: int):
        return instr & op_m


class Register(Instruction):
    @staticmethod
    def code(opcode: int, args: list[int]) -> int:
        instruct = 0
        instruct += (opcode << 0) & (op_m)
        instruct += (args[0] << rd_offs) & (rd_m)
        instruct += (args[1] << rs1_offs) & (rs1_m)
        instruct += (args[2] << rs2_offs) & (rs2_m)
        return instruct

    @staticmethod
    def decode(instruct: int) -> tuple[int, int, int, int]:
        rd = (instruct & rd_m) >> rd_offs
        rs1 = (instruct & rs1_m) >> rs1_offs
        rs2 = (instruct & rs2_m) >> rs2_offs
        imm = 0
        # opcode = (instruct & op_m)
        return rd, rs1, rs2, imm


class Immediate(Instruction):
    @staticmethod
    def code(opcode: int, args: list[int]) -> int:
        instruct = 0
        instruct += (opcode << 0) & (op_m)
        instruct += (args[0] << rd_offs) & (rd_m)
        instruct += (args[1] << rs1_offs) & (rs1_m)
        instruct += (args[2] << imm_offs) & (imm_m)
        return instruct

    @staticmethod
    def decode(instruct: int) -> tuple[int, int, int, int]:
        rd = (instruct & rd_m) >> rd_offs
        rs1 = (instruct & rs1_m) >> rs1_offs
        rs2 = 0
        imm = ((instruct & (imm_m | rs2_m)) >> imm_offs)
        # opcode = (instruct & op_m)
        return rd, rs1, rs2, imm


class Branch(Instruction):
    @staticmethod
    def code(opcode: int, args: list[int]) -> int:
        instruct = 0
        instruct += (opcode << 0) & (op_m)
        instruct += (args[0] << rs1_offs) & (rs1_m)
        instruct += (args[1] << rs2_offs) & (rs2_m)
        # add imm
        instruct += ((args[2] << imm_offs) & (rs2_m)) << (REG_SIZE * 2)
        instruct += ((args[2] << imm_offs) & (imm_m))

        return instruct

    @staticmethod
    def decode(instruct: int) -> tuple[int, int, int, int]:
        # rd = (instruct & rd_m) >> rd_offs
        rd = 0
        rs1 = (instruct & rs1_m) >> rs1_offs
        rs2 = (instruct & rs2_m) >> rs2_offs
        imm = (instruct & imm_m) >> imm_offs
        imm += ((instruct & rd_m) >> (REG_SIZE * 2)) >> imm_offs
        # opcode = (instruct & op_m)
        return rd, rs1, rs2, imm


class Jump(Instruction):
    @staticmethod
    def code(opcode: int, args: list[int]) -> int:
        instruct = 0
        instruct += (opcode << 0) & (op_m)
        instruct += (args[0] << imm_offs) & (imm_m)
        return instruct

    @staticmethod
    def decode(instruct: int) -> tuple[int, int, int, int]:
        rd = 0
        rs1 = 0
        rs2 = 0
        imm = (instruct & imm_m) >> imm_offs
        # opcode = (instruct & op_m)
        return rd, rs1, rs2, imm


class OpcodeFormat(NamedTuple):
    number: int
    instruction_type: Instruction


class Opcode(OpcodeFormat, Enum):

    HALT = OpcodeFormat(0, Jump)

    LW = OpcodeFormat(1, Register)  # A <- [B]
    SW = OpcodeFormat(2, Register)  # [A] <- B
    LWI = OpcodeFormat(3, Immediate)  # A <- [IMM]
    SWI = OpcodeFormat(4, Immediate)  # [A] <- IMM

    JMP = OpcodeFormat(5, Jump)  # unconditional transition

    BEQ = OpcodeFormat(7, Branch)  # Branch if EQual (A == B)
    BNE = OpcodeFormat(8, Branch)  # Branch if Not Equal (A != B)
    BLT = OpcodeFormat(9, Branch)  # Branch if Less Than (A < B)
    BGT = OpcodeFormat(10, Branch)  # Branch if greater then (A > B)
    BNL = OpcodeFormat(11, Branch)  # Branch if Not Less than (A >= B)
    BNG = OpcodeFormat(12, Branch)  # Branch if less or equals then (A <= B)

    ADD = OpcodeFormat(13, Register)  # t,a,b
    SUB = OpcodeFormat(14, Register)
    MUL = OpcodeFormat(15, Register)
    DIV = OpcodeFormat(16, Register)
    REM = OpcodeFormat(17, Register)

    ADDI = OpcodeFormat(18, Immediate)  # t,a,i
    MULI = OpcodeFormat(19, Immediate)
    SUBI = OpcodeFormat(20, Immediate)
    DIVI = OpcodeFormat(21, Immediate)
    REMI = OpcodeFormat(22, Immediate)


opcodes_by_number = dict([opcode.number, opcode] for opcode in Opcode)


ops_args_count = {
    "LW": 2,
    "LWI": 2,
    "SW": 2,
    "SWI": 2,

    "JMP": 1,
    "HALT": 0,

    "ADD": 3,
    "SUB": 3,
    "MUL": 3,
    "DIV": 3,
    "REM": 3,

    "ADDI": 3,
    "SUBI": 3,
    "MULI": 3,
    "DIVI": 3,
    "REMI": 3,

    "BEQ": 3,
    "BNE": 3,
    "BLT": 3,
    "BGT": 3,
    "BNL": 3,
    "BNG": 3,
}

ops_gr = {}
ops_gr["mem"] = set([
    Opcode.LW,
    Opcode.SW,
    Opcode.LWI,
    Opcode.SWI
])
ops_gr["branch"] = set([
    Opcode.BEQ,
    Opcode.BNE,
    Opcode.BLT,
    Opcode.BNL,
    Opcode.BGT,
    Opcode.BNG
])
ops_gr["imm"] = set([
    Opcode.ADDI,
    Opcode.SUBI,
    Opcode.MULI,
    Opcode.DIVI,
    Opcode.REMI,
    Opcode.LWI,
    Opcode.SWI,
])
ops_gr["arith"] = set([
    Opcode.ADDI,
    Opcode.SUBI,
    Opcode.MULI,
    Opcode.DIVI,
    Opcode.REMI,

    Opcode.ADD,
    Opcode.SUB,
    Opcode.MUL,
    Opcode.DIV,
    Opcode.REM,
])

STDIN, STDOUT = 696, 969


def normalize(code: list[dict]):
    opcodes_by_name = dict(map(lambda opcode: [opcode.name, opcode], Opcode))
    normalized_code = []
    for instr in code:
        if isinstance(instr, dict):
            opcode = opcodes_by_name[instr["opcode"]]
            normalized_instr = {"opcode": opcode.name}
            if opcode in [Opcode.SW, Opcode.SWI]:
                destination, source = instr['args']
                normalized_instr["args"] = [0, destination, source]
            elif opcode is Opcode.LW:
                destination, source = instr['args']
                normalized_instr["args"] = [destination, source, 0]
            elif opcode is Opcode.LWI:
                destination, source = instr['args']
                normalized_instr["args"] = [destination, 0, source]
            elif opcode is Opcode.HALT:
                normalized_instr["args"] = [0]
            else:
                normalized_instr["args"] = instr['args']
            normalized_instr['args'] = list(
                map(lambda arg: int(arg), normalized_instr['args']))

            normalized_code.append(normalized_instr)
        else:
            normalized_code.append(instr)
    return normalized_code


def write_bin_code(target, code):
    """Записать машинный код в bin файл."""
    program = bytearray()
    normalized = normalize(code)
    _start = 0
    # record section data
    while not isinstance(normalized[_start], dict):
        cell = int(normalized[_start])
        for _ in range(4):
            program.append(cell & 255)
            cell = cell >> 8
        _start += 1
    # record section code
    program_start = bytearray()
    program_start.append(_start & 255)
    program_start.append((_start >> 8) & 255)
    program_start.append((_start >> 16) & 16)
    program_start.append((_start >> 24) & 255)

    program = bytearray(program_start + program)
    opcodes_by_name = dict(map(lambda opcode: [opcode.name, opcode], Opcode))
    for instr in normalized[_start:]:
        opcode = opcodes_by_name[instr["opcode"]]
        args = [int(arg) for arg in instr["args"]]
        coded = opcode.instruction_type.code(opcode.number, args)
        for _ in range(4):
            program.append(coded & 255)
            coded = coded >> 8
    with open(target, "wb") as file:
        file.write(program)


def read_bin_code(target):
    """Записать машинный код в bin файл."""
    memory = []
    with open(target, "rb") as file:
        while (bytes4 := file.read(4)):
            memory.append(int.from_bytes(bytes4, "little"))
    return memory


def write_json_code(filename: str, program: list):
    """Записать машинный код в json файл."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(json.dumps(program, indent=4))


def read_json_code(filename: str) -> Tuple[dict, dict]:
    """Прочесть машинный код из json файла."""
    with open(filename, encoding="utf-8") as file:
        return json.loads(file.read())


def encode_instruct(instruct):
    ct = Instruction.decode_opcode(instruct)
    opcode = opcodes_by_number[ct]
    rd, rs1, rs2, imm = opcode.instruction_type.decode(instruct)
    return opcode, rd, rs1, rs2, imm
