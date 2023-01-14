# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

from abc import ABC, abstractmethod

from enum import Enum
import json
from typing import NamedTuple, Tuple, Union

OPCODE_SIZE = 5
REG_SIZE = 3
IMM_SIZE = 18
op_m, op_offs = 0b000_000_000_000000000000000000_11111, 0
rd_m = 0b111_000_000_0000000000000000_0000000
rd_offs = REG_SIZE * 2 + IMM_SIZE + OPCODE_SIZE
rs1_m, rs1_offs = 0b000_111_000_000000000000000000_00000, REG_SIZE + \
    IMM_SIZE + OPCODE_SIZE
rs2_m, rs2_offs = 0b000_000_111_000000000000000000_00000, IMM_SIZE + OPCODE_SIZE
imm_m_j, imm_j_offs = 0b111_111_111_111111111111111111_00000, OPCODE_SIZE
imm_m_l, imm_l_offs = 0b000_111_111_111111111111111111_00000, OPCODE_SIZE
imm_m, imm_offs = 0b000_000_000_111111111111111111_00000, OPCODE_SIZE


class Instruction(ABC):
    @staticmethod
    @abstractmethod
    def encode(opcode: int, args: list[int]) -> int:
        pass

    @staticmethod
    @abstractmethod
    def decode(instruct: int) -> tuple[int, int, int, int]:
        pass

    @staticmethod
    def decode_opcode(instr: int) -> int:
        return instr & op_m


class Register(Instruction):
    @staticmethod
    def encode(opcode: int, args: list[int]) -> int:
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
        return rd, rs1, rs2, imm


class Immediate(Instruction):
    @staticmethod
    def encode(opcode: int, args: list[int]) -> int:
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
        return rd, rs1, rs2, imm


class Branch(Instruction):
    @staticmethod
    def encode(opcode: int, args: list[int]) -> int:
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
        rd = 0
        rs1 = (instruct & rs1_m) >> rs1_offs
        rs2 = (instruct & rs2_m) >> rs2_offs
        imm = (instruct & imm_m) >> imm_offs
        imm += ((instruct & rd_m) >> (REG_SIZE * 2)) >> imm_offs
        return rd, rs1, rs2, imm


class Jump(Instruction):
    @staticmethod
    def encode(opcode: int, args: list[int]) -> int:
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
        return rd, rs1, rs2, imm


class OpcodeFormat(NamedTuple):
    number: int
    instruction_type: Instruction


class Opcode(OpcodeFormat, Enum):

    HALT = OpcodeFormat(number=0, instruction_type=Jump)

    LW = OpcodeFormat(number=1, instruction_type=Register)  # A <- [B]
    SW = OpcodeFormat(number=2, instruction_type=Register)  # [A] <- B
    LWI = OpcodeFormat(number=3, instruction_type=Immediate)  # A <- [IMM]
    SWI = OpcodeFormat(number=4, instruction_type=Immediate)  # [A] <- IMM

    # unconditional transition
    JMP = OpcodeFormat(number=5, instruction_type=Jump)

    # Branch if EQual (A == B)
    BEQ = OpcodeFormat(number=7, instruction_type=Branch)
    # Branch if Not Equal (A != B)
    BNE = OpcodeFormat(number=8, instruction_type=Branch)
    # Branch if Less Than (A < B)
    BLT = OpcodeFormat(number=9, instruction_type=Branch)
    # Branch if greater then (A > B)
    BGT = OpcodeFormat(number=10, instruction_type=Branch)
    # Branch if Not Less than (A >= B)
    BNL = OpcodeFormat(number=11, instruction_type=Branch)
    # Branch if less or equals then (A <= B)
    BNG = OpcodeFormat(number=12, instruction_type=Branch)

    ADD = OpcodeFormat(number=13, instruction_type=Register)  # t,a,b
    SUB = OpcodeFormat(number=14, instruction_type=Register)
    MUL = OpcodeFormat(number=15, instruction_type=Register)
    DIV = OpcodeFormat(number=16, instruction_type=Register)
    REM = OpcodeFormat(number=17, instruction_type=Register)

    ADDI = OpcodeFormat(number=18, instruction_type=Immediate)  # t,a,i
    MULI = OpcodeFormat(number=19, instruction_type=Immediate)
    SUBI = OpcodeFormat(number=20, instruction_type=Immediate)
    DIVI = OpcodeFormat(number=21, instruction_type=Immediate)
    REMI = OpcodeFormat(number=22, instruction_type=Immediate)


opcodes_by_number = dict((opcode.number, opcode) for opcode in Opcode)
opcodes_by_name = dict(map(lambda opcode: (opcode.name, opcode), Opcode))

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

STDIN, STDOUT = 696, 969


def normalize(code: list[dict]):
    normalized_code = []
    for instr in code:
        if isinstance(instr, dict):
            opcode = opcodes_by_name[instr["opcode"]]
            normalized_instr: dict[str, Union[str, list[int]]] = {
                "opcode": opcode.name}
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
                map(int, normalized_instr['args']))

            normalized_code.append(normalized_instr)
        else:
            normalized_code.append(instr)
    return normalized_code


def encode_instr(instr):
    bin_instr = bytearray()
    opcode = opcodes_by_name[instr["opcode"]]
    args = [int(arg) for arg in instr["args"]]
    coded = opcode.instruction_type.encode(opcode.number, args)
    for _ in range(4):
        bin_instr.append(coded & 255)
        coded = coded >> 8
    return bytes(bin_instr)


def decode_instr(instruct: int):
    ct = Instruction.decode_opcode(instruct)
    opcode = opcodes_by_number[ct]
    rd, rs1, rs2, imm = opcode.instruction_type.decode(instruct)
    return opcode, rd, rs1, rs2, imm


def format_instr(instr):
    opcode, rd, rs1, rs2, imm = decode_instr(instr)
    if opcode.instruction_type is Register:
        return f"{opcode.name} {rd}, {rs1}, {rs2}"
    if opcode.instruction_type is Immediate:
        return f"{opcode.name} {rd}, {rs1}, {imm}"
    if opcode.instruction_type is Branch:
        return f"{opcode.name} {rs1}, {rs2}, {imm}"
    if opcode.instruction_type is Jump:
        return f"{opcode.name} {imm}"
    return ""


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
    for instr in normalized[_start:]:
        program.extend(encode_instr(instr))
    with open(target, "wb") as file:
        file.write(program)

    return len(program)


def read_bin_code(target):
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
