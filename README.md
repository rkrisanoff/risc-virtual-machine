# Транслятор и модель

### Вариант ###

`asm | risc  | harv   | hw | tick  | bin  | stream | mem | prob5`

| Особенность             |     |
|-------------------------|--------|
| ЯП. Синтаксис           |  синтаксис ассемблера. Необходима поддержка label-ов |
| Архитектура             | Система команд должна быть упрощенной, в духе RISC архитектур |
| Организация памяти      |  Гарвардская архитектура|
| Control Unit            | hardwired. Реализуется как часть модели.|
| Точность модели         |  процессор необходимо моделировать с точностью до такта|
| Представление маш. кода |  бинарное представление. При этом необходимо сделать экспорт в формате с мнемоникой команд для возможности анализа машинного кода. |
| Ввод-вывод              |  Ввод-вывод осуществляется как поток токенов |
| Ввод-вывод ISA          |  Memory-mapped|
| Алгоритм                | Каково наименьшее положительное число, которое  делится без остатка на все числа от 1 до 20?|

## Язык программирования

```bnf
<program> ::= 
        "section" " "+ "data:" <whitespace>* <data_section>?
        <whitespace> 
        "section" " "+ "text:" <whitespace>* <instruction_section>?
<data_section> ::= <data> (<whitespace> <data>)*
<data> ::= (<label_declaration>) " "* (<char_literal> | <number>) ("," (<char_literal> | <number>))*
<instruction_section> ::= <instruction> (<whitespace> <instruction>)*
<instruction> ::= (<label_declaration>)? " "* <letter>+ (" " (<address>  | (<reg> "," <address>) | (<reg> "," <reg> "," <address>)))? 
<address> ::= <number> | <label>
<reg> ::= "x" <number>
<label_declaration> ::= <label> ":"
<label> ::= <letter>+
<char_literal> ::= "'" (<letter> | <digit> | <whitespace>)+ "'"
<letter> ::= <lower_letter> | <upper_letter>
<lower_letter> ::= [a-z]
<upper_letter> ::= [A-Z]
<whitespace> ::= " " | "\n" | "\t"
<number> ::= <digit> | ( ([1-9]) <digit>+ )
<digit> ::= [0-9]
```

В программе должны быть две обязательные секции - для данных (может быть пустой) и для кода - section data: и section text: соответственно
Поддерживаются метки.
Метка - это символьное имя, обозначающее ячейку памяти, которая содержит некоторую команду или данные. Метка обязана начинаться с строчной или прописной буквы, кроме того, может содержать в своём имени цифры
Метки `INPUT` и `OUTPUT` зарезервированы.

В секции данных после метки следуют значения, располагающиеся последовательно по адресам, на которые указывает метка. Поддерживаются строковые литералы.
Фактически в памяти будет массив, содержащий коды символов строки.
В секции кода метка будет содержать адрес следующей после нее инструкции

Поддерживаются однострочные комментарии, начинаются с символа `;`

Код выполняется последовательно.

Пример:

```nasm
section data:
hello: 'Hello, World!',0

section text:
    _start:
        addi x2,x0,hello
        addi x3,x0,OUTPUT
    write:
        lw x1,x2
        beq x1,x0,end
        sw x3,x1
        addi x2,x2,1
        jmp write
    end:
        halt
```

### Поддерживаемые команды

- `lw  rd,rs` - загружает из памяти по адресу `rs` в регистр `rd`
- `lwi rd,imm` - загружает из памяти по адресу `imm` в регистр `rd`
- `swi rd,imm` - записывает значение `imm` в память по адресу `rd`
- `sw rd, rs` - записывает значение регистра `rs` в память по адресу `rd`
- `add rd,rs1,rs2` - складывает `rs1` и `rs2` и записывает результат в регистр `rd`
- `sub rd,rs1,rs2` - вычитает из `rs1` `rs2` и записывает результат в регистр rd
- `mul rd,rs1,rs2` - умножает `rs1` и `rs2` и записывает результат в регистр rd
- `div rd,rs1,rs2` - делит `rs1` на `rs2` и записывает результат целочисленного деления в регистр `rd`
- `rem rd,rs1,rs2` - делит `rs1` и `rs2` и записывает результат остаток от деления в регистр `rd`
- `addi rd,rs1,imm` - складывает `rs1` и `imm` и записывает результат в регистр `rd`
- `subi rd,rs1,imm` - вычитает из `rs1` `imm` и записывает результат в регистр `rd`
- `muli rd,rs1,imm` - умножает `rs1` и `imm` и записывает результат в регистр `rd`
- `divi rd,rs1,imm` - делит `rs1` на `imm` и записывает результат целочисленного деления в регистр rd
- `remi rd,rs1,imm` - делит `rs1` и `imm` и записывает результат остаток от деления в регистр `rd`
- `beq rs1,rs2,imm` - условный переход по адресу `imm`, если значения в регистрах `rs1` == `rs2`
- `bne rs1,rs2,imm` - условный переход по адресу `imm`, если значения в регистрах `rs1` != `rs2`
- `blt rs1,rs2,imm` - условный переход по адресу `imm`, если значения в регистрах `rs1` < `rs2`
- `bgt rs1,rs2,imm` - условный переход по адресу `imm`, если значения в регистрах `rs1` > `rs2`
- `bng rs1,rs2,imm` - условный переход по адресу `imm`, если значения в регистрах `rs1` <= `rs2`
- `bnl rs1,rs2,imm` - условный переход по адресу `imm`, если значения в регистрах `rs1` >= `rs2`
- `jmp imm` - безусловный переход по адресу `imm`
- `halt` - завершение программы

### Организация памяти

Работа с памятью

Модель памяти процессора:

```text
i - number of instructions
     Memory
     Instruction memory
+-----------------------------+
| 00  : instruct 1            |
| 01  : instruct 2            |
| 02  : instruct 3            |
|    ...                      |
| i   : halt                  |
+-----------------------------+

  Data memory
+------------------------------+
| 00  : variable 1             |
| 01  : variable 2             |
| 02  : array variable 1,len=l |
|    ...                       |
| l+2 : variable 3             |
| l+3 : variable 4             |
|    ...                       |
|                              |
+------------------------------+
```

## Транслятор

Реализовано в модуле [translator.py](./translator.py)

## Система команд Процессора

Особенности процессора:

- Машинное слово - 32 бита
- 4 регистра
- размер команд и типы аргументов фиксированы, имеет 4 типа
  - Register
  - Immediate
  - Branch
  - Jump
- каждая инструкция выполняется за 5 этапов (каждый по такт)
  - `fetch_instruction` - загрузка инструкции из памяти данных
  - `decode_instruction` - декодирование инструкций
  - `execute` - выполнение инструкций (вычисления в АЛУ, вычисления флагов по результату сравнения в branch comparator)
  - `memory_access` - доступ к памяти - для инструкций
  - `write_back` - запись результирующего значения (из памяти или АЛУ в регистр). На этом же этапе в инструкциях переходов переписывается значение pc'a

Модель памяти процессора:

- Память разделена на память команд и память данных.
- Машинное слово -- 32 бита, знаковое. Линейное адресное пространство.

### Регистры

Процессор в модели содержит 4 регистра общего назначения

#### Непосредственное значение

Для того, чтобы загружать значения непосредственно в DataPath существует функциональный элемент - Immediately Generator.

### Кодирование инструкций

Инструкции представляют собой 32-битные машинные слова в следующем формате

- `rd` - register destination - регистр, куда будет записано значение после выполнения инструкции
- `rs1` и `rs2` - register source 1,2 - регистры, значения которых будут использоваться для вычисления результата операции
- `imm`* - immediate - непосредственное значение
- `opcode` - номер инструкции

```ascii


  31        30   29   28   27      26   25       5   4    0    Bits
+-----------------------------------------------------------+
|      rd      |   rs1   |    rs2     |            | opcode | Register type
+-----------------------------------------------------------+
|      rd      |   rs1   | imm[22:21] | imm[20:0]  | opcode | Immediate type
+-----------------------------------------------------------+
|  imm[22:21]  |   rs1   |    rs2     | imm[20:0]  | opcode | Branch type
+-----------------------------------------------------------+
|                       imm                        | opcode | Jump type
+-----------------------------------------------------------+
```

\* imm - имеет переменный размер, извлекается из инструкции  в immediate generator

### Набор инструкции

Совпадают с [поддерживаемым командами](#поддерживаемые-команды)

| Инструкция | Тип инструкции |
|:-----------------|--------|
| ADD | Register |
| SUB | Register |
| MUL | Register |
| DIV | Register |
| REM | Register |
| ADDI | Immediate |
| SUBI | Immediate |
| MULI | Immediate |
| DIVI | Immediate |
| REMI | Immediate |
| BEQ | Branch |
| BNE | Branch |
| BLT | Branch |
| BGT | Branch |
| BNL | Branch |
| BNG | Branch |

#### Struct

Так же для

- Машинный код сериализуется в список JSON.
- Один элемент списка, одна инструкция (так как в risc размер инструкций фиксированный).
- Индекс списка - адрес инструкции. Используется для команд перехода.

Пример:

```json
[
    {
        "opcode": "ADD",
        "args": [
            "arg1",
            "arg2",
            "arg3"
        ]
    }
]
```

где:

- `opcode` -- строка с кодом операции;
- `args` -- список аргументов (может отсутствовать (только в случае HALT));

### Модель процессора

Реализовано в модуле [machine.py](./machine.py)

#### DataPath & ControlUnit

![Scheme](Scheme.png)

#### ControlUnit

Реализован в классе `ControlUnit`.

- Hardwired (реализовано полностью на python).
- Моделирование на уровне тактов.
- Трансляция инструкции в последовательность (5 тактов) сигналов: `tick_by_tick`.

Особенности работы модели:

- Для журнала состояний процессора используется стандартный модуль logging.
- Количество инструкций для моделирования ограничено hardcoded константой.
- Остановка моделирования осуществляется при помощи исключений:
  - `EOFError` -- если нет данных для чтения из порта ввода-вывода;
  - `StopIteration` -- если выполнена инструкция `halt`.
- Управление симуляцией реализовано в функции `simulate`.

## Апробация

| ФИО              | алг.  | LoC       | code байт|code инстр. | инстр. | такт. |
|------------------|-------|---------|---------|---------|--------|------|
| Сагайдак Алина Алексеевна |cat| 46| 15 | 188  | 240     | 1201     |
|  |hello| 25| 8 | 92  | 69    | 346    |
|  |prob5| 242 | 89 | 496 | 3778 | 18891|

где:

- алг. -- название алгоритма (hello, cat, или как в варианте)
- LoC -- кол-во строк кода в реализации алгоритма
- code байт -- кол-во байт в машинном коде (если бинарное представление)
- code инстр. -- кол-во инструкций в машинном коде
- инстр. -- кол-во инструкций, выполненных при работе алгоритма
- такт. -- кол-во тактов, которое заняла работа алгоритма

1. [hello world](./programs/hello_world.asm) - выводит `Hello, world!` в stdin.
2. [cat](./programs/cat.asm) - программа `cat`, повторяем ввод на выводе.
3. [prob5](./programs/prob5.asm) - problem 5

Интеграционные тесты реализованы тут: [integration_test](./integration_test.py)

CI:

``` yaml
lab3:
  stage: test
  image:
    name: python-tools
    entrypoint: [""]
  script:
    - python3-coverage run -m pytest --verbose
    - find . -type f -name "*.py" | xargs -t python3-coverage report
    - find . -type f -name "*.py" | xargs -t pycodestyle --ignore=E501,W291
    - find . -type f -name "*.py" | xargs -t pylint

```

где:

- `python3-coverage` -- формирование отчёта об уровне покрытия исходного кода.
- `pytest` -- утилита для запуска тестов.
- `pycodestyle` -- утилита для проверки форматирования кода. `E501` (длина строк)
- `pylint` -- утилита для проверки качества кода. Некоторые правила отключены в отдельных модулях с целью упрощения кода.
- Docker image `python-tools` включает в себя все перечисленные утилиты. Его конфигурация: [Dockerfile](./Dockerfile).

Пример использования и журнал работы процессора на примере `prob5`:

```bash
> python3 translator.py programs/prob5.json programs/prob5.bin
source LoC: 242 code instr: 112 bytes: 452

```

[prob5.json](programs/prob5.bin)
[prob5.bin](programs/prob5.bin)

```python

> ./machine.py programs/prob5.bin programs/input.txt



```
