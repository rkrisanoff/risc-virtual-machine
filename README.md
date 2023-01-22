# Транслятор и модель

### Вариант ###

`asm | risc  | neum   | hw | tick  | bin  | stream | mem | prob5`

| Особенность             |     |
|-------------------------|--------|
| ЯП. Синтаксис           |  синтаксис ассемблера. Необходима поддержка label-ов |
| Архитектура             | Система команд должна быть упрощенной, в духе RISC архитектур |
| Организация памяти      |  фон Неймановская архитектура |
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

- `lw  rd,rs` - загружает из памяти по адресу rs в регистр rd
- `lwi rd,imm` - загружает из памяти по адресу imm в регистр rd
- `swi rd,imm` - записывает значение imm в память по адресу rd
- `sw rd, rs` - записывает значение регистра rs в память по адресу rd
- `add rd,rs1,rs2` - складывает rs1 и rs2 и записывает результат в регистр rd
- `sub rd,rs1,rs2` - вычитает из rs1 rs2 и записывает результат в регистр rd
- `mul rd,rs1,rs2` - умножает rs1 и rs2 и записывает результат в регистр rd
- `div rd,rs1,rs2` - делит rs1 на rs2 и записывает результат целочисленного деления в регистр rd
- `rem rd,rs1,rs2` - делит rs1 и rs2 и записывает результат остаток от деления в регистр rd
- `addi rd,rs1,imm` - складывает rs1 и imm и записывает результат в регистр rd
- `subi rd,rs1,imm` - вычитает из rs1 imm и записывает результат в регистр rd
- `muli rd,rs1,imm` - умножает rs1 и imm и записывает результат в регистр rd
- `divi rd,rs1,imm` - делит rs1 на imm и записывает результат целочисленного деления в регистр rd
- `remi rd,rs1,imm` - делит rs1 и imm и записывает результат остаток от деления в регистр rd
- `beq rs1,rs2,imm` - условный переход по адресу imm, если значения в регистрах rs1 == rs2
- `bne rs1,rs2,imm` - условный переход по адресу imm, если значения в регистрах rs1 != rs2
- `blt rs1,rs2,imm` - условный переход по адресу imm, если значения в регистрах rs1 < rs2
- `bgt rs1,rs2,imm` - условный переход по адресу imm, если значения в регистрах rs1 > rs2
- `bng rs1,rs2,imm` - условный переход по адресу imm, если значения в регистрах rs1 <= rs2
- `bnl rs1,rs2,imm` - условный переход по адресу imm, если значения в регистрах rs1 >= rs2
- `jmp imm` - безусловный переход по адресу imm
- `halt` - завершение программы

### Организация памяти

Работа с памятью

Модель памяти процессора:

```text
s - size of memory
n - number of variables
d = size of data
i - number of instructions
     Memory
+-----------------------------------+
| 00      : variable 1              |
| 01      : variable 2              |
| 02      : array variable 3,len=l  |
|    ...                            |
| l+2 : variable 4                  |
|    ...                            |
| d-2     : variable n-1            | 
| d-1     : variable n              |
| d       : instruct 1              |
| d+1     : instruct 2              |
| d+2     : instruct 3              |
|    ...                            | 
| d+i-1   : halt (instruct i)       |
|    ...                            |
| s-1     : top of the stack        |
+-----------------------------------+
```

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
  - fetch_instruction
  - decode_instruction
  - execute
  - memory_access
  - write_back

Модель памяти процессора:

- Память для команд и данных общая.
- Машинное слово -- 32 бита, знаковое. Линейное адресное пространство.

### Регистры

Процессор в модели содержит 4 регистров

#### Непосредственное значение

Для того, чтобы загружать значения непосредственно в DataPath существует функциональный элемент - Immediately Generator, который загружает непосредственно (в коде) указанные значения в АЛУ.
Команды, которые, которые используют такое значение в качестве ***последнего*** операнда, имеют в названии постфикс `I`

### Кодирование инструкций

Инструкции представляют собой 32-битные машинные слова в следующем формате

- rd - register destination - регистр, куда будет записано значение после выполнения инструкции
- rs1 и rs2 - register source 1,2 - регистры, значения которых будут использоваться для вычисления результата операции
- imm - immediate - непосредственное значение
- opcode - номер инструкции

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

### Набор инструкции

| Синтаксис |  Кол-во тактов | Комментарий                          |
|:-------|--------|:---------------------------------|
| `HALT`    |  0  | Останавливает выполнение программы|

#### Команды, работющие с памятью

Команды, отвечающие за запись и загрузку данных из памяти

| Синтаксис | операнд1 | операнд2 | Кол-во тактов | Тип инструкции* |
|:-------|:-------------|------|------|----|
| `LW`    |  регистр, куда будет записано значение | регистр, содержащий адрес  памяти, откуда в целевой будет записано значение   | 3 |   Register                    |
| `LWI`|регистр, куда будет записано значение|непосредственно указанный адрес памяти, откуда будет извлечено значение |3 | Immediate|
| `SW`    |  регистр, содержащий адрес, куда будет записано занчение            | регистр, содержащий адрес  памяти                         |2| Register|
| `SWI`    |  регистр, содержащий адрес, куда будет записано занчение            | Непосредственно указанный адрес памяти                         |2| Immediate|

#### Команды переходов

Команды, реализующие условные переходы. Принаджежат* к типу Branch-инструкций.
Имеют три аргумента - первый и второй операнды, участвующие в сравнении, а третий содержит абсолютный адрес перехода

| Синтаксис | Условие перехода   | Кол-во тактов |
|:-------|:-------------|---|
| `BEQ`    | операнды равны    | 3|
| `BNE`    | операнды не равны    | 3 |
| `BLT` | первый операнд меньше второго         |3 |
| `BNL` | первый операнд меньше или равен второму         |3
| `BGT` | первый операнд больше  второго         | 3|
| `JMP`    | всегда | 2 |

\*кроме JUMP

#### Арифметическо-логические

Команды, реализующие арифметические операции. В качестве операндом используют 3 аргумента.
Принаджежат к типу Register-инструкций.

| Синтаксис | Арифметическая операция | Кол-во тактов |
|:-------|:-------------|:-------|
|`ADD`|сложение |3|
|`SUB`|вычитание|3|
|`MUL`|умножене|3|
|`DIV`|целочисленное деление|3|
|`REM`|отстаток от целочисленного деления|3|

Те же операции, только в качестве третьего аргумента принимающие непосредственные значения.
Принаджежат к типу Immediate-инструкций.

| Синтаксис | Арифметическая операция | Кол-во тактов |
|:-------|:-------------|:-------|
|`ADDI`|сложение|3|
|`SUBI`|вычитание|3|
|`MULI`|умножене |3|
|`DIVI`|деление|3|
|`REMI`|отстаток от целочисленного деления|3|

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

## Транслятор

### Модель процессора

Реализовано в модуле [machine.py](./machine.py)

#### DataPath & ControlUnit

![Scheme](Scheme.png)

#### ControlUnit

Реализован в классе `ControlUnit`.

- Hardwired (реализовано полностью на python).
- Моделирование на уровне тактов.
- Трансляция инструкции в последовательность (5 тактов) сигналов: `tick_by_tick`.
- 

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
| Алина Сагайдак |cat| 46| 46 | 188  | 210     | 509     |
| Алина Сагайдак |hello| 33| 23 | 96  | 69    | 169    |
| Алина Сагайдак |prob5| 242 | 112 | 452 | 3777 | 10399|

В качестве тестов использовано два алгоритма:

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
- `pycodestyle` -- утилита для проверки форматирования кода. `E501` (длина строк) и `W291` отключены.
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
