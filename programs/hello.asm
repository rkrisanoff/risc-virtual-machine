section data:
hello: 'Hello, World!',0
lenght: 13

section text:
    _start:
        addi 2,0,hello
        addi 3,0,969 # 969 - OUTPUT
    write:
        lw 1,2
        beq 1,0,end # f
        sw 3,1
        addi 2,2,1
        jmp write
    end:
        halt
