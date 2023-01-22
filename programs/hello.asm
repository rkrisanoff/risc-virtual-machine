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
