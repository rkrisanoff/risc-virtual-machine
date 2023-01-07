section data:
buffer: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

section text:
    _start:
        addi 2,0,buffer
        addi 3,0,696 # STDIN
    read:
        lw 1,3
        sw 2,1
        beq 1,0,finish_read
        addi 2,2,1
        jmp read
    finish_read:
        addi 2,0,buffer
        addi 3,0,969 # STDOUT

    write:
        lw 1,2
        sw 3,1
        beq 1,0,end # f
        addi 2,2,1
        jmp write
    end:
        halt