section data:
buffer: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

section text:
    _start:
        addi x2,x0,buffer
        addi x3,x0,INPUT
    read:
        lw x1,x3
        sw x2,x1
        beq x1,x0,finish_read
        addi x2,x2,1
        jmp read
    finish_read:
        addi x2,x0,buffer
        addi x3,x0,OUTPUT

    write:
        lw x1,x2
        sw x3,x1
        beq x1,x0,end
        addi x2,x2,1
        jmp write
    end:
        halt