section data:
divisors: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
n: 2
current_number: 0 
divisor: 0
divisor_idx: 0
result: 1
digits: 0,0,0,0,0,0,0,0,0,0

section text:
    _start:
    loop_divisor_collect:
        ; condition
        addi x2,x0,21
        lwi x1,n
        bnl x1,x2,loop_divisor_collect_end
        ; body
        lwi x1,n
        addi x2,x0,current_number
        sw x2,x1

        addi x2,x0,divisor_idx
        sw x2,x0
         
        loop_check_divisable:
        ; condition 1
        addi x2,x0,21
        lwi x1,divisor_idx
        bnl x1,x2,loop_check_divisable_end

        ; condition 2
        lwi x1,divisor_idx
        addi x2,x0,divisors
        add x1,x1,x2
        lw x2,x1
        beq x2,x0,loop_check_divisable_end
        ; body
        addi x1,x0,divisor
        lwi x2,divisor_idx
        addi x2,x2,divisors
        lw x2,x2
        sw x1,x2

        ; condition rem is 0
        lwi x1,current_number
        lwi x2,divisor
        rem x3,x1,x2
        bne x3,x0,added_divisor_end

        ; add
        addi x1,x0,current_number
        lwi x2,current_number
        lwi x3,divisor
        div x2,x2,x3
        sw x1,x2

        ; added
        added_divisor_end:

        lwi x1,divisor_idx
        addi x1,x1,1
        addi x2,x0,divisor_idx 
        sw x2,x1

        jmp loop_check_divisable

        loop_check_divisable_end:
        
        lwi x1,current_number
        addi x2,x0,1
        beq x1,x2,check_is_divisor_end
        
        addi x1,x0,divisors
        lwi x2,divisor_idx
        add x1,x1,x2
        lwi x2,current_number
        sw x1,x2

        check_is_divisor_end:

        lwi x1,n
        addi x1,x1,1
        addi x2,x0,n 
        sw x2,x1
        
        jmp loop_divisor_collect
    loop_divisor_collect_end:

    addi x1,x0,divisor_idx
    sw x1,x0

    loop_multiplex:
        ; condition 1
        addi x2,x0,21
        lwi x1,divisor_idx
        bnl x1,x2,loop_multiplex_end

        ; condition 2
        lwi x1,divisor_idx
        addi x2,x0,divisors
        add x1,x1,x2
        lw x2,x1
        beq x2,x0,loop_multiplex_end

        ; mul result
        lwi x1,divisor_idx
        addi x1,x1,divisors
        lw x1,x1
        lwi x2,result
        mul x3,x1,x2
        addi x1,x0,result
        sw x1,x3

        ; inc
        lwi x1,divisor_idx
        addi x1,x1,1
        addi x2,x0,divisor_idx 
        sw x2,x1

        jmp loop_multiplex

    loop_multiplex_end:
    write_int:
        lwi x2,result
        lwi x3,digits
        sw x3,x0
        subi x3,x3,1
        to_stack:
            beq x2,x0,to_stack_end
            remi x1,x2,10
            addi x1,x1,'0'
            sw x3,x1
            subi x3,x3,1
            divi x2,x2,10
            jmp to_stack
        to_stack_end:
        addi x2,x0,OUTPUT
        addi x3,x3,1
        from_stack:
            lw x1,x3
            beq x1,x0,from_stack_end
            sw x2,x1
            addi x3,x3,1
            jmp from_stack
        from_stack_end:

    write_int_end:
        halt