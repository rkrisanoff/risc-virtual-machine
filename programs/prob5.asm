section data:
divisors: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
n: 2
current_number: 0 
divisor: 0
divisor_idx: 0
result: 1

section text:
    _start:
    loop_divisor_collect:
        # condition
        addi 2,0,21
        lwi 1,n
        bnl 1,2,loop_divisor_collect_end
        # body
        lwi 1,n
        addi 2,0,current_number
        sw 2,1

        addi 2,0,divisor_idx
        sw 2,0
         
        loop_check_divisable:
        # condition 1
        addi 2,0,21
        lwi 1,divisor_idx
        bnl 1,2,loop_check_divisable_end

        # condition 2
        lwi 1,divisor_idx
        addi 2,0,divisors
        add 1,1,2
        lw 2,1
        beq 2,0,loop_check_divisable_end
        # body
        addi 1,0,divisor
        lwi 2,divisor_idx
        addi 2,2,divisors
        lw 2,2
        sw 1,2

        # condition rem is 0
        lwi 1,current_number
        lwi 2,divisor
        rem 3,1,2
        bne 3,0,added_divisor_end

        # add
        addi 1,0,current_number
        lwi 2,current_number
        lwi 3,divisor
        div 2,2,3
        sw 1,2

        # added
        added_divisor_end:

        lwi 1,divisor_idx
        addi 1,1,1
        addi 2,0,divisor_idx 
        sw 2,1

        jmp loop_check_divisable

        loop_check_divisable_end:
        
        lwi 1,current_number
        addi 2,0,1
        beq 1,2,check_is_divisor_end
        
        addi 1,0,divisors
        lwi 2,divisor_idx
        add 1,1,2
        lwi 2,current_number
        sw 1,2

        check_is_divisor_end:

        lwi 1,n
        addi 1,1,1
        addi 2,0,n 
        sw 2,1
        
        jmp loop_divisor_collect
    loop_divisor_collect_end:

    addi 1,0,divisor_idx
    sw 1,0

    loop_multiplex:
        # condition 1
        addi 2,0,21
        lwi 1,divisor_idx
        bnl 1,2,loop_multiplex_end

        # condition 2
        lwi 1,divisor_idx
        addi 2,0,divisors
        add 1,1,2
        lw 2,1
        beq 2,0,loop_multiplex_end

        # mul result
        lwi 1,divisor_idx
        addi 1,1,divisors
        lw 1,1
        lwi 2,result
        mul 3,1,2
        addi 1,0,result
        sw 1,3

        # inc
        lwi 1,divisor_idx
        addi 1,1,1
        addi 2,0,divisor_idx 
        sw 2,1

        jmp loop_multiplex

    loop_multiplex_end:
    write_int:
        lwi 2,result
        sw 4,0
        subi 4,4,1
        to_stack:
            beq 2,0,to_stack_end
            remi 1,2,10
            addi 1,1,48 # ord('0') = 48
            sw 4,1
            subi 4,4,1
            divi 2,2,10
            jmp to_stack
        to_stack_end:
        addi 3,0,969
        addi 4,4,1
        from_stack:
            lw 1,4
            beq 1,0,from_stack_end
            sw 3,1
            addi 4,4,1
            jmp from_stack
        from_stack_end:

    write_int_end:
        halt