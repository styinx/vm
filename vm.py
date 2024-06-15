import sys
from math import log10


KB = 1024
MB = KB * KB


class Operator:
    def __class_getitem__(cls, val) -> str:
        assert val < cls.Max , f'{val} >= {cls.Max}'
        for k, v in cls.__dict__.items():
            if v == val:
                return k
        return ''

    Max = 23
    Operators = \
        Nop, Exit, \
        PushR, PushI, Pop, \
        LoadRI, StoreRI, \
        MoveRR, MoveRI, \
        AddRR, AddRI, \
        SubRR, SubRI, \
        CmpRR, CmpRI, \
        JmpR, JmpI, \
        JmpLR, JmpLI, \
        JmpGR, JmpGI, \
        JmpLRR, JmpLRI, \
        = range(Max)



class Cpu:
    class RegisterFlag:
        Max = 4
        Flags = Carry, Parity, Zero, Sign, = range(Max)

    class Register:
        def __class_getitem__(cls, val) -> str:
            assert val < cls.Max , f'{val} >= {cls.Max}'
            for k, v in cls.__dict__.items():
                if v == val:
                    return k
            return ''
        Max = 3
        SP_Registers = RF, SP, PC = range(Max)

    def __init__(self, bits :int, num_registers: int = 16) -> None:
        self.bits = bits
        self.bytes = bits // 8
        self.registers = [0] * (len(Cpu.Register.SP_Registers) + num_registers)

        Cpu.Register.Max = len(Cpu.Register.SP_Registers) + num_registers
        for i in range(num_registers):
            setattr(Cpu.Register, f'R{i}', len(Cpu.Register.SP_Registers) + i)

    def __setitem__(self, register: int, value: int):
        assert register < len(self.registers) , f'{register} >= {len(self.registers)}'
        self.registers[register] = value

    def __getitem__(self, register: int) -> int:
        assert register < len(self.registers) , f'{register} >= {len(self.registers)}'
        return self.registers[register]

    def __str__(self) -> str:
        reg_digits = int(log10(len(self.registers))) + 2
        entry_len = max(self.bytes, reg_digits) + 1
        sp_registers = len(Cpu.Register.SP_Registers)
        gp_registers = len(self.registers) - sp_registers
        separator = f'{("+" + "-" * (self.bytes + 4)) * len(self.registers)}+\n'
        regs = separator
        vals = separator
        for i in range(sp_registers + gp_registers):
            if i < sp_registers:
                regs += '| {:<{len}s} '.format(Cpu.Register[i], len=entry_len)
            else:
                regs += '| {:<{len}s} '.format(f'R{i - sp_registers}', len=entry_len)
            vals += '| {:<{len}s} '.format(f'#{self.registers[i]:0{self.bytes}x}', len=entry_len)
        return '\nCPU:\n' + regs + '|\n' + vals + '|\n' + separator

    def __repr__(self) -> str:
        return str(self)


class Memory:
    def __init__(self, bits: int, size: int) -> None:
        self.size = size
        self.bits = bits
        self.bytes = bits // 8
        self.memory = [0] * size
        self.text = 0x0
        self.data = 0x0
        self.bss = 0x0
        self.rw = 0x0

        self.data = int(len(self.memory) * 0.5) # 50% text
        self.bss = int(len(self.memory) * 0.6)  # 10% data
        self.rw = int(len(self.memory) * 0.7)   # 10% bss,  30% rw

    def load(self, program: list):
        assert len(program) < self.data, f'Program too big {len(program)} > {self.data}'
        self.memory[0:len(program)] = program

    def __setitem__(self, address: int, value: int):
        assert address >= self.rw, f'Attempt to write protected memory 0x{address:x} < 0x{self.rw:x}'
        self.memory[address] = value

    def __getitem__(self, address: int) -> int:
        return self.memory[address]

    def __str__(self) -> str:
        val_digits = self.bits // 8
        mem_digits = int(log10(len(self.memory))) + 1
        cols = {8: 80, 16: 64, 32: 32, 64: 16}[self.bits]
        s = ' ' * (mem_digits + 5)
        for j in range(cols):
            s += f'{j:0{val_digits}x} '
        s += f'\n{" " * (mem_digits + 3)}+{"-" * ((val_digits + 1) * cols)}'
        i = 0
        while i < len(self.memory):
            if i % cols == 0:
                s += f'\n0x{i:0{mem_digits}x} | '
            s += f'{self.memory[i]:0{val_digits}x} '
            i += 1
        return f'\nMemory: ({self.size} B, {self.size // 1024} KB)\n\n' + s

    def __repr__(self) -> str:
        return str(self)


class VM:
    def __init__(self, bits: int, num_registers: int, memory_size: int) -> None:
        self.registers = [0] * num_registers
        self.memory = Memory(bits, memory_size)
        self.cpu = Cpu(bits, num_registers)
        self.stack = []
        self.bits = bits
        self.bytes = bits // 8

    def __bool__(self) -> bool:
        return self.cpu[Cpu.Register.PC] < self.memory.size

    def __str__(self) -> str:
        return str(self.cpu) + '\n' + str(self.memory) + '\n'

    def __repr__(self) -> str:
        return str(self)

    def _print_op(self, op: int):
        print(f'{Operator[op]:8}', end=' ')

    def _print_reg_name(self, reg: int):
        print(f'{Cpu.Register[reg]:{self.bytes * 2 + 1}s}', end=' ')

    def _print_reg_value(self, reg: int):
        self._print_val(self.cpu[reg])

    def _print_val(self, val: int):
        print(f'#{val:0{self.bytes * 2}x}', end=' ')

    def _print_fill(self):
        print(f'{" " * (self.bytes * 2)}', end=' ')

    def _print_bits(self, val: int):
        print(f'{val:{self.bytes * 2 + 1}b}', end=' ')

    def _set_flag(self, flag: int):
        assert flag < Cpu.RegisterFlag.Max, f'{flag} >= {Cpu.RegisterFlag.Max}'
        self.cpu[Cpu.Register.RF] |= (1 << flag)

    def _clear_flag(self, flag: int):
        assert flag < Cpu.RegisterFlag.Max, f'{flag} >= {Cpu.RegisterFlag.Max}'
        self.cpu[Cpu.Register.RF] &= ~(1 << flag)

    def _is_flag_set(self, flag: int) -> bool:
        assert flag < Cpu.RegisterFlag.Max, f'{flag} >= {Cpu.RegisterFlag.Max}'
        is_set = self.cpu[Cpu.Register.RF] & (1 << flag)
        self._clear_flag(flag)
        return is_set

    def int(self, number: int) -> int:
        return number.to_bytes(length=self.bytes, byteorder=sys.byteorder)

    def instruction(self, *args):
        b = []
        for arg in args:
            if isinstance(arg, bytes):
                b += arg
            else:
                b += arg.to_bytes(length=1, byteorder=sys.byteorder)
        return b

    def next(self, bytes: int = 1):
        pc = self.cpu[Cpu.Register.PC]

        if bytes > 1:
            val = int.from_bytes(self.memory.memory[pc : pc + bytes], byteorder=sys.byteorder)
        else:
            val = self.memory.memory[pc]

        self.cpu[Cpu.Register.PC] += bytes

        return val

    def execute(self):
        try:

            while self:
                op = self.next()
                self._print_op(op)

                # No operation
                if op == Operator.Nop:
                    pass

                # Stop VM
                elif op == Operator.Exit:
                    print()
                    return

                # Pushes the value stored in the given register onto the stack
                elif op == Operator.PushR:
                    src = self.next()
                    self.stack.append(self.cpu[src])
                    self._print_reg_name(src)

                # Pushes the value onto the stack
                elif op == Operator.PushI:
                    val = self.next(self.bytes)
                    self.stack.append(val)
                    self._print_val(val)

                # Pops the top most element from the stack into the given register
                elif op == Operator.Pop:
                    dst = self.next()
                    self.cpu[dst] = self.stack.pop()
                    self._print_reg_name(dst)

                # Load value at given address into register
                elif op == Operator.LoadRI:
                    dst = self.next()
                    adr = self.next(self.bytes)
                    val = self.memory[adr]
                    self.cpu[dst] = val
                    self._print_reg_name(dst)
                    self._print_val(adr)

                # Store register value at given address
                elif op == Operator.StoreRI:
                    dst = self.next()
                    adr = self.next(self.bytes)
                    val = self.cpu[dst]
                    self.memory[adr] = val
                    self._print_reg_name(dst)
                    self._print_val(val)

                # Copy value between registers
                elif op == Operator.MoveRR:
                    dst = self.next()
                    src = self.next()
                    self.cpu[dst] = self.cpu[src]
                    self._print_reg_name(dst)
                    self._print_reg_name(src)

                # Copy value into register
                elif op == Operator.MoveRI:
                    dst = self.next()
                    val = self.next(self.bytes)
                    self.cpu[dst] = val
                    self._print_reg_name(dst)
                    self._print_val(val)

                # Add register to register
                elif op == Operator.AddRR:
                    dst = self.next()
                    src = self.next()
                    self.cpu[dst] += self.cpu[src]
                    self._print_reg_name(dst)
                    self._print_reg_name(src)

                # Add value to register
                elif op == Operator.AddRI:
                    dst = self.next()
                    val = self.next(self.bytes)
                    self.cpu[dst] += val
                    self._print_reg_name(dst)
                    self._print_val(val)

                # Subtract register to register
                elif op == Operator.SubRR:
                    dst = self.next()
                    src = self.next()
                    self.cpu[dst] -= self.cpu[src]
                    self._print_reg_name(dst)
                    self._print_reg_name(src)

                # Subtract value from register
                elif op == Operator.SubRI:
                    dst = self.next()
                    val = self.next(self.bytes)
                    self.cpu[dst] -= val
                    self._print_reg_name(dst)
                    self._print_val(val)

                # Compare 2 registers against each other.
                elif op == Operator.CmpRR:
                    src1 = self.next()
                    src2 = self.next()
                    if self.cpu[src1] < self.cpu[src2]:
                        self._set_flag(Cpu.RegisterFlag.Sign)
                    elif self.cpu[src1] > self.cpu[src2]:
                        self._set_flag(Cpu.RegisterFlag.Carry)
                    self._print_reg_name(src1)
                    self._print_reg_name(src2)

                # Compare a register against a number.
                elif op == Operator.CmpRI:
                    src = self.next()
                    val = self.next(self.bytes)
                    if self.cpu[src] < val:
                        self._set_flag(Cpu.RegisterFlag.Sign)
                    elif self.cpu[src] > val:
                        self._set_flag(Cpu.RegisterFlag.Carry)
                    self._print_reg_name(src)
                    self._print_val(val)

                # Jumps to the address given by the register
                elif op == Operator.JmpR:
                    src = self.next()
                    self.cpu[Cpu.Register.PC] = self.cpu[src]
                    self._print_reg_name(src)

                # Jumps to the given address
                elif op == Operator.JmpI:
                    val = self.next(self.bytes)
                    self.cpu[Cpu.Register.PC] = val
                    self._print_val(val)

                # Jumps to the given address if the carry flag is 1.
                elif op == Operator.JmpLR:
                    src = self.next()
                    adr = self.cpu[src]
                    val = self.memory[adr]
                    if self._is_flag_set(Cpu.RegisterFlag.Sign):
                        self.cpu[Cpu.Register.PC] = val
                    self._print_reg_name(src)

                # Jumps to the given address if the carry flag is 1.
                elif op == Operator.JmpLI:
                    adr = self.next(self.bytes)
                    val = self.memory[adr]
                    if self._is_flag_set(Cpu.RegisterFlag.Sign):
                        self.cpu[Cpu.Register.PC] = val
                    self._print_val(val)
                    self._print_fill()
                    self._print_bits(self.cpu[Cpu.Register.RF])

                # Jumps to the given address if the carry flag is 0.
                elif op == Operator.JmpGR:
                    src = self.next()
                    adr = self.cpu[src]
                    val = self.memory[adr]
                    if self._is_flag_set(Cpu.RegisterFlag.Carry):
                        self.cpu[Cpu.Register.PC] = val
                    self._print_reg_name(src)

                # Jumps to the given address if the carry flag is 0.
                elif op == Operator.JmpGI:
                    adr = self.next(self.bytes)
                    val = self.memory[adr]
                    if self._is_flag_set(Cpu.RegisterFlag.Carry):
                        self.cpu[Cpu.Register.PC] = val
                    self._print_val(val)

                else:
                    print(f'Unknown Operator: {op} (at PC: {self.cpu[Cpu.Register.PC] - 1})')

                print()

        except StopIteration:
            pass


vm = VM(bits=16, num_registers=16, memory_size=2*KB)

instructions = [
    vm.instruction(Operator.Nop),
    vm.instruction(Operator.AddRI,      Cpu.Register.R8,    vm.int(0x1)), # Increment R8
    vm.instruction(Operator.CmpRI,      Cpu.Register.R8,    vm.int(0x8)), # while R8 < 8
    vm.instruction(Operator.JmpLI,      vm.int(0x0)),                     # Otherwise Jump to start
    vm.instruction(Operator.Nop),
    vm.instruction(Operator.MoveRI,     Cpu.Register.R0,    vm.int(0xff)),
    vm.instruction(Operator.MoveRR,     Cpu.Register.R1,    Cpu.Register.R0),
    vm.instruction(Operator.MoveRI,     Cpu.Register.R1,    vm.int(0x12)),
    vm.instruction(Operator.StoreRI,    Cpu.Register.R1,    vm.int(0x0600)),
    vm.instruction(Operator.LoadRI,     Cpu.Register.R2,    vm.int(0xfa)),
    vm.instruction(Operator.AddRI,      Cpu.Register.R3,    vm.int(0x5)),
    vm.instruction(Operator.AddRI,      Cpu.Register.R3,    vm.int(0x4)),
    vm.instruction(Operator.AddRR,      Cpu.Register.R4,    Cpu.Register.R3),
    vm.instruction(Operator.MoveRI,     Cpu.Register.R5,    vm.int(0x0)),
    vm.instruction(Operator.MoveRI,     Cpu.Register.R10,   vm.int(0xa)),
    # Overwrite the next operand (Exit) with a zero to prevent the VM from exiting.
    # Not possible since the .text memory location is write protected.
    # vm.make_instruction(Operator.StoreRI, Cpu.Register.R5, vm.int(0x27)),
    vm.instruction(Operator.Exit)
]

program = [b for i in instructions for b in i]

vm.memory.load(program)
print(vm)
vm.execute()
print(vm)

