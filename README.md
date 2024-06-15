A generic virtual machine written in python.

```python
vm = VM(bits=16, num_registers=16, memory_size=2*KB)

instructions = [
    vm.instruction(Operator.Nop),
    vm.instruction(Operator.MoveRI,     Cpu.Register.R0,    vm.int(0x1234)),
    vm.instruction(Operator.Exit)
]

program = [b for i in instructions for b in i]

vm.memory.load(program)
```
