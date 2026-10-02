.section .text.start
.globl _start
_start:
  li sp, 0xfff0
  call main
1: j 1b
