ENTRY(_start)
SECTIONS {
  /DISCARD/ : { *(.eh_frame*) *(.comment*) *(.note*) }
  . = 0x80;
  .text : { *(.text.start) *(.text*) }
  .rodata : { *(.rodata*) }
  .data : { *(.data*) }
  .bss : { *(.bss*) *(COMMON) }
  ASSERT(. < 0x2000, "Firmware overlaps test mailbox")
}
