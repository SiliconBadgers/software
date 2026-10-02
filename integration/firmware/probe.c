// Bare-metal integration probe. Mailbox is simulation-only, MMIO follows slide offsets.
typedef unsigned int u32;
#define REG(off) (*(volatile u32 *)(0x10000000u + (off)))
#define MAIL(off) (*(volatile u32 *)(0x2000u + (off)))
static void fence(void) { __asm__ volatile ("fence iorw, iorw" ::: "memory"); }
int main(void) {
  MAIL(20) = 0x52454144; // READY marker, observed by host harness.
  for (;;) {
    if (!MAIL(0)) continue;
    fence();
    while (!(REG(0x0c) & 1)) {}
    REG(0x10) = 0x2100;
    REG(0x14) = 0x20000000; // Descriptor SRAM region, per slide baseline.
    REG(0x18) = 128;
    REG(0x1c) = MAIL(4);
    fence();
    REG(0x20) = 1;
    while (!(REG(0x0c) & 12)) {}
    fence();
    MAIL(8) = REG(0x28);
    MAIL(12) = REG(0x24);
    REG(0x34) = 1;
    fence();
    MAIL(0) = 0;
    fence();
    MAIL(16) = 1;
  }
}
