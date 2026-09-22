/*
 * embedded/startup.c
 *
 * Minimal Cortex-M3 reset/vector table for the QEMU lm3s6965evb target.
 * No RTOS, no vendor HAL -- this is the actual bring-up code a flight
 * computer's boot path runs before your control loop ever executes.
 * We only need Reset + a handful of core exception vectors for the
 * ARMv7-M architecture to accept the image; unused vectors point at a
 * spin-loop fault handler (a real flight computer would instead log
 * the fault and trigger a safe abort sequence).
 */

extern unsigned long _estack;
extern unsigned long _sidata, _sdata, _edata, _sbss, _ebss;

void Reset_Handler(void);
static void Default_Handler(void) { while (1) { } }

int main(void);

__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))&_estack,   /* initial stack pointer */
    Reset_Handler,               /* Reset */
    Default_Handler,             /* NMI */
    Default_Handler,             /* HardFault */
    Default_Handler,             /* MemManage */
    Default_Handler,             /* BusFault */
    Default_Handler,             /* UsageFault */
    0, 0, 0, 0,                  /* reserved */
    Default_Handler,             /* SVCall */
    Default_Handler,             /* DebugMonitor */
    0,                            /* reserved */
    Default_Handler,             /* PendSV */
    Default_Handler,             /* SysTick */
};

void Reset_Handler(void) {
    /* Copy initialized .data from flash to RAM */
    unsigned long *src = &_sidata;
    unsigned long *dst = &_sdata;
    while (dst < &_edata) { *dst++ = *src++; }

    /* Zero .bss */
    dst = &_sbss;
    while (dst < &_ebss) { *dst++ = 0; }

    main();
    while (1) { }
}
