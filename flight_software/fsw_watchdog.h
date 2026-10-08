/* fsw_watchdog.h -- task-liveness ("window of check-ins") watchdog.
 * The hardware watchdog is kicked ONLY if every required task checked in
 * since the last service call. A stalled task therefore stops the kicks and
 * the MCU resets, instead of a healthy timer ISR keeping a dead main loop
 * alive (the classic mistake of kicking the dog from a timer interrupt). */
#ifndef FSW_WATCHDOG_H
#define FSW_WATCHDOG_H
#include <stdint.h>

typedef struct {
    uint32_t required_mask;
    uint32_t checked_mask;
    uint16_t miss_count;     /* consecutive service periods with a missing task */
    uint16_t miss_limit;
    uint32_t kicks;
    uint8_t  tripped;        /* software-detected persistent stall (for TLM)    */
} fsw_wdg_t;

void fsw_wdg_init(fsw_wdg_t *w, uint32_t required_mask, uint16_t miss_limit);
void fsw_wdg_checkin(fsw_wdg_t *w, uint32_t task_bit);
/* Returns 1 => caller should kick the hardware watchdog now. */
int  fsw_wdg_service(fsw_wdg_t *w);

#endif
