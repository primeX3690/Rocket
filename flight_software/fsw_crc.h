/* fsw_crc.h -- CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection).
 * Check value for ASCII "123456789" is 0x29B1. Bitwise implementation:
 * no lookup table, so it costs ~0 flash and is easy to audit. */
#ifndef FSW_CRC_H
#define FSW_CRC_H
#include <stddef.h>
#include <stdint.h>

uint16_t fsw_crc16_ccitt(const uint8_t *data, size_t len, uint16_t init);

#endif
