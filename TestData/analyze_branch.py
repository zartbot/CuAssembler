#!/usr/bin/env python3
"""Analyze BRA/CALL/RET offset encoding on SM_110."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuAsmLogger import CuAsmLogger
CuAsmLogger.initLogger(stdout_level=40)

feeder = CuInsFeeder('TestData/all_sm110_v4.sass', archfilter='sm_110')
calls = []
bras = []
for addr, code, s, ctrl in feeder:
    s = s.strip()
    if 'CALL.REL.NOINC' in s and ';' in s:
        parts = s.replace(';', '').split()
        for p in parts:
            if p.startswith('0x'):
                target = int(p, 16)
                calls.append((addr, target, code))
                break
    elif 'RET.REL.NODEC' in s:
        calls.append((addr, 0, code))  # RET has offset=0

print(f'CALL/RET instances: {len(calls)}')
print()
print('=== CALL offset encoding analysis ===')
print(f'{"addr":>6} {"target":>6} {"off>>4":>6} {"code_low64":>18} {"verify"}')

mismatches = 0
for addr, target, code in calls[:40]:
    if target == 0:
        continue
    offset = target - addr - 16
    off4 = offset >> 4
    code_low64 = code & ((1 << 64) - 1)
    # hypothesis: off4[5:0] -> code[23:18], off4[7:6] -> code[35:34]
    byte2 = (code_low64 >> 16) & 0xFF
    byte4 = (code_low64 >> 32) & 0xFF
    code_field_low = byte2 >> 2
    code_field_hi = (byte4 >> 2) & 0x3
    off_low = off4 & 0x3f
    off_hi = (off4 >> 6) & 0x3
    ok = (code_field_low == off_low and code_field_hi == off_hi)
    if not ok:
        mismatches += 1
    status = 'OK' if ok else f'MISMATCH low:{code_field_low:#x}!={off_low:#x} hi:{code_field_hi:#x}!={off_hi:#x}'
    print(f'0x{addr:04x} 0x{target:04x} 0x{off4:04x}  0x{code_low64:016x}  {status}')

# Try extended analysis for larger offsets
print(f'\nMismatches in first 40: {mismatches}')

# Check if off4 needs more bits (>8)
print('\n=== Large offset analysis (soup_full kernel) ===')
large_calls = [(a, t, c) for a, t, c in calls if t > 0 and (t - a - 16) >> 4 > 0xff]
print(f'Calls with offset>>4 > 0xFF: {len(large_calls)}')
for addr, target, code in large_calls[:10]:
    offset = target - addr - 16
    off4 = offset >> 4
    code_low64 = code & ((1 << 64) - 1)
    # Try: off4[5:0] -> code[23:18], off4[11:6] -> code[35:30]?
    byte2 = (code_low64 >> 16) & 0xFF
    byte3 = (code_low64 >> 24) & 0xFF
    byte4 = (code_low64 >> 32) & 0xFF
    # Try: off4 encoded as (off4 & 0x3f) at bits [23:18] and (off4>>6) at bits [35:30]
    code_low6 = byte2 >> 2
    code_mid = ((byte4 & 0xFC) >> 2) | ((byte3 & 0xC0) >> 6)  # bits 30-35
    off_low6 = off4 & 0x3f
    off_mid = (off4 >> 6) & 0x3f
    print(f'  0x{addr:04x}->0x{target:04x} off4=0x{off4:04x} code=0x{code_low64:016x}')
    print(f'    off4_bin={off4:012b} code_bytes=[{byte2:08b}|{byte3:08b}|{byte4:08b}]')
    print(f'    off_low6={off_low6:06b} code_low6={code_low6:06b} {"OK" if off_low6==code_low6 else "MISMATCH"}')
    print(f'    off_mid6={off_mid:06b} code_mid6={code_mid:06b} {"OK" if off_mid==code_mid else "MISMATCH"}')
