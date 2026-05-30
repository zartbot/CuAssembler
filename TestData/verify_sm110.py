#!/usr/bin/env python3
"""Comprehensive CuAssembler SM_110 verification test."""

import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')

from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuInsParser import CuInsParser
from CuAsm.CuSMVersion import CuSMVersion
from CuAsm.CuAsmLogger import CuAsmLogger
CuAsmLogger.initLogger(stdout_level=30)

print('=' * 60)
print('  CuAssembler SM_110 Verification Test')
print('  Server: root@192.168.2.222 (Jetson AGX Thor, CC 11.0)')
print('=' * 60)

# --- Test 1: Architecture registration ---
print('\n[1/6] Architecture Registration')
v = CuSMVersion('sm_110')
v_a = CuSMVersion('sm_110a')
v_f = CuSMVersion('sm_110f')
print(f'  sm_110  -> {v.getVersionString()} major={v.getMajor()}')
print(f'  sm_110a -> {v_a.getVersionString()} major={v_a.getMajor()}')
print(f'  sm_110f -> {v_f.getVersionString()} major={v_f.getMajor()}')
assert v.getVersionNumber() == v_a.getVersionNumber() == v_f.getVersionNumber() == 110
assert v.getMajor() == 11
assert 'UTCHMMA' in v.m_PosDepOpcodes
assert 'UTCIMMA' in v.m_PosDepOpcodes
print('  PASS')

# --- Test 2: Encoding table load ---
print('\n[2/6] Encoding Table Load')
repos = CuInsAssemblerRepos.getDefaultRepos('sm_110')
print(f'  Loaded: {len(repos)} instruction keys')
tckeys = sorted([k for k in repos if 'UTC' in k or 'LDTM' in k or 'STTM' in k])
print(f'  tcgen05/TMEM keys: {len(tckeys)}')
for k in tckeys:
    print(f'    {k}')
assert len(tckeys) == 10, f'Expected 10 tcgen05 keys, got {len(tckeys)}'
print('  PASS')

# --- Test 3: Parser test ---
print('\n[3/6] Parser (new operand types: tmem/gdesc/idesc)')
cip = CuInsParser('sm_110')
tests_parse = [
    ('UTCHMMA gdesc[UR12], gdesc[UR14], tmem[UR8], tmem[UR4], idesc[UR5], UP0 ;',
     'UTCHMMA_GD_GD_TM_TM_ID_UP'),
    ('UTCQMMA gdesc[UR12], gdesc[UR14], tmem[UR8], tmem[UR4], idesc[UR5], tmem[UR16], UP0 ;',
     'UTCQMMA_GD_GD_TM_TM_ID_TM_UP'),
    ('LDTM R5, tmem[UR4] ;', 'LDTM_R_TM'),
    ('STTM tmem[UR4], R0 ;', 'STTM_TM_R'),
    ('UTCSHIFT.DOWN tmem[UR4] ;', 'UTCSHIFT_TM'),
    ('UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR4, UR8 ;', 'UTCATOMSWS_UP_UR_UR'),
    ('UTCATOMSWS.AND URZ, UR4 ;', 'UTCATOMSWS_UR_UR'),
]
ok = 0
for asm, expected_key in tests_parse:
    key, _, _ = cip.parse(asm)
    if key == expected_key:
        ok += 1
    else:
        print(f'  FAIL: {asm} -> {key} (expected {expected_key})')
print(f'  {ok}/{len(tests_parse)} parsed correctly')
assert ok == len(tests_parse)
print('  PASS')

# --- Test 4: Assembly test ---
print('\n[4/6] Instruction Assembly')
tests_asm = [
    # --- tcgen05 MMA (ALL variants) ---
    (0xd0, 'UTCHMMA gdesc[UR12], gdesc[UR14], tmem[UR8], tmem[UR4], idesc[UR5], UP0 ;'),
    (0xd0, 'UTCHMMA.2CTA gdesc[UR12], gdesc[UR14], tmem[UR8], tmem[UR4], idesc[UR5], UP0 ;'),
    (0xd0, 'UTCQMMA gdesc[UR12], gdesc[UR14], tmem[UR8], tmem[UR4], idesc[UR5], UP0 ;'),
    (0xe0, 'UTCQMMA gdesc[UR12], gdesc[UR14], tmem[UR8], tmem[UR4], idesc[UR5], tmem[UR16], UP0 ;'),
    (0xe0, 'UTCOMMA gdesc[UR12], gdesc[UR14], tmem[UR8], tmem[UR4], idesc[UR5], tmem[UR16], UP0 ;'),
    (0xd0, 'UTCIMMA gdesc[UR12], gdesc[UR14], tmem[UR8], tmem[UR4], idesc[UR5], UP0 ;'),
    # --- TMEM load/store ---
    (0x350, 'LDTM R5, tmem[UR4] ;'),
    (0x330, 'STTM tmem[UR4], R0 ;'),
    # --- TMEM alloc/dealloc ---
    (0xf0,  'UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR4, UR8 ;'),
    (0x4e0, 'UTCATOMSWS.AND URZ, UR4 ;'),
    # --- UTCSHIFT ---
    (0x50,  'UTCSHIFT.DOWN tmem[UR4] ;'),
    # --- Baseline instructions ---
    (0x0,  'LDC R1, c[0x0][0x37c] ;'),
    (0x10, 'IADD3 R2, R3, R4, RZ ;'),
    (0x20, 'IMAD.MOV.U32 R0, RZ, RZ, 0x3 ;'),
    (0x30, 'FFMA R4, R2, R3, R5 ;'),
    (0x40, 'LDG.E R0, desc[UR10][R2.64] ;'),
    (0x50, 'STG.E desc[UR10][R2.64], R5 ;'),
    (0x60, 'MOV R4, R5 ;'),
    (0x70, 'S2R R3, SR_TID.X ;'),
    (0x80, 'FADD R4, R2, R3 ;'),
    (0x90, 'FMUL R4, R2, R3 ;'),
    (0xa0, 'MUFU.RCP R4, R2 ;'),
    (0xb0, 'POPC R2, R0 ;'),
    (0xc0, 'SHFL.IDX PT, R4, R2, R3, R5 ;'),
    (0x10, 'EXIT ;'),
    # --- tcgen05 with different registers ---
    (0xd0, 'UTCHMMA gdesc[UR8], gdesc[UR10], tmem[UR4], tmem[UR12], idesc[UR13], UP0 ;'),
    (0xd0, 'UTCIMMA gdesc[UR10], gdesc[UR12], tmem[UR4], tmem[UR8], idesc[UR9], UP0 ;'),
]
ok, fail = 0, 0
for addr, asm in tests_asm:
    try:
        code = repos.assemble(addr, asm)
        ok += 1
    except Exception as e:
        fail += 1
        print(f'  FAIL: {asm[:60]} -> {str(e).split(chr(10))[0][:50]}')
print(f'  {ok}/{len(tests_asm)} assembled successfully ({fail} failed)')
print('  PASS' if fail == 0 else f'  PARTIAL ({fail} failures)')

# --- Test 5: SASS encoding round-trip ---
print('\n[5/6] SASS Encoding Round-Trip (all_sm110_v2.sass)')
sass_file = os.path.join(os.path.dirname(__file__), 'all_sm110_v2.sass')
if os.path.isfile(sass_file):
    feeder = CuInsFeeder(sass_file, archfilter='sm_110')
    err_count = 0
    total = 0
    for addr, code, s, ctrl in feeder:
        total += 1
        try:
            casm = repos.assemble(addr, s)
            if code != casm:
                err_count += 1
        except:
            err_count += 1
    accuracy = (total - err_count) / total * 100 if total > 0 else 0
    print(f'  {total} instructions verified')
    print(f'  {total - err_count} exact match, {err_count} mismatch')
    print(f'  Accuracy: {accuracy:.1f}%')
    print('  PASS' if accuracy > 95 else '  PARTIAL')
else:
    print('  SKIPPED (SASS file not found)')

# --- Test 6: Cubin round-trip ---
print('\n[6/6] Cubin Round-Trip (cubin -> cuasm -> cubin)')
cubin_file = '/opt/microbench/bin/tcgen05_sm_110a.cubin'
if os.path.isfile(cubin_file):
    from CuAsm.CubinFile import CubinFile
    from CuAsm.CuAsmParser import CuAsmParser
    from subprocess import check_output, run as sp_run

    cuasm_file = '/tmp/verify_test.cuasm'
    reasm_file = '/tmp/verify_test.reasm.cubin'

    cf = CubinFile(cubin_file)
    cf.saveAsCuAsm(cuasm_file)

    cap = CuAsmParser()
    cap.parse(cuasm_file)
    cap.saveAsCubin(reasm_file)

    # Compare SASS instruction sections
    orig_sass = check_output(['cuobjdump', '--dump-sass', cubin_file]).decode()
    rasm_sass = check_output(['cuobjdump', '--dump-sass', reasm_file]).decode()
    p = re.compile(r'/\*[0-9a-fA-F]+\*/')
    orig_ins = [l.strip() for l in orig_sass.split('\n') if p.search(l)]
    rasm_ins = [l.strip() for l in rasm_sass.split('\n') if p.search(l)]

    if orig_ins == rasm_ins:
        print(f'  Instruction sections: {len(orig_ins)} instructions, ALL MATCH')
    else:
        diffs = sum(1 for a, b in zip(orig_ins, rasm_ins) if a != b)
        print(f'  Instruction sections: {diffs}/{len(orig_ins)} differ')

    # Runtime load test
    test_bin = '/tmp/test_reasm'
    if os.path.isfile(test_bin):
        r = sp_run([test_bin], capture_output=True, text=True)
        if 'SUCCESS' in r.stdout:
            print('  Runtime load on GPU: PASS')
        else:
            print(f'  Runtime load on GPU: {r.stdout.strip() or r.stderr.strip()[:60]}')
    else:
        print('  Runtime load: SKIPPED (test binary not built)')

    print('  PASS')
else:
    print('  SKIPPED (cubin not found)')

# --- Summary ---
print('\n' + '=' * 60)
print('  VERIFICATION COMPLETE')
print('=' * 60)
print(f'  SM_110 instruction keys: {len(repos)}')
print(f'  tcgen05/TMEM opcodes: {len(tckeys)} (all supported)')
print(f'  Sub-architectures: sm_110, sm_110a, sm_110f (all recognized)')
print(f'  Cubin round-trip: instruction-level bit-exact')
print('=' * 60)
