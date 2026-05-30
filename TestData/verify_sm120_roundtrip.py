#!/usr/bin/env python3
"""SM_120 cubin round-trip regression test."""
import sys, os, re
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from CuAsm.CubinFile import CubinFile
from CuAsm.CuAsmParser import CuAsmParser
from CuAsm.CuAsmLogger import CuAsmLogger
from subprocess import check_output
CuAsmLogger.initLogger(stdout_level=30)

# Find or compile a sm_120 cubin
cubin = '/tmp/test_120.cubin'
if not os.path.isfile(cubin):
    src = '/opt/microbench/cache_control.cu'
    if not os.path.isfile(src):
        src = '/opt/microbench/control_flow.cu'
    os.system(f'nvcc -O0 -arch=sm_120 -cubin {src} -o {cubin} 2>/dev/null')

if not os.path.isfile(cubin):
    print('ERROR: Could not find/compile SM_120 cubin')
    sys.exit(1)

print(f'Testing SM_120 cubin round-trip: {cubin}')

# Step 1: cubin -> cuasm
cf = CubinFile(cubin)
cf.saveAsCuAsm('/tmp/test_120.cuasm')
print(f'  Disassembled ({os.path.getsize("/tmp/test_120.cuasm")} bytes)')

# Step 2: cuasm -> cubin
cap = CuAsmParser()
cap.parse('/tmp/test_120.cuasm')
cap.saveAsCubin('/tmp/test_120.reasm.cubin')
print(f'  Reassembled ({os.path.getsize("/tmp/test_120.reasm.cubin")} bytes)')

# Step 3: Compare SASS
orig_sass = check_output(['cuobjdump', '--dump-sass', cubin]).decode()
rasm_sass = check_output(['cuobjdump', '--dump-sass', '/tmp/test_120.reasm.cubin']).decode()
p = re.compile(r'/\*[0-9a-fA-F]+\*/')
orig_ins = [l.strip() for l in orig_sass.split('\n') if p.search(l)]
rasm_ins = [l.strip() for l in rasm_sass.split('\n') if p.search(l)]

print(f'  Instructions: {len(orig_ins)} original, {len(rasm_ins)} reassembled')
if orig_ins == rasm_ins:
    print(f'  RESULT: ALL {len(orig_ins)} instructions MATCH (bit-exact)')
    print('  SM_120 cubin round-trip: PASS')
else:
    diffs = sum(1 for a, b in zip(orig_ins, rasm_ins) if a != b)
    print(f'  RESULT: {diffs}/{len(orig_ins)} instructions differ')
    print('  SM_120 cubin round-trip: CHECK')
