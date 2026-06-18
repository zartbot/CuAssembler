#!/bin/bash
# Build SM_100 encoding repos on sm_103a server.
# Usage: bash build_sm100_repos.sh
set -e

cd /opt/microbench/CuAssembler/TestData

ARCH=sm_103a
NVCC_FLAGS="-std=c++17 -lineinfo -gencode arch=compute_103a,code=sm_103a"

echo "=== Compiling mega_soup_sm100.cu at O0 ==="
nvcc -O0 $NVCC_FLAGS -cubin -o mega_soup_sm100_O0.cubin mega_soup_sm100.cu 2>&1 || {
    echo "O0 compile failed, trying without some features..."
    # Try without cvt.rs and redux.f32 if they fail
    nvcc -O0 $NVCC_FLAGS -cubin -o mega_soup_sm100_O0.cubin mega_soup_sm100.cu 2>&1
}

echo "=== Compiling mega_soup_sm100.cu at O2 ==="
nvcc -O2 $NVCC_FLAGS -cubin -o mega_soup_sm100_O2.cubin mega_soup_sm100.cu 2>&1

echo "=== Compiling tcgen05_encoding_soup.cu at O0 (for SM_100) ==="
nvcc -O0 $NVCC_FLAGS -cubin -o tcgen05_sm100_O0.cubin tcgen05_encoding_soup.cu 2>&1

echo "=== Compiling encoding_exerciser.cu at O0 ==="
nvcc -O0 $NVCC_FLAGS -cubin -o encoding_sm100_O0.cubin encoding_exerciser.cu 2>&1

echo "=== Compiling encoding_exerciser.cu at O2 ==="
nvcc -O2 $NVCC_FLAGS -cubin -o encoding_sm100_O2.cubin encoding_exerciser.cu 2>&1

echo "=== Compiling branch_exerciser.cu at O0 ==="
nvcc -O0 $NVCC_FLAGS -cubin -o branch_sm100_O0.cubin branch_exerciser.cu 2>&1 || echo "branch_exerciser skip"

echo ""
echo "=== Dumping SASS ==="
cuobjdump --dump-sass mega_soup_sm100_O0.cubin > all_sm100_O0.sass 2>&1
cuobjdump --dump-sass mega_soup_sm100_O2.cubin > all_sm100_O2.sass 2>&1
cuobjdump --dump-sass tcgen05_sm100_O0.cubin >> all_sm100_O0.sass 2>&1
cuobjdump --dump-sass encoding_sm100_O0.cubin >> all_sm100_O0.sass 2>&1
cuobjdump --dump-sass encoding_sm100_O2.cubin >> all_sm100_O2.sass 2>&1
[ -f branch_sm100_O0.cubin ] && cuobjdump --dump-sass branch_sm100_O0.cubin >> all_sm100_O0.sass 2>&1

# Merge O0 + O2 into one big sass file
cat all_sm100_O0.sass all_sm100_O2.sass > all_sm100.sass

echo ""
echo "=== SASS line counts ==="
wc -l all_sm100_O0.sass all_sm100_O2.sass all_sm100.sass

echo ""
echo "=== Building repos file ==="
cd /opt/microbench/CuAssembler
python3 -c "
import sys
sys.path.insert(0, '.')
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuAsmLogger import CuAsmLogger
CuAsmLogger.initLogger(stdout_level=30)

arch = 'sm_100'
sass_file = 'TestData/all_sm100.sass'
repos_file = 'CuAsm/InsAsmRepos/DefaultInsAsmRepos.sm_100.txt'

print(f'Building repos from {sass_file} for {arch}...')
feeder = CuInsFeeder(sass_file, archfilter=arch)
repos = CuInsAssemblerRepos(arch=arch)
repos.update(feeder)

print(f'Verifying...')
feeder.restart()
repos.verify(feeder)

repos.save2file(repos_file)
print(f'Saved to {repos_file}')
print(f'Total instruction keys: {len(repos.m_InsAsmDict)}')

# List all instruction keys
for k in sorted(repos.m_InsAsmDict.keys()):
    print(f'  {k}')
"

echo ""
echo "=== Done ==="
wc -l CuAsm/InsAsmRepos/DefaultInsAsmRepos.sm_100.txt
