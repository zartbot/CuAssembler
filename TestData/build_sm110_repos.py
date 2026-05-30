#!/usr/bin/env python3
"""Build DefaultInsAsmRepos.sm_110.txt encoding table.

Usage (on server1 with CUDA 13.3):
  1. Compile test kernels:
     nvcc -O0 -arch=sm_110a -cubin -o tcgen05_soup.cubin tcgen05_encoding_soup.cu
  2. Extract SASS:
     cuobjdump --dump-sass /opt/microbench/bin/*.cubin > all_sm110.sass
     cuobjdump --dump-sass tcgen05_soup.cubin >> all_sm110.sass
  3. Run this script:
     python3 build_sm110_repos.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuAsmLogger import CuAsmLogger

CuAsmLogger.initLogger()

def build_repos(sass_files, output_file, base_arch='sm_120'):
    print(f'=== Building SM_110 InsAsmRepos ===')

    # Start from sm_120 base for Blackwell baseline instructions
    print(f'Loading base repos from {base_arch}...')
    repos = CuInsAssemblerRepos.getDefaultRepos(base_arch)
    repos.convertArch('sm_110')
    print(f'  Base repos: {len(repos)} instruction keys')

    # Feed all SM_110 SASS data
    for sass_file in sass_files:
        if not os.path.isfile(sass_file):
            print(f'  Skipping {sass_file} (not found)')
            continue
        print(f'Feeding {sass_file}...')
        feeder = CuInsFeeder(sass_file, archfilter='sm_110')
        ncnt = repos.update(feeder)
        print(f'  Updated {ncnt} new entries, total: {len(repos)} keys')

    # Add predicate encoding variants
    print('Completing predicate codes...')
    repos.completePredCodes()
    print(f'  Final: {len(repos)} instruction keys')

    # Save
    print(f'Saving to {output_file}...')
    repos.save2file(output_file)
    print(f'=== Done ===')

    return repos


def verify_repos(sass_files, repos_file):
    print(f'\n=== Verifying SM_110 InsAsmRepos ===')
    repos = CuInsAssemblerRepos(repos_file, arch='sm_110')
    print(f'  Loaded {len(repos)} instruction keys')

    for sass_file in sass_files:
        if not os.path.isfile(sass_file):
            continue
        print(f'Verifying against {sass_file}...')
        feeder = CuInsFeeder(sass_file, archfilter='sm_110')
        repos.verify(feeder)


if __name__ == '__main__':
    sass_files = ['all_sm110.sass']
    output_file = os.path.join(os.path.dirname(__file__), '..',
                               'CuAsm', 'InsAsmRepos',
                               'DefaultInsAsmRepos.sm_110.txt')

    if len(sys.argv) > 1:
        sass_files = sys.argv[1:]

    repos = build_repos(sass_files, output_file)
    verify_repos(sass_files, output_file)
