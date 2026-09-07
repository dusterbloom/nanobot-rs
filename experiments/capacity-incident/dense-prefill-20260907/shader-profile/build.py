from pathlib import Path
import subprocess
p=Path('/private/tmp/higgs-fused-profile');b=Path('/Users/peppi/Dev/higgs/target/release/build/mlx-sys-e955ea9cb0a9ee5d/out/build');d=b/'_deps'
cmd=['/usr/bin/clang++','-std=c++20','-O3','-DNDEBUG','-DMLX_STATIC','-DMLX_METAL_NO_NAX','-DFMT_HEADER_ONLY=1','-mmacosx-version-min=26.0']
for inc in ['mlx-src','metal_cpp-src','fmt-src/include','json-src/single_include/nlohmann']:cmd+=['-I'+str(d/inc)]
subprocess.run(cmd+[str(p/'probe.cpp'),str(p/'attention.cpp'),str(b/'lib/libmlx.a'),'-framework','Metal','-framework','Foundation','-framework','Accelerate','-o',str(p/'probe')],check=True)
