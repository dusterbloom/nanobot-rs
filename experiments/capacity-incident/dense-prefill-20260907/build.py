import pathlib,subprocess
b=pathlib.Path('/Users/peppi/Dev/higgs/target/release/build/mlx-sys-e955ea9cb0a9ee5d/out/build');d=b/'_deps';p=pathlib.Path('/private/tmp/higgs-dense-prefill')
cmd=['/usr/bin/clang++','-std=c++20','-O3','-DNDEBUG','-DMLX_STATIC','-DMLX_METAL_NO_NAX','-DFMT_HEADER_ONLY=1','-mmacosx-version-min=26.0']
for inc in ['mlx-src','metal_cpp-src','fmt-src/include','json-src/single_include/nlohmann']:cmd+=['-I'+str(d/inc)]
cmd += [str(p/'probe.cpp'),str(p/'attention-probe.cpp'),str(b/'lib/libmlx.a'),'-framework','Metal','-framework','Foundation','-framework','Accelerate','-o',str(p/'probe')]
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-fno-fast-math','-I'+str(d/'mlx-src'),'-c',str(p/'d256.metal'),'-o',str(p/'d256.air')],check=True)
subprocess.run(['xcrun','metallib',str(p/'d256.air'),'-o',str(p/'d256.metallib')],check=True)
r=subprocess.run(cmd);raise SystemExit(r.returncode)
