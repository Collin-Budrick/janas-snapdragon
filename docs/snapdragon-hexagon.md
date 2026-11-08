# Reproducing the optional Windows ARM64 Hexagon runtime

Janas submits matrix operations to a separately built GGML Hexagon backend. Janas still owns model execution, routing, expert caching and CPU fallback. This path has been validated on this Snapdragon X Elite laptop, Windows ARM64, **HTP0 / Hexagon v73**. The package also contains v75/v79/v81 DSP libraries; their presence does not establish support on those devices. These steps do not establish NPU-only model execution.

The packaged build records these versions in `build-info.json`:

| Component | Recorded version |
| --- | --- |
| llama.cpp / GGML source | `1ec81880944a63bc4aaf1abfe9a6d35c7569a757` |
| Native Windows ARM64 host compiler | LLVM Clang **23.1.5**, GNU-style frontend targeting `arm64-pc-windows-msvc` |
| Hexagon SDK | **6.6.0.0** |
| Hexagon compiler | **19.0.07** |
| Host MSVC libraries | **14.51.36231**, ARM64 |
| Host Windows SDK headers/libraries | **10.0.26100.0**, ARM64 |
| Catalog generation tool | Windows Kit **10.0.28000.0**, x86 `Inf2Cat.exe` |

The checked package is at `%USERPROFILE%\.cache\janas-toolchains\llama-hexagon-kquants-runtime`. Keep its matching headers, host DLLs and DSP libraries together. Preserve an existing working runtime when rebuilding.

## Build the runtime

Use a native Windows ARM64 developer environment with the recorded MSVC/Windows SDK `INCLUDE` and ARM64 `LIB` paths configured, plus CMake, Ninja, Python and LLVM on `PATH`. The recorded build used `HostARM64/ARM64` MSVC tools. The Qualcomm SDK was already installed at `C:\Qualcomm\Hexagon_SDK\6.6.0.0`; obtain and use it under its own license.

In PowerShell, set `$Source`, `$Build` and `$Runtime` to separate source/build/package directories. Set `$Ninja` to the installed `ninja.exe`. Use a fresh source checkout pinned to the commit above:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git "$Source"
git -C "$Source" checkout --detach 1ec81880944a63bc4aaf1abfe9a6d35c7569a757

$env:HEXAGON_SDK_ROOT = 'C:\Qualcomm\Hexagon_SDK\6.6.0.0'
$env:HEXAGON_TOOLS_ROOT = "$env:HEXAGON_SDK_ROOT\tools\HEXAGON_Tools\19.0.07"
$env:HEXAGON_HTP_CERT = ''
$env:PATH = "C:\Program Files\LLVM\bin;$env:PATH"

cmake -S "$Source" -B "$Build" -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$Source\cmake\arm64-windows-llvm.cmake" `
  "-DCMAKE_MAKE_PROGRAM=$Ninja" -DCMAKE_BUILD_TYPE=Release `
  -DGGML_HEXAGON=ON -DGGML_OPENCL=OFF -DGGML_OPENMP=OFF `
  "-DHEXAGON_SDK_ROOT=$env:HEXAGON_SDK_ROOT" `
  "-DHEXAGON_TOOLS_ROOT=$env:HEXAGON_TOOLS_ROOT" `
  -DPREBUILT_LIB_DIR=windows_aarch64 '-DGGML_HEXAGON_HTP_CERT=' `
  -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON `
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_UI=OFF

cmake --build "$Build" --parallel 4 --target `
  ggml ggml-hexagon test-backend-ops htp-v73 htp-v75 htp-v79 htp-v81
```

These options and targets reproduce the recorded build configuration. The upstream toolchain file supplies the host architecture/float flags; the Hexagon sub-build supplies its DSP flags and SDK libraries. Keep `configure.log`, `build.log` and `CMakeCache.txt`. The original command also supplied `Python3_EXECUTABLE`; CMake reported it unused.

The host build used Windows SDK resource tools, including `C:\Qualcomm\sdkbin\arm64\rc.exe`, on `PATH`. Use the installed tool and its dependencies. For catalog generation below, use the complete Windows Kit installation: a copied x86 `Inf2Cat.exe` in `C:\Qualcomm\sdkbin\x86` lacked dependencies in this session.

## Stage and sign the package

The recorded staging copied all built host DLLs, the standalone operation checker, all four DSP libraries, the original INF, public headers and upstream license:

```powershell
New-Item -ItemType Directory -Force "$Runtime\bin", "$Runtime\htp", "$Runtime\include" | Out-Null
Copy-Item "$Build\bin\*.dll" "$Runtime\bin"
Copy-Item "$Build\bin\test-backend-ops.exe" "$Runtime\bin"
Copy-Item "$Build\ggml\src\ggml-hexagon\libggml-htp-v*.so" "$Runtime\htp"
Copy-Item "$Source\ggml\src\ggml-hexagon\libggml-htp.inf" "$Runtime\htp"
Copy-Item "$Source\ggml\include\*.h" "$Runtime\include"
Copy-Item "$Source\LICENSE" "$Runtime\LICENSE"
```

The checked `bin` contains `ggml-base.dll`, `ggml-hexagon.dll`, `ggml-cpu.dll`, `ggml.dll`, `llama.dll`, `llama-common.dll` and `test-backend-ops.exe`. Janas directly loads `ggml-base.dll` and `ggml-hexagon.dll`; the other files support the independent reference checker. The installed Qualcomm driver supplies `libcdsprpc.dll` from the Windows DriverStore.

Use an existing signing identity whose trust and custom-DSP loading policy are already configured for the target machine. Set `$SigningPfx` to the user-provided signing file and `$ExpectedSignerThumbprint` to its expected public signer identity. Keep signing material outside the source/runtime package. The recorded command used no password argument.

```powershell
$Inf2Cat = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.28000.0\x86\Inf2Cat.exe'
$SignTool = 'C:\Qualcomm\sdkbin\arm64\signtool.exe'
$Catalog = "$Runtime\htp\libggml-htp.cat"

& $Inf2Cat "/driver:$Runtime\htp" /os:10_25H2_ARM64
& $SignTool sign /fd SHA256 /f "$SigningPfx" "$Catalog"
```

This is catalog signing: do not change the `.so` files or INF after generating the catalog. The upstream optional CMake signing target uses the same OS identifier, but this package was signed after staging with `GGML_HEXAGON_HTP_CERT` empty during the build. These commands require no boot-policy or trust-store changes.

## Verify before using it

Check the expected signer and verify catalog membership for each DSP library:

```powershell
$signature = Get-AuthenticodeSignature -LiteralPath $Catalog
if ($signature.Status -ne 'Valid' -or
    $signature.SignerCertificate.Thumbprint -ne $ExpectedSignerThumbprint) {
    throw 'Catalog signer/status mismatch'
}
foreach ($version in 'v73', 'v75', 'v79', 'v81') {
    & $SignTool verify /pa /c "$Catalog" "$Runtime\htp\libggml-htp-$version.so"
    if ($LASTEXITCODE -ne 0) { throw "Catalog verification failed: $version" }
}
```

The recorded package's catalog signature was `Valid`, and its recorded `verify.log` verifies v73. The loop above extends that membership check to all staged libraries. Inspect host binaries with `llvm-readobj --file-headers`: host DLLs must be ARM64 PE images; DSP `.so` files must be Hexagon ELF images, not Windows DLLs.

For an existing package, verify its recorded manifest:

```powershell
Get-Content "$Runtime\sha256.json" -Raw | ConvertFrom-Json | ForEach-Object {
    $file = Join-Path $Runtime $_.File
    if ((Get-Item -LiteralPath $file).Length -ne $_.Size -or
        (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $_.SHA256) {
        throw "Runtime manifest mismatch: $($_.File)"
    }
}
```

A newly built/signed package needs its own `build-info.json`, SHA-256 manifest and verification logs; signing and build metadata can change hashes. Record the exact commit/tool versions, signer identity and whether hardware inference has actually run. Do not treat a successful build or valid catalog as an inference result.

## Build and run Janas

Janas uses LLVM-MinGW for its Windows ARM64 build, separately from the runtime's MSVC-targeted Clang build. In the WSL build shell, use the installed ARM64 LLVM-MinGW tools and the staged headers:

```bash
task_toolchain="$HOME/.cache/janas-toolchains/llvm-mingw-20261104-ucrt-ubuntu-22.04-aarch64/bin"
task_runtime='/mnt/c/Users/colli/.cache/janas-toolchains/llama-hexagon-kquants-runtime'

JANAS_OS=windows \
CC="$task_toolchain/aarch64-w64-mingw32-clang" \
AR="$task_toolchain/llvm-ar" \
WINDRES="$task_toolchain/aarch64-w64-mingw32-windres" \
JANAS_HEXAGON_INCLUDE="$task_runtime/include" \
./build.sh release build
```

Adjust the example runtime path for your account. `JANAS_HEXAGON_INCLUDE` enables the provider only for a Windows ARM64 target; no GGML import libraries are linked. Keep the existing Vulkan-header/glslang setup if comparing GPU combinations; this setting enables Hexagon independently.

In the native Windows process environment used to launch Janas:

```powershell
$env:JANAS_HEXAGON_ROOT = $Runtime
$env:ADSP_LIBRARY_PATH = "$Runtime\htp"
$env:PATH = "$Runtime\bin;$env:PATH"
$env:JANAS_NPU = '1'
$env:JANAS_NPU_CHECK = '1'
$env:JANAS_NPU_STATS = '1'

& "$Janas\bin\aarch64-windows\npu_hexagon_check.exe" --packed
```

`$Janas` is the Janas checkout. These environment assignments affect this process and its children. Restart Janas before switching runtime directories: GGML installs a process-wide terminate handler, so the adapter deliberately retains/pins its runtime DLLs for the process lifetime. `JANAS_NPU_MAX_MB` limits unique device weight-buffer bytes; per-batch compute buffers and host memory are additional allocations.

The probe exercises Janas's matrix API, including packed Q4_K/Q6_K, batch shapes, shared weights and context lifetimes. It does not run a whole model. Janas's model adapter keeps supported packed formats, dequantizes unsupported formats to F16, and uses CPU fallback when a shape/type/allocation cannot run. JNS Q6_K_P is plane-based and is never reinterpreted as raw GGML Q6_K.

## Evidence and source provenance

`HardwareInferenceTested: false` in the immutable package metadata describes its packaging-time state. Later real-device logs are in `lib/validation/janas-hexagon-packed-20261108.log` and `lib/validation/combined-npu_hexagon_check-20261108.log`: HTP0/v73 F16, Q4_K and Q6_K operations at batches 1/2/4/16 passed, including shared-weight rebatching. Model throughput and output-quality evidence belong to the separate model benchmark reports.

The packed-operation probe checks finite outputs and NMSE against the pinned upstream MUL_MAT tolerance of `5e-4`, and reports maximum absolute error. Small-batch packed kernels dynamically quantize activations, so they are not expected to match F32 reference products bit for bit.

Source references at the exact runtime commit:

- [Hexagon CMake build, DSP targets and catalog command](https://github.com/ggml-org/llama.cpp/blob/1ec81880944a63bc4aaf1abfe9a6d35c7569a757/ggml/src/ggml-hexagon/CMakeLists.txt).
- [Windows ARM64 LLVM toolchain](https://github.com/ggml-org/llama.cpp/blob/1ec81880944a63bc4aaf1abfe9a6d35c7569a757/cmake/arm64-windows-llvm.cmake).
- [MUL_MAT correctness criteria](https://github.com/ggml-org/llama.cpp/blob/1ec81880944a63bc4aaf1abfe9a6d35c7569a757/tests/test-backend-ops.cpp#L4844-L4870).
- [Runtime MIT license](https://github.com/ggml-org/llama.cpp/blob/1ec81880944a63bc4aaf1abfe9a6d35c7569a757/LICENSE), copied into the runtime package. Janas's adapter remains under the repository's GPL-3.0-or-later license. Qualcomm SDK/toolchain components retain their own licensing terms.
