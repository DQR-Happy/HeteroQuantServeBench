# Jetson Environment Manifest

Generated: 2026-08-06T23:53:28+08:00

## Device
```text
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
aarch64
```

## Jetson Linux / L4T
```text
# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 01:49:37 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## Operating system
```text
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```

## CUDA
```text
CUDA_HOME=not-set
nvcc=/usr/local/cuda-12.6/bin/nvcc
/usr/local/cuda-12.6
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:53:06_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```

## Toolchain
```text
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
cmake version 3.22.1
1.13.0.git.kitware.jobserver-pipe-1
Python 3.10.12
git version 2.34.1
```

## NVIDIA packages
```text
nvidia-l4t-core	36.4.3-20250107174145
```

## Memory
```text
               total        used        free      shared  buff/cache   available
Mem:           7.4Gi       4.1Gi       822Mi        27Mi       2.6Gi       3.1Gi
Swap:           11Gi        40Mi        11Gi
```

## Storage
```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  233G   95G  128G  43% /
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  233G   95G  128G  43% /
NAME           SIZE TYPE FSTYPE   MOUNTPOINTS
loop0            4K loop          /snap/bare/5
loop1        182.2M loop          /snap/chromium/3422
loop2        184.4M loop          /snap/chromium/3445
loop3         68.9M loop          /snap/core22/2134
loop4           69M loop          /snap/core22/2412
loop5         61.9M loop          /snap/core24/1644
loop6         47.9M loop          /snap/cups/1198
loop7         47.9M loop          /snap/cups/1208
loop8        493.5M loop squashfs /snap/gnome-42-2204/201
loop9          503M loop squashfs /snap/gnome-42-2204/245
loop10       552.9M loop squashfs /snap/gnome-46-2404/154
loop11        91.7M loop squashfs /snap/gtk-common-themes/1535
loop12       174.6M loop squashfs /snap/mesa-2404/1166
loop13        42.6M loop squashfs /snap/snapd/26869
loop14        38.7M loop squashfs /snap/snapd/23546
loop15          16M loop          
zram0          635M disk          [SWAP]
zram1          635M disk          [SWAP]
zram2          635M disk          [SWAP]
zram3          635M disk          [SWAP]
zram4          635M disk          [SWAP]
zram5          635M disk          [SWAP]
nvme0n1      238.5G disk          
├─nvme0n1p1    237G part ext4     /
├─nvme0n1p2    128M part          
├─nvme0n1p3    768K part          
├─nvme0n1p4   31.6M part          
├─nvme0n1p5    128M part          
├─nvme0n1p6    768K part          
├─nvme0n1p7   31.6M part          
├─nvme0n1p8     80M part          
├─nvme0n1p9    512K part          
├─nvme0n1p10    64M part vfat     /boot/efi
├─nvme0n1p11    80M part          
├─nvme0n1p12   512K part          
├─nvme0n1p13    64M part          
├─nvme0n1p14   400M part          
└─nvme0n1p15 479.5M part          
```

## Power mode
```text
NV Power Mode: 15W
0
```

## Clock state
```text
SOC family:tegra234  Machine:NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
Online CPUs: 0-5
cpu0:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1497600 CurrentFreq=1497600 IdleStates: WFI=1 c7=1 
cpu1:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1497600 CurrentFreq=1190400 IdleStates: WFI=1 c7=1 
cpu2:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1497600 CurrentFreq=1344000 IdleStates: WFI=1 c7=1 
cpu3:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1497600 CurrentFreq=1190400 IdleStates: WFI=1 c7=1 
cpu4:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1497600 CurrentFreq=729600 IdleStates: WFI=1 c7=1 
cpu5:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1497600 CurrentFreq=729600 IdleStates: WFI=1 c7=1 
GPU MinFreq=306000000 MaxFreq=612000000 CurrentFreq=306000000
Active GPU TPCs: 4
EMC MinFreq=204000000 MaxFreq=2133000000 CurrentFreq=2133000000 FreqOverride=0
FAN Dynamic Speed Control=kernel hwmon0_pwm1=88
NV Power Mode: 15W
```
