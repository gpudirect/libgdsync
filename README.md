GPUDirect Async
========

Introduction
===
libgdsync implements GPUDirect Async support for Infiniband verbs.

GPUDirect Async is all about moving control logic from third-party devices
to the GPU.

libgdsync provides APIs which are similar to Infiniband verbs but
synchronous to CUDA streams.


Requirements
===
This prototype has been tested on RHEL 6.x only.

A recent display driver, i.e. r361, r367 or later, is required.

A recent CUDA Toolkit is required, minimally 8.0, because of the CUDA
driver MemOP APIs.

Note that GPU peer mappings must be explicitly enabled, more on this below.

Mellanox OFED (MOFED) 4.0 or newer is required, because of the peer-direct verbs
extensions. As an alternative, it is possible to use MOFED 3.4 and replace the stock libmlx5 
with the one at https://github.com/gpudirect/libmlx5/tree/fixes.

Peer-direct verbs are only supported on the libmlx5 low-level plug-in
module, so either Connect-IB or ConnectX-4 HCAs are required.

The Mellanox OFED GPUDirect RDMA kernel module,
https://github.com/Mellanox/nv_peer_memory, is required to allow the HCA to
access the GPU memory.

The GDRCopy library (https://github.com/NVIDIA/gdrcopy) is required to
create CPU-side user-space mappings of GPU memory, currently used when
allocating verbs objects on GPU memory.


Caveats
===
Tests have been done using Mellanox Connect-IB. Any HCA driven by mlx5
driver should work.

Kepler or newer Tesla/Quadro GPUs are required because of GPUDirect RDMA.

A special HCA firmware setting is currently necessary in combination with GPUs
prior to Pascal. Use `mlxconfig` to set the `NON_PREFETCHABLE_PF_BAR` parameter
on your HCA to 1. For more information see [Mellanox Firmware Tools (MFT) User
Manual](https://www.mellanox.com/related-docs/MFT/MFT_user_manual_4_6_0.pdf).


Build
===
Git repository does not include autotools files. The first time the directory
must be configured by running autogen.sh

As an example, the build.sh script is provided. You should modify it
according to the desired destination paths as well as the location
of the dependencies.


Enabling GPU peer mappings
===

GPUDirect Async depends on the ability to create GPU peer mappings of the
HCA BAR space.

GPU peer mappings are mappings (in the sense of cuMemHostRegister) to the
PCI Express resource space of a third party device.
That feature is normally disable due to potential security problems, 
see https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2015-5053.
In fact, unless the PeerMappingOverride registry of the NVIDIA
kernel-mode driver is enabled, only root user can use that feature.

To enable GPU peer mappings for all users, the PeerMappingOverride registry
must be set to 1:
```shell
$ cat /etc/modprobe.d/nvidia.conf
options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"
```

After that, either reboot or manually reload the NVIDIA kernel
module:
```shell
# unload all kernel modules which depends on nvidia.ko
$ service gdrcopy stop
$ service nv_peer_mem stop
$ modprobe -r nvidia_uvm
$ modprobe -r nvidia
$ modprobe nvidia
...
```

Acknowledging GPUDirect Async
===

If you find this software useful in your work, please cite:

["GPUDirect Async: exploring GPU synchronous communication techniques for InfiniBand clusters"](https://www.sciencedirect.com/science/article/pii/S0743731517303386), E. Agostini, D. Rossetti, S. Potluri. Journal of Parallel and Distributed Computing, Vol. 114, Pages 28-45, April 2018

["Offloading communication control logic in GPU accelerated applications"](http://ieeexplore.ieee.org/document/7973709), E. Agostini, D. Rossetti, S. Potluri. Proceedings of the 17th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (CCGrid’ 17), IEEE Conference Publications, Pages 248-257, Nov 2016
