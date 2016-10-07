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

Mellanox OFED 3.5 or newer is required, because of the peer-direct verbs
extensions.

Peer-direct verbs are only supported on the libmlx5 low-level plug-in
module, so either Connect-IB or ConnectX-4 HCAs are required.

The GDRCopy library (https://github.com/NVIDIA/gdrcopy) is necessary to
create CPU-side user-space mappings of GPU memory, currently used when
allocating verbs objects on GPU memory.


Caveats
===
Tests have been done using Mellanox Connect-IB. Any HCA driven by mlx5
driver should work.

Kepler or newer Tesla/Quadro GPUs are required because of GPUDirect RDMA.

A special HCA firmware is currently necessary in combination with GPUs
prior to Pascal.


Build
===
Git repository does not include autotools files. The first time the directory
must be configured by running autogen.sh

As an example, the build.sh script is provided. You should modify it
according to the desired destination paths as well as the location
of the dependencies.
