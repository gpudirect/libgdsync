GPUDirect Sync
========

Introduction
===

GPUDirect Sync (aka PeerSync) is all about moving control logic from
third-party devices to the GPU.

The CPU is taken off the control path, replaced by the GPU which is now
able to schedule both computation and network communication tasks
seamlessly. There are substantial improvements for both time-to-solution
(40% less latency) and power-to-solution (45% less CPU load) scenarios.


Requirements
===

This prototype has been tested on RHEL 6.x only.

A recent display driver, i.e. r361, r367 or later, is required.

A recent CUDA Toolkit is required, minimally 8.0, because of the CUDA driver MemOP APIs.

Mellanox OFED 2.5 or newer is required, because of the peer-direct verbs extensions.

The GDRCopy library (https://github.com/drossetti/gdrcopy) is necessary to
create CPU-side user-space mappings of GPU memory, currently used when
allocating a CQ on GPU memory.



Caveats
===

Tests have been done using Mellanox Connect-IB. Any HCA driven by mlx5
driver should work.

Kepler and Maxwell Tesla/Quadro GPUs are required for RDMA.

A special HCA firmware is currently necessary in combination with GPUs
prior to Pascal.


Build
===

Git repository does not include autotools files. The first time the directory
must be configured by running autogen.sh

As an example, the build.sh script is provided. You should modify it
according to the desired destination paths as well as the location
of the dependencies.
