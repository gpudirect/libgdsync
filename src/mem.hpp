#pragma once

int gds_peer_mfree(int peer_id, uint64_t peer_data, void *host_addr, void *handle);
int gds_peer_malloc(int peer_id, uint64_t peer_data, void **host_addr, CUdeviceptr *peer_addr, size_t req_size, gds_memory_type_t mem_type, void **phandle);
int gds_peer_malloc_ex(int peer_id, uint64_t peer_data, void **host_addr, CUdeviceptr *peer_addr, size_t req_size, gds_memory_type_t mem_type, bool has_cpu_mapping, void **phandle);



