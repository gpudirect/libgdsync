#pragma once

// lookup dev ptr, potentially registering it first with gds_register_mem
int gds_map_mem(void *_ptr, size_t size, gds_memory_type_t mem_type, CUdeviceptr *dev_ptr);
// 1st time registration of memory, for HOST and IO
int gds_register_mem(void *_ptr, size_t size, gds_memory_type_t type, CUdeviceptr *dev_ptr);
int gds_unregister_mem(void *_ptr, size_t size);

/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */
