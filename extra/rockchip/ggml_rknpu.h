

#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include <stdint.h>

#define DMA_HEAP_IOC_MAGIC		'H'
#define DMA_HEAP_IOCTL_ALLOC	_IOWR(DMA_HEAP_IOC_MAGIC, 0x0,\
				      struct dma_heap_allocation_data)

#define DMA_BUF_SYNC_READ      (1 << 0)
#define DMA_BUF_SYNC_WRITE     (2 << 0)
#define DMA_BUF_SYNC_RW        (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START     (0 << 2)
#define DMA_BUF_SYNC_END       (1 << 2)
#define DMA_BUF_BASE		'b'
struct ggml_sync_data_pack {
	uint64_t data;
};

#define DMA_BUF_IOCTL_SYNC	_IOW(DMA_BUF_BASE, 0, struct ggml_sync_data_pack)
#define CMA_HEAP_SIZE	(1024 * 1024)

#define GGML_RKNPU2_MAX_MATMUL_KERNELS 16


struct ggml_rknpu2_data_pack
{
    rknn_tensor_type type;
    void* ordered_data;
    int initialized;

    // RKNPU2 API structs
    rknn_tensor_mem* B;
};

struct ggml_rknpu2_matmul_kernel
{
    rknn_matmul_info matmul_info;
    rknn_matmul_ctx matmul_ctx;
    rknn_matmul_io_attr matmul_io_attr;

    rknn_tensor_mem* A;
    rknn_tensor_mem* C;
};

struct dma_heap_allocation_data {
	uint64_t len;
	uint32_t fd;
	uint32_t fd_flags;
	uint64_t heap_flags;
};
