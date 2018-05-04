/* =========================== flusher.hpp =========================== */
#define GDS_FLUSHER_TYPE_NONE 0
#define GDS_FLUSHER_TYPE_NATIVE 1
#define GDS_FLUSHER_TYPE_CPU 2
#define GDS_FLUSHER_TYPE_NIC 3

#define GDS_FLUSHER_OPS_NATIVE 0
#define GDS_FLUSHER_OPS_CPU 2
#define GDS_FLUSHER_OPS_NIC 5

typedef struct gds_flusher_buf {
    CUdeviceptr buf_d;
    void * buf_h;
    int size;
    gdr_mh_t mh;
    gds_mem_desc_t *desc;
    struct ibv_mr * mr;
} gds_flusher_buf;
typedef gds_flusher_buf * gds_flusher_buf_t;

typedef struct flusher_qp_info {
    struct gds_qp *loopback_qp;
    struct ibv_pd *pd;
    struct ibv_context *context;
    int gpu_id;
    struct ibv_ah * ah;
    char gid_string[INET6_ADDRSTRLEN];
    union ibv_gid gid_bin;
    int lid;
    int qpn;
    int psn;
    int ib_port;
} flusher_qp_info;
typedef flusher_qp_info * flusher_qp_info_t;


class Flusher {
	public:
		Flusher(){ }
		virtual ~Flusher()=0;

		/*
		 * Prepare flusher data structures, IB elements, etc..
		 *
		 * No flusher: nothing
		 * Native flusher: nothing
		 * CPU flusher:
		 *		- Allocate and pin GMEM memory area and flags
		 *		- Start CPU thread
		 *
		 * NIC flusher:
		 *		- Allocate and pin GMEM memory area and flags
		 *		- Create flusher QP recalling with special flags
		 *		- Prepare PUT buffer with IB elements
		 */
		virtual int setup(struct ibv_pd *pd, struct ibv_context *context, int gpu_id)=0;

		/*
		 * Used in case of SA model (i.e. post flusher ops on CUDA stream).
		 * In libgdsync this should be called in gds_post_ops()->IBV_EXP_PEER_OP_POLL_* 
		 * after gds_fill_poll();
		 *
		 * - No flusher  -> 0 ops
		 * - Native flusher ?? currently in libgdsync this is handled by gds_fill_poll()
		 * - CPU Flusher -> 1 write + 1 wait (2 ops)
		 * - NIC Flusher -> 1 write + 1 put (3 ops) + 1 wait (5 ops)
		*/
		virtual int post_ops(...);

		/*
		 * Used in case of KI model (i.e. flusher ops triggered by CUDA threads).
		 *
		 * - No flusher -> 0 ops
		 * - Native flusher
		 * - CPU Flusher -> ...
		 * - NIC Flusher -> ...
		*/
		virtual int prepare_ops(...);

		/*
		 * Used in case of KI model (i.e. flusher ops triggered by CUDA threads).
		 *
		 * This function triggers the flusher:
		 * - No flusher -> 0 ops
		 * - Native flusher
		 * - CPU Flusher -> write + wait on memory
		 * - NIC Flusher -> write + put + wait
		*/		
		__device__ virtual int flush();
}	

/*
 * fl_index depends on the env var GDS_FLUSHER_TYPE
 * that could assume a GDS_FLUSHER_TYPE_* value
 */
Flusher * getFlObj();
/*
Flusher *getFlObj()
{
	int gds_use_flusher=GDS_FLUSHER_TYPE_NONE;
        const char *env = getenv("GDS_FLUSHER_TYPE");
        if (env)
        {
            gds_use_flusher = atoi(env);
            if(
                gds_use_flusher != GDS_FLUSHER_TYPE_NONE &&
                gds_use_flusher != GDS_FLUSHER_TYPE_NATIVE &&
                gds_use_flusher != GDS_FLUSHER_TYPE_CPU &&
                gds_use_flusher != GDS_FLUSHER_TYPE_NIC
            )
            {
                gds_err("Erroneous flusher type=%d (allowed values 0-%d)\n", gds_use_flusher, GDS_FLUSHER_NIC);
                gds_use_flusher=GDS_FLUSHER_NONE;
            }
        }

        gds_warn("GDS_FLUSHER_TYPE=%d\n", gds_use_flusher);
    
        if((tl_array[gds_use_flusher])())
                return ((fl_array[gds_use_flusher])());

        return NULL;
}
*/

typedef Flusher*(*fl_creator)();
void add_flusher_creator(int id, fl_creator c);

/* =========================== flusher_native.hpp =========================== */
class FlusherNIC : public Flusher {
	protected:
		...
	public:
		...
}


/* =========================== flusher_cpu.hpp =========================== */
class FlusherCPU : public Flusher {
	protected:
		/*
		 * Create and start CPU thread
		 */
		int start_thread(pthread_t *fThread);
		/*
		 * Stop CPU thread (cancel + join)
		 */
		int stop_thread(pthread_t *fThread);
		/*
		 * CPU Thread function:
		 *	1. while( ack_signal_h <= last_ack_value_h) //waiting for GPU ack
		 *	2. last_ack_value_h = ack_signal_h
		 *	3. tmp = buf_h	// this dummy CPU read actually does the flush
		 *	4. ack_read_h = last_ack_value_h
		 *
		 * 	N.B. buf_h is GMEM mapped in HostMem by means of GDRCopy
		 */
		void * func_thread(void * arg);

	public:
		...
}


/* =========================== flusher_nic.hpp =========================== */
class FlusherNIC : public Flusher {
	protected:
		...
	public:
		...
}

