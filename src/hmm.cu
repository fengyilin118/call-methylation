#include "logsum.h"
#include "hmm.h"
#include "f5c.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>

#define eslINFINITY INFINITY
#define PSR9_KMER_SKIP 0
#define PSR9_BAD_EVENT 1
#define PSR9_MATCH 2
#define  PSR9_NUM_STATES 3
#define HMT_FROM_SAME_M 0
#define HMT_FROM_PREV_M 1
#define HMT_FROM_SAME_B 2
#define HMT_FROM_PREV_B 3
#define HMT_FROM_PREV_K 4

#define HMT_FROM_SOFT 5
#define HMT_NUM_MOVEMENT_TYPES 6
#define HAF_ALLOW_PRE_CLIP 1
#define HAF_ALLOW_POST_CLIP 2
#define BAD_EVENT_PENALTY 0.0f
#define TRANS_START_TO_CLIP 0.5
#define TRANS_CLIP_SELF 0.9

#define MIN_SEPARATION 10
#define MIN_FLANK 10

#define MAX_EVENT_TO_BP_RATIO 20

#define METHYLATED_SYMBOL 'M'

const char* complement_dna = "TGCA";
const uint8_t rank_dna[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 

#define CUDA_CHK()                                                             \
    { gpu_assert(__FILE__, __LINE__); }

    static inline void gpu_assert(const char* file, uint64_t line) {
        cudaError_t code = cudaGetLastError();
        if (code != cudaSuccess) {
            fprintf(stdout, "[%s::ERROR]\033[1;31m Cuda error: %s \n in file : %s line number : %lu\033[0m\n",
                    __func__, cudaGetErrorString(code), file, line);
            if (code == cudaErrorLaunchTimeout) {
                ERROR("%s", "The kernel timed out. You have to first disable the cuda "
                            "time out.");
                fprintf(
                    stdout,
                    "On Ubuntu do the following\nOpen the file /etc/X11/xorg.conf\nYou "
                    "will have a section about your NVIDIA device. Add the following "
                    "line to it.\nOption \"Interactive\" \"0\"\nIf you do not have a "
                    "section about your NVIDIA device in /etc/X11/xorg.conf or you do "
                    "not have a file named /etc/X11/xorg.conf, run the command sudo "
                    "nvidia-xconfig to generate a xorg.conf file and do as above.\n\n");
            }
            exit(-1);
        }
    }
    
static inline uint64_t cuda_freemem(int32_t devicenum) {

    uint64_t freemem, total;
    cudaMemGetInfo(&freemem, &total);
    CUDA_CHK();
 //   fprintf(stdout, "[%s] (%lld) %.2f GB free of total %.2f GB GPU memory\n",__func__,freemem / double(1024 * 1024 * 1024), freemem,total / double(1024 * 1024 * 1024));
          
    return freemem;
}

static inline uint64_t tegra_freemem(int32_t devicenum) {

    uint64_t freemem, total;
    cudaMemGetInfo(&freemem, &total);
    CUDA_CHK();

    // RAM //from tegrastats
    FILE* f = fopen("/proc/meminfo", "r");
    int64_t totalRAMkB = -1, freeRAMkB = -1, memAvailablekB=-1, buffersRAMkB = -1, cachedRAMkB = -1;

    if(f)
    {
        // add if (blah) {} to get around compiler warning
        if (fscanf(f, "MemTotal: %ld kB\n", &totalRAMkB)) {}
        if (fscanf(f, "MemFree: %ld kB\n", &freeRAMkB)) {}
        if (fscanf(f, "MemAvailable: %ld kB\n", &memAvailablekB)) {}
        if (fscanf(f, "Buffers: %ld kB\n", &buffersRAMkB)) {}
        if (fscanf(f, "Cached: %ld kB\n", &cachedRAMkB)) {}
        fclose(f);
    }
    if(totalRAMkB>0 && freeRAMkB>0 && buffersRAMkB>0 && cachedRAMkB>0){
        freemem += (cachedRAMkB+buffersRAMkB)*1024;
    }
    else{
        WARNING("%s","Reading /proc/meminfo failed. Inferred free GPU memory might be wrong.");
    }

 //   fprintf(stdout, "[%s] %.2f GB free of total %.2f GB GPU memory\n",__func__,freemem / double(1024 * 1024 * 1024),total / double(1024 * 1024 * 1024));

    return freemem;
}




#define log_inv_sqrt_2pi  -0.918938f // Natural logarithm

__device__ float
log_normal_pdf(float x, float gp_mean, float gp_stdv, float gp_log_stdv) {
    /*INCOMPLETE*/
  //  float log_inv_sqrt_2pi = -0.918938f; // Natural logarithm
    float a = (x - gp_mean) / gp_stdv;
    return log_inv_sqrt_2pi - gp_log_stdv + (-0.5f * a * a);
    // return 1;
}

__device__ float
log_probability_match_r9(scalings_t scaling, model_t cpgmodel, event_t* event,
                         int event_idx, uint32_t kmer_rank) {
    
 //  assert(kmer_rank < 15625);

        float unscaledLevel = event[event_idx].mean;
        float scaledLevel = unscaledLevel;
        float gp_mean = scaling.scale * cpgmodel.level_mean + scaling.shift;
        float gp_stdv = cpgmodel.level_stdv * scaling.var;
        float gp_log_stdv = cpgmodel.level_log_stdv + scaling.log_var;
      
        float lp =  log_normal_pdf(scaledLevel, gp_mean, gp_stdv, gp_log_stdv);

    return lp;
}

__device__ double
add_logs(const double a, const double b)
{
    //return a+b;
    if(a == -INFINITY && b == -INFINITY)
        return -INFINITY;

   if(a > b) {
       double diff = b - a;
       return a + log(1.0 + exp(diff));
    } else {
        double diff = a - b;
        return b + log(1.0 + exp(diff));
}
}

static inline uint32_t get_rank(char base) {
    if (base == 'A') { //todo: do we neeed simple alpha?
        return 0;
    } else if (base == 'C') {
        return 1;
    } else if (base == 'G') {
        return 2;
    } else if (base == 'M') {
        return 3;
    } else if (base == 'T') {
        return 4;
    } else {
        WARNING("A None ACGMT base found : %c", base);
        return 0;
    }
}

// reverse-complement a DNA string
std::string reverse_complement(const std::string& str) {
    std::string out(str.length(), 'A');
    size_t i = 0;             // input
    int j = str.length() - 1; // output
    while (i < str.length()) {
        // complement a single base
        assert(str[i] != METHYLATED_SYMBOL);
        out[j--] = complement_dna[rank_dna[(int)str[i++]]];
    }
    return out;
}

// return the lexicographic rank of the kmer amongst all strings of
// length k for this alphabet
static inline uint32_t get_kmer_rank(const char* str, uint32_t k) {
    uint32_t p = 1;
    uint32_t r = 0;

    // from last base to first
    for (uint32_t i = 0; i < k; ++i) {
        //r += rank(str[k - i - 1]) * p;
        //p *= size();
        r += get_rank(str[k - i - 1]) * p;
        p *= 5;
    }
    return r;
}

struct RecognitionMatch
{
    unsigned offset; // the matched position in the recognition site
    unsigned length; // the length of the match, 0 indicates no match
    bool covers_methylated_site; // does the match cover an M base?
};

const uint32_t num_recognition_sites = 1;
const uint32_t recognition_length = 2;
const char* recognition_sites[] = { "CG" };
const char* recognition_sites_methylated[] = { "MG" };
const char* recognition_sites_methylated_complement[] = { "GM" };

// Check whether a recognition site starts at position i of str
inline RecognitionMatch match_to_site(const std::string& str, size_t i, const char* recognition, size_t rl)
{
    RecognitionMatch match;
    match.length = 0;
    match.offset = 0;
    match.covers_methylated_site = false;

    // Case 1: str is a substring of recognition
    const char* p = strstr(recognition, str.c_str());
    if(i == 0 && p != NULL) {
        match.offset = p - recognition;
        match.length = str.length();
    } else {
        // Case 2: the suffix str[i..n] is a prefix of recognition
        size_t cl = std::min(rl, str.length() - i);
        if(str.compare(i, cl, recognition, cl) == 0) {
            match.offset = 0;
            match.length = cl;
        }
    }

    //printf("Match site: %s %s %s %d %d\n", str.c_str(), str.substr(i).c_str(), recognition, match.offset, match.length);
    if(match.length > 0) {
        match.covers_methylated_site =
            str.substr(i, match.length).find_first_of(METHYLATED_SYMBOL) != std::string::npos;
    }

    return match;
}

// If the alphabet supports methylated bases, convert str
// to a methylated string using the recognition sites
std::string methylate(const std::string& str)
{
    std::string out(str);
    size_t i = 0;
    while(i < out.length()) {
        size_t stride = 1;

        // Does this location match a recognition site?
        for(size_t j = 0; j < num_recognition_sites; ++j) {

            RecognitionMatch match = match_to_site(str, i, recognition_sites[j], recognition_length);
            // Require the recognition site to be completely matched
            if(match.length == recognition_length) {
                // Replace by the methylated version
                out.replace(i, recognition_length, recognition_sites_methylated[j]);
                stride = match.length; // skip to end of match
                break;
            }
        }

        i += stride;
    }
    return out;
}

// reverse-complement a string meth aware
// when the string contains methylated bases, the methylation
// symbol transfered to the output strand in the appropriate position
std::string reverse_complement_meth(const std::string& str)
{
    std::string out(str.length(), 'A');
    size_t i = 0; // input
    int j = str.length() - 1; // output
    while(i < str.length()) {
        int recognition_index = -1;
        RecognitionMatch match;

        // Does this location (partially) match a methylated recognition site?
        for(size_t j = 0; j < num_recognition_sites; ++j) {
            match = match_to_site(str, i, recognition_sites_methylated[j], recognition_length);
            if(match.length > 0 && match.covers_methylated_site) {
                recognition_index = j;
                break;
            }
        }

        // If this subsequence matched a methylated recognition site,
        // copy the complement of the site to the output
        if(recognition_index != -1) {
            for(size_t k = match.offset; k < match.offset + match.length; ++k) {
                out[j--] = recognition_sites_methylated_complement[recognition_index][k];
                i += 1;
            }
        } else {
            // complement a single base
            assert(str[i] != METHYLATED_SYMBOL);
            //out[j--] = complement(str[i++]);
            out[j--] = complement_dna[rank_dna[(int)str[i++]]];
        }
    }
    return out;
}

__global__  void group_motif(ptr_t* read_ptr, char* read, uint32_t* cpg_sites, uint32_t* group_start, 
    uint32_t* group_end, uint32_t* group_size, uint32_t* num_site, uint32_t num_reads )
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  //  printf("%d, %d, %d\n", thread_id,read_idx[thread_id], has_events[thread_id]);

    if(thread_id < num_reads)
    {
    
            uint32_t site_index = 0;
           
             // Scan the sequence for CpGs
           for (ptr_t j = read_ptr[thread_id]; j < (read_ptr[thread_id+1]-1); j++)
            { 
                if(read[j] == 'C' && read[j+1] == 'G') {
                    cpg_sites[thread_id*1400 + site_index] = j-read_ptr[thread_id];
                    site_index++;
                }
            }
            num_site[thread_id] = site_index;

            
           // Batch the CpGs together into groups that are separated by some minimum distance
           int curr_idx = 0;
           int group_index = 0;
           while(curr_idx < site_index) 
           {
               int end_idx = curr_idx + 1;
               while(end_idx < site_index) {
                if(cpg_sites[thread_id*1400 + end_idx] - cpg_sites[thread_id*1400 + end_idx - 1] > MIN_SEPARATION)
                    break;
                end_idx += 1; 
               }
            group_start[thread_id * 1400+group_index] = curr_idx;
            group_end[thread_id * 1400+group_index] = end_idx;
            group_index++;
            curr_idx = end_idx;
           }

           group_size[thread_id] = group_index;
           
    }
}

__global__ void lookUpEvent(uint32_t* cpg_sites, uint32_t* group_start, uint32_t* group_end, uint32_t* group_size, 
    uint32_t* sub_start_pos, uint32_t* sub_end_pos, uint32_t* ref_start, ptr_t* alignment_ptr, AlignedPair* alignment,  uint32_t* num_rows, uint32_t* num_cols,
    uint32_t* event_start_idx, uint32_t* event_stop_idx,int8_t* event_stride, uint32_t* n_kmers, uint32_t* n_events,
     uint32_t num_reads, uint32_t kmer_size)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x; 

    uint32_t read_id = i/1400;
    uint32_t group_id = i%1400;
        
       if(group_id < group_size[read_id])
       { 

        uint32_t start_idx = group_start[i];
        uint32_t end_idx = group_end[i];

       if(start_idx>0||end_idx>0)
      {
        // the coordinates on the reference substring for this group of sites
        sub_start_pos[i] = cpg_sites[read_id * 1400 + start_idx] - MIN_FLANK;
        sub_end_pos[i] = cpg_sites[read_id * 1400 + end_idx - 1] + MIN_FLANK;
        uint32_t span = cpg_sites[read_id * 1400 + end_idx - 1] - cpg_sites[read_id * 1400 + start_idx];
       
        // skip if too close to the start of the read alignment or
        // if the reference range is too large to efficiently call
        event_start_idx[i] = -1;
        event_stop_idx[i] = -1;
      
         if(sub_start_pos[i]>2000000||sub_start_pos[i]<=0)
              sub_start_pos[i] = 0;

        
        if(sub_start_pos[i] > MIN_SEPARATION && span <= 200) 
        {


            uint32_t calling_start = sub_start_pos[i] + ref_start[read_id];
            uint32_t calling_end = sub_end_pos[i] + ref_start[read_id];
           
             ptr_t start_index = alignment_ptr[read_id];
             ptr_t end_index = alignment_ptr[read_id+1];
             int e1 = -1, e2 = -1;
             bool left_bounded, right_bounded;

        for(ptr_t i = start_index; i<end_index; i++)
        {
            if(alignment[i].ref_pos >= calling_start)
            {
                e1 = alignment[i].read_pos;
                break;
            }
        }
    
        for(ptr_t i = start_index; i<end_index; i++)
        {
            if(alignment[i].ref_pos >= calling_end)
            {
                e2 = alignment[i].read_pos;
                break;
            }
        }

        double ratio = fabs((double)(e2 - e1)) / (calling_start - calling_end);
        if (abs(e2 - e1) <= 10 || ratio > MAX_EVENT_TO_BP_RATIO) {
                 e1=-1;
                 e2=-1;
        }


     
         if(e1>=0&&e2>=0)
        { 
                event_start_idx[i] = e1;
                event_stop_idx[i] = e2;
        }
      else 
        {
              event_start_idx[i] = 0;
              event_stop_idx[i] = 0;
        }

       uint32_t length_m_seq = sub_end_pos[i] - sub_start_pos[i] + 1;
        uint32_t n_states;
        if(length_m_seq>0)
        {
            n_kmers[i] = length_m_seq- kmer_size + 1;
            n_states = PSR9_NUM_STATES * (n_kmers[i]  + 2); // + 2 for explicit terminal states
        }else{
           n_kmers[i]=0;
           n_states = 0;
        }

        if(event_stop_idx[i]  > event_start_idx[i])
            n_events[i] = event_stop_idx[i]  - event_start_idx[i] + 1;
         else
            n_events[i] = event_start_idx[i] - event_stop_idx[i]  + 1;
        if(n_events[i]>1000)
              n_events[i]=0;
  
        num_rows[i] =  n_events[i] + 1;
        num_cols[i] = n_states;
       event_stride[i] = event_start_idx[i] <= event_stop_idx[i] ? 1 : -1;

        }
      }

  }
}




__global__ void profile_initialize_kernel(uint32_t* group_size, float* matrix, ptr_t* matrix_ptr, uint32_t* num_rows, 
    uint32_t* num_cols, uint32_t n_group)
{

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
   
    if(thread_id < n_group)
    {
        uint32_t read_id = thread_id/1400;
        uint32_t group_id = thread_id%1400;

        if(group_id<group_size[read_id])
        {

           ptr_t m_ptr = matrix_ptr[thread_id];
           ptr_t n_col = num_cols[thread_id];
       
          for(ptr_t col = 0; col<num_cols[thread_id]; col++)
          {
            ptr_t ptr= m_ptr+col;
             matrix[ptr] = -INFINITY;
          }

        for(ptr_t row = 0; row<num_rows[thread_id]; row++)
        {
          matrix[m_ptr + n_col*row +PSR9_KMER_SKIP ] = -INFINITY;
            matrix[m_ptr + n_col*row +PSR9_BAD_EVENT ] = -INFINITY;
            matrix[m_ptr + n_col*row +PSR9_MATCH ] = -INFINITY;
        }
      }
    }
}

__global__ void calculate_transitions(uint32_t* group_size, BlockTransitions* transitions,  uint32_t* trans_ptr, double* events_per_base,int32_t n_group)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread_id < n_group)
    {
        uint32_t read_id = thread_id/1400;
        uint32_t group_id = thread_id%1400;

        if(group_id<group_size[read_id])
        { 
           for(uint32_t ki = trans_ptr[thread_id]; ki < trans_ptr[thread_id+1]; ki++) {
            float p_stay = 1 - (1 / events_per_base[thread_id/1400]);
            float p_skip = 0.0025;
            float p_bad = 0.001;
            float p_bad_self = p_bad;
            float p_skip_self = 0.3;
            
            float p_mm_next = 1.0f - p_stay - p_skip - p_bad;
            float p_bk, p_bm_next, p_bm_self;
            p_bk = p_bm_next = p_bm_self = (1.0f - p_bad_self) / 3;
            float p_km = 1.0f - p_skip_self;

            transitions[ki].lp_mk = log(p_skip);
            transitions[ki].lp_mb = log(p_bad);
            transitions[ki].lp_mm_self = log(p_stay);
            transitions[ki].lp_mm_next = log(p_mm_next);
    
            transitions[ki].lp_bb = log(p_bad_self);
            transitions[ki].lp_bk = log(p_bk);
            transitions[ki].lp_bm_next = log(p_bm_next);
            transitions[ki].lp_bm_self = log(p_bm_self);
    
            transitions[ki].lp_kk = log(p_skip_self);
            transitions[ki].lp_km = log(p_km);
           }
        }

    }
}

__global__ void flank_fill_kernel(uint32_t* group_size, float* pre_flank, ptr_t* pre_flank_ptr, float* post_flank, ptr_t* post_flank_ptr,
    uint32_t* event_start_idx, int8_t* event_stride, uint32_t* event_stop_idx, uint32_t n_group)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread_id < n_group)
    {
        uint32_t read_id = thread_id/1400;
        uint32_t group_id = thread_id%1400;

        if(group_id<group_size[read_id])
        {
        uint32_t pre_ptr = pre_flank_ptr[thread_id];
        pre_flank[pre_ptr] = log(1 - TRANS_START_TO_CLIP);
        pre_flank[pre_ptr+1] = log(TRANS_START_TO_CLIP) +  -3.0f + log(1 - TRANS_CLIP_SELF); 
        for(size_t i = pre_ptr+2; i < pre_flank_ptr[thread_id+1]; ++i) {
            pre_flank[i] = log(TRANS_CLIP_SELF) +
                           -3.0f + 
                           pre_flank[i - 1]; 
          }

        uint32_t post_ptr = post_flank_ptr[thread_id+1];
        post_flank[post_ptr - 1] = log(1 - TRANS_START_TO_CLIP);
        uint32_t num_events = post_flank_ptr[thread_id+1]-post_flank_ptr[thread_id];
        
        if(num_events>1)
          {
         uint32_t event_idx = event_start_idx[thread_id] + (num_events - 1) * event_stride[thread_id]; 
         //   assert(event_idx == event_stop_idx[thread_id]);
         //  if(event_idx != event_stop_idx[thread_id])
         //  {
               
              post_flank[post_ptr-2] = log(TRANS_START_TO_CLIP) + 
                                         -3.0f + 
                                         log(1 - TRANS_CLIP_SELF);
              for(int i = post_ptr - 3; i >= post_flank_ptr[thread_id]; --i) {
                    post_flank[i] = log(TRANS_CLIP_SELF) +
                            -3.0f + 
                            post_flank[i + 1]; 
                }
           // }
           }
        }
    }

}


__global__ void profile_fill_kernel(uint32_t* group_size, BlockTransitions* transitions, uint32_t* trans_ptr,float* matrix,ptr_t* matrix_ptr,
     uint32_t* num_rows, uint32_t* num_cols, ptr_t* kmer_ranks_ptr, uint32_t* kmer_ranks, uint32_t* event_start_idx, int8_t* event_stride, 
     scalings_t* scalings, model_t* cpgmodels, ptr_t* event_ptr, event_t* event_table, HMMUpdateScores* scores, uint32_t hmm_flags, float* pre_flank, 
     ptr_t* pre_flank_ptr, float* post_flank, ptr_t* post_flank_ptr, float* lp_end, uint32_t n_group)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(thread_id < n_group)
    {
        uint32_t read_id = thread_id/1400;
        uint32_t group_id = thread_id%1400;

        if(group_id<group_size[read_id])
        {
        uint32_t read_id = thread_id/1400;
        uint32_t num_col = num_cols[thread_id];
        uint32_t num_blocks = num_cols[thread_id]/PSR9_NUM_STATES;
        uint32_t last_event_row_idx = num_rows[thread_id] - 1;
        uint32_t last_kmer_idx = num_blocks - 3;
        float lp_sm, lp_ms;
        lp_sm = lp_ms = 0.0f;
        lp_end[thread_id] = -INFINITY;
       for(uint32_t row = 1; row < num_rows[thread_id]; row++)
       {
        for(uint32_t block = 1; block <num_blocks-1; block++) 
        {
            uint32_t kmer_idx = block - 1;
            BlockTransitions& bt = transitions[trans_ptr[thread_id]+kmer_idx];

            uint32_t prev_block = block - 1;
            uint32_t prev_block_offset = PSR9_NUM_STATES * prev_block;
            uint32_t curr_block_offset = PSR9_NUM_STATES * block;

            // Emission probabilities
            uint32_t event_idx = event_start_idx[thread_id] + (row - 1) * event_stride[thread_id];
            uint32_t kmer_rank = kmer_ranks[kmer_ranks_ptr[thread_id]+kmer_idx];

            scalings_t scaling = scalings[read_id];
            //--------------------------read_ptr?
          
            event_t* events = &event_table[event_ptr[read_id]];
            float lp_emission_m =0;
          //  printf("kmer_rank %d\n",kmer_rank);
            if(kmer_rank<15625)
                lp_emission_m = log_probability_match_r9(scaling, cpgmodels[kmer_rank], events, event_idx,kmer_rank);
        
            float lp_emission_b = BAD_EVENT_PENALTY;
            uint32_t m_ptr = matrix_ptr[thread_id];
            

            
            // state PSR9_MATCH
            scores[thread_id].x[HMT_FROM_SAME_M] = bt.lp_mm_self + matrix[m_ptr + num_col*(row-1) + (curr_block_offset + PSR9_MATCH)];
            scores[thread_id].x[HMT_FROM_PREV_M] = bt.lp_mm_next + matrix[m_ptr + num_col*(row-1) + (prev_block_offset + PSR9_MATCH)];
            scores[thread_id].x[HMT_FROM_SAME_B] = bt.lp_bm_self + matrix[m_ptr + num_col*(row-1)+ curr_block_offset + PSR9_BAD_EVENT];
            scores[thread_id].x[HMT_FROM_PREV_B] = bt.lp_bm_next + matrix[m_ptr+ num_col*(row-1) + prev_block_offset + PSR9_BAD_EVENT];
            scores[thread_id].x[HMT_FROM_PREV_K] = bt.lp_km + matrix[m_ptr+ num_col*(row-1) + prev_block_offset + PSR9_KMER_SKIP];

            scores[thread_id].x[HMT_FROM_SOFT] = (kmer_idx == 0 &&
                (event_idx == event_start_idx[thread_id] ||
                     (hmm_flags & HAF_ALLOW_PRE_CLIP))) ? lp_sm + pre_flank[pre_flank_ptr[thread_id]+(row - 1)] : -INFINITY;

            float sum = scores[thread_id].x[0];
            for(auto i = 1; i < HMT_NUM_MOVEMENT_TYPES; ++i) {
                sum = add_logs(sum, scores[thread_id].x[i]);
            }
           
            sum += lp_emission_m;
            matrix[m_ptr + num_col*row + curr_block_offset + PSR9_MATCH] = sum;

          

          

            // state PSR9_BAD_EVENT
            scores[thread_id].x[HMT_FROM_SAME_M] = bt.lp_mb + matrix[m_ptr +  num_col*(row-1) + (curr_block_offset + PSR9_MATCH)];  
            scores[thread_id].x[HMT_FROM_PREV_M] = -INFINITY;
            scores[thread_id].x[HMT_FROM_SAME_B] = bt.lp_bb + matrix[m_ptr + num_col*(row-1) + (curr_block_offset + PSR9_BAD_EVENT)];
            scores[thread_id].x[HMT_FROM_PREV_B] = -INFINITY;
            scores[thread_id].x[HMT_FROM_PREV_K] = -INFINITY;
            scores[thread_id].x[HMT_FROM_SOFT] = -INFINITY;

            sum = scores[thread_id].x[0];
            for(auto i = 1; i < HMT_NUM_MOVEMENT_TYPES; ++i) {
                sum = add_logs(sum, scores[thread_id].x[i]);
            }

            sum += lp_emission_b;
            matrix[m_ptr + num_col*row + curr_block_offset + PSR9_BAD_EVENT] = sum;
            //Zmatrix[m_ptr + num_col*(row-1) + (curr_block_offset + PSR9_BAD_EVENT)];
            


            // state PSR9_KMER_SKIP
            scores[thread_id].x[HMT_FROM_SAME_M] = -INFINITY;
            scores[thread_id].x[HMT_FROM_PREV_M] = bt.lp_mk + matrix[m_ptr + num_col*row + prev_block_offset + PSR9_MATCH];
            scores[thread_id].x[HMT_FROM_SAME_B] = -INFINITY;
            scores[thread_id].x[HMT_FROM_PREV_B] = bt.lp_bk + matrix[m_ptr + num_col*row + prev_block_offset + PSR9_BAD_EVENT];
            scores[thread_id].x[HMT_FROM_PREV_K] = bt.lp_kk + matrix[m_ptr + num_col*row + prev_block_offset + PSR9_KMER_SKIP];
            scores[thread_id].x[HMT_FROM_SOFT] = -INFINITY;

            sum = scores[thread_id].x[0];
            for(auto i = 1; i < HMT_NUM_MOVEMENT_TYPES; ++i) {
                sum = add_logs(sum, scores[thread_id].x[i]);
            }
            sum += 0.0f;
            matrix[m_ptr + num_col*row +  curr_block_offset + PSR9_KMER_SKIP] = sum;
            
      
            if(kmer_idx == last_kmer_idx && ( (hmm_flags & HAF_ALLOW_POST_CLIP) ||row == last_event_row_idx))
            {
                float lp1 = lp_ms + matrix[m_ptr + num_col*row + curr_block_offset + PSR9_MATCH]+ post_flank[post_flank_ptr[thread_id]+(row - 1)];
                float lp2 = lp_ms + matrix[m_ptr + num_col*row + curr_block_offset + PSR9_BAD_EVENT]+post_flank[post_flank_ptr[thread_id]+(row - 1)];
                float lp3 = lp_ms + matrix[m_ptr + num_col*row + curr_block_offset + PSR9_KMER_SKIP] + post_flank[post_flank_ptr[thread_id]+(row - 1)];
           
                lp_end[thread_id] = add_logs(lp_end[thread_id], lp1);
                lp_end[thread_id] = add_logs(lp_end[thread_id], lp2);
                lp_end[thread_id] = add_logs(lp_end[thread_id], lp3);
                
            }  
          }
        }
      }
    }

}

void hmm_cuda(core_t *core, db_t* db, int batch_id)
{
    uint32_t num_reads =  db->n_bam_rec;
    
    uint32_t num_groups = num_reads*1400;

    if(core->total_num_reads == 0)

   {
        core->host_read_ptr = (ptr_t*)malloc(sizeof(ptr_t)*(num_reads+1));
        core->host_alignment_ptr = (ptr_t*)malloc(sizeof(ptr_t) * (num_reads+1));
        core->host_ref_start_pos = (uint32_t*)malloc(sizeof(uint32_t) * num_reads); 
        core->host_event_ptr = (ptr_t*)malloc(sizeof(ptr_t)*num_reads);
        core->host_scalings = (scalings_t*)malloc(sizeof(scalings_t)*num_reads);
        core->bam_rec = (bam1_t**)malloc(sizeof(bam1_t*) * num_reads);
        core->qname = (char**)malloc(sizeof(char*)*num_reads);
      
   }
   else{
    core->host_read_ptr = (ptr_t*)realloc(core->host_read_ptr, sizeof(ptr_t)*(core->total_num_reads+num_reads+1));
    core->host_alignment_ptr = (ptr_t*)realloc(core->host_alignment_ptr , sizeof(ptr_t)*(core->total_num_reads+num_reads+1));
    core->host_ref_start_pos = (uint32_t*)realloc(core->host_ref_start_pos , sizeof(uint32_t)*(core->total_num_reads+num_reads));
    core->host_event_ptr = (ptr_t*)realloc(core->host_event_ptr , sizeof(ptr_t)*(core->total_num_reads+num_reads));
    core->host_scalings = (scalings_t*)realloc(core->host_scalings, sizeof(scalings_t)*(core->total_num_reads+num_reads));
    core->bam_rec = (bam1_t**)realloc(core->bam_rec,sizeof(bam1_t*)*(core->total_num_reads+num_reads));
   core->qname = (char**)realloc(core->qname, sizeof(char*)*(core->total_num_reads+num_reads));
   }

   uint32_t nreads_new = 0;

   for (uint32_t i = 0; i < num_reads; i++) {
  
    
    if(!db->read_stat_flag[i])
    {
        uint32_t j = nreads_new + core->total_num_reads;
        core->host_read_ptr[j] = core->sum_read_len;
        std::string ref_seq = db->fasta_cache[i]; 
        ref_seq = disambiguate(ref_seq); 
        core->ref_seq.push_back(ref_seq);
        core->sum_read_len += (ref_seq.size() + 1); //with null term
        
    core->host_alignment_ptr[j] = core->sum_alignment;
     AlignedPair *event_align_record = NULL;
     ptr_t event_align_record_size =
         get_event_alignment_record(db->bam_rec[i], db->read_len[i], db->base_to_event_map[i], &event_align_record, core->kmer_size);
  
    core->sum_alignment += event_align_record_size;
   
    core->host_ref_start_pos[j]=db->bam_rec[i]->core.pos;
    core->host_event_ptr[j] = core->sum_n_events;
    core->sum_n_events += db->et[i].n;  
    core->host_scalings[j]=db->scalings[i]; 
    core->bam_rec[j]= bam_init1();
    memcpy(core->bam_rec[j],db->bam_rec[i], 8);
    char* bam_qname = bam_get_qname(db->bam_rec[i]);
    core->qname[j]=(char*)malloc(sizeof(char)*40);
    strcpy(core->qname[j], bam_qname);
    nreads_new++;
    }      
}



   core->host_read_ptr[nreads_new+core->total_num_reads] = core->sum_read_len;
   core->host_alignment_ptr[nreads_new+core->total_num_reads] = core->sum_alignment;

   if(core->total_num_reads==0)
   {
       core->host_read = (char*)malloc(sizeof(char) * core->sum_read_len);
       core->host_alignment = (AlignedPair*)malloc(sizeof(AlignedPair) * core->sum_alignment);
       core->host_event_table = (event_t*)malloc(sizeof(event_t)* core->sum_n_events);
       core->host_rc = (uint8_t*)malloc(sizeof(uint8_t) * core->sum_read_len);
   }
   else
   {
       core->host_read = (char*)realloc(core->host_read, sizeof(char)*core->sum_read_len);
       core->host_alignment = (AlignedPair*)realloc(core->host_alignment, sizeof(AlignedPair)*core->sum_alignment);
       core->host_event_table = (event_t*)realloc(core->host_event_table, sizeof(event_t)*core->sum_n_events);
       core->host_rc = (uint8_t*)realloc(core->host_rc, sizeof(uint8_t)*core->sum_read_len);
   }

   nreads_new=0;
   for (uint32_t i = 0; i < num_reads; i++) {

 
    if(!db->read_stat_flag[i])
    {
    uint32_t j = nreads_new + core->total_num_reads;
    ptr_t idx = core->host_read_ptr[j];
    std::string ref_seq = db->fasta_cache[i]; 
    ref_seq = disambiguate(ref_seq); 
    strcpy(&core->host_read[idx], ref_seq.c_str());
    
    core->host_rc[j] = bam_is_rev(db->bam_rec[i]);
    AlignedPair *event_align_record = NULL;
    int32_t event_align_record_size =
        get_event_alignment_record(db->bam_rec[i], db->read_len[i], db->base_to_event_map[i], &event_align_record, core->kmer_size);
  
    ptr_t alignment_idx = core->host_alignment_ptr[j];
    
    memcpy(&core->host_alignment[alignment_idx], event_align_record, sizeof(AlignedPair) * event_align_record_size);

    idx = core->host_event_ptr[j];
    memcpy(&core->host_event_table[idx], db->et[i].event, sizeof(event_t) * db->et[i].n); 
    nreads_new++;
    } 
}

    core->total_num_reads += nreads_new;

    if(batch_id%3==2)
    {
   
          profile_hmm_score_cuda(core, db);
      
    }

}

void profile_hmm_score_cuda(core_t *core, db_t* db)
{
   
 
  //  fprintf(stdout,"meth_cuda start\n");
    
    uint32_t num_reads =  core->total_num_reads;
    uint32_t num_groups = num_reads*1400;
    //cuda data
    char* read;
    ptr_t* read_ptr; //index pointer for flattedned "reads"
    uint32_t* cpg_sites ;
    uint32_t* group_start ;
    uint32_t* group_end ;
    uint32_t* group_size ;
    uint32_t* sub_start_pos;
    uint32_t* sub_end_pos;

    ptr_t* alignment_ptr;
    AlignedPair* alignment;
    uint32_t* ref_start_pos;

    uint32_t* n_kmers;
    uint32_t* n_events;

    float* matrix;
    ptr_t* matrix_ptr;
    uint32_t* num_rows; 
    uint32_t* num_cols;
    uint32_t* trans_ptr;
    BlockTransitions* transitions;
    double* events_per_base;

    float* pre_flank;
    ptr_t* pre_flank_ptr;
    float* post_flank;
    ptr_t* post_flank_ptr;
    uint32_t* event_start_idx;
    uint32_t* event_stop_idx;
    int8_t* event_stride;
    ptr_t* kmer_ranks_ptr;
    uint32_t* kmer_ranks;
    uint32_t* mcpg_kmer_ranks;

    ptr_t* event_ptr;
    event_t* event_table;
    scalings_t* scalings;
    model_t* cpgmodels;

    HMMUpdateScores* scores;
    float* lp_end;
    float* mcpg_lp_end;


  //  fprintf(stdout,"num_reads %ld size * num_groups %ld sum_alignment %ld\n", num_reads, num_groups*sizeof(uint32_t),core->sum_alignment);

   /* for(int i=0;i<10;i++)
    {
         fprintf(stderr,"\n i %d\n",i);
        for(int ptr=core->host_read_ptr[i]; ptr<core->host_read_ptr[i+1]; ptr++)
       fprintf(stderr,"%c", core->host_read[ptr]);
    }*/

    cudaMalloc((void**)&read, core->sum_read_len*sizeof(char));
    CUDA_CHK();
    cudaMalloc((void **)&read_ptr, sizeof(ptr_t)*(num_reads+1));
    CUDA_CHK();
    cudaMalloc((void **)&cpg_sites, sizeof(uint32_t)*num_groups);
    CUDA_CHK();
    cudaMalloc((void **)&group_start, sizeof(uint32_t)*num_groups);
    CUDA_CHK();
    cudaMalloc((void **)&group_end, sizeof(uint32_t)*num_groups);
    CUDA_CHK();
    cudaMalloc((void **)&group_size, sizeof(uint32_t)*num_reads);
    CUDA_CHK();
    cudaMalloc((void **)&sub_start_pos, sizeof(uint32_t) * num_groups);
    CUDA_CHK();
    cudaMalloc((void **)&sub_end_pos, sizeof(uint32_t) * num_groups);
    CUDA_CHK();

    cudaMalloc((void **)&ref_start_pos,sizeof(uint32_t)*num_reads );
    CUDA_CHK();
    cudaMalloc((void **)&alignment_ptr,sizeof(ptr_t) * (num_reads+1));
    CUDA_CHK();
    cudaMalloc((void **)&alignment,sizeof(AlignedPair) * core->sum_alignment);
    CUDA_CHK();

    cudaMalloc((void**)&num_rows, num_groups*sizeof(uint32_t));
    CUDA_CHK();
    cudaMalloc((void**)&num_cols, num_groups*sizeof(uint32_t));
    CUDA_CHK();
    ptr_t allo_size =  num_groups*sizeof(uint32_t);
    cudaMalloc((void**)&event_start_idx, allo_size);
    CUDA_CHK();
    cudaMalloc((void**)&event_stop_idx, allo_size);
    CUDA_CHK();
    cudaMalloc((void**)&event_stride, sizeof(int8_t)*num_groups);
    CUDA_CHK();
    cudaMalloc((void**)&n_kmers, num_groups*sizeof(uint32_t));
    CUDA_CHK();
    cudaMalloc((void**)&n_events, num_groups*sizeof(uint32_t));
    CUDA_CHK();


  //  fprintf(stdout,"start cudaMemcpy\n");

    cudaMemcpy(read_ptr, core->host_read_ptr, sizeof(ptr_t)*(num_reads+1), cudaMemcpyHostToDevice);
    CUDA_CHK();
    cudaMemcpy(read, core->host_read, core->sum_read_len * sizeof(char), cudaMemcpyHostToDevice);
    CUDA_CHK();
    cudaMemcpy(ref_start_pos, core->host_ref_start_pos, sizeof(uint32_t)*num_reads, cudaMemcpyHostToDevice);
    CUDA_CHK();
    cudaMemcpy(alignment_ptr, core->host_alignment_ptr,  sizeof(ptr_t) * (num_reads+1), cudaMemcpyHostToDevice);
    CUDA_CHK();
    cudaMemcpy(alignment, core->host_alignment, sizeof(AlignedPair) * core->sum_alignment, cudaMemcpyHostToDevice);
    CUDA_CHK();

    uint32_t* num_site; 
    cudaMalloc((void **)&num_site, sizeof(uint32_t)*num_reads);
    CUDA_CHK();

    int threadPerBlock = 512; 
    int blockPerGrid = num_reads/threadPerBlock + 1;
    
    group_motif <<< blockPerGrid,  threadPerBlock >>> (read_ptr, read, cpg_sites, group_start, group_end, 
    group_size, num_site, num_reads);
    cudaDeviceSynchronize();
    CUDA_CHK();

    uint32_t* host_group_start = (uint32_t*)malloc(sizeof(uint32_t)*num_groups);
    MALLOC_CHK(host_group_start);
    cudaMemcpy(host_group_start, group_start, sizeof(uint32_t)*num_groups, cudaMemcpyDeviceToHost);
    CUDA_CHK();

    uint32_t* host_group_end = (uint32_t*)malloc(sizeof(uint32_t)*num_groups);
    MALLOC_CHK(host_group_end);
    cudaMemcpy(host_group_end, group_end, sizeof(uint32_t)*num_groups, cudaMemcpyDeviceToHost);
    CUDA_CHK();

    uint32_t* host_group_size = (uint32_t*)malloc(sizeof(uint32_t)*num_reads);
    MALLOC_CHK(host_group_size);
    cudaMemcpy(host_group_size, group_size, sizeof(uint32_t)*num_reads, cudaMemcpyDeviceToHost);
    CUDA_CHK();
    
/*
    for(int i =0;i<10;i++)
    {
        fprintf(stderr,"read %d\n",i);
       for(int group=0;group<host_group_size[i];group++)
          fprintf(stderr,"  start %d end %d\n", host_group_start[i * 1400+group], host_group_end[i * 1400+group]);
    }*/
    threadPerBlock = 1024;
    blockPerGrid = num_groups/threadPerBlock + 1;

   lookUpEvent <<< blockPerGrid, threadPerBlock >>>(cpg_sites, group_start, group_end, group_size, 
     sub_start_pos, sub_end_pos, ref_start_pos, alignment_ptr, alignment, num_rows, num_cols,
    event_start_idx, event_stop_idx, event_stride,  n_kmers, n_events,
     num_reads, core->kmer_size);
     cudaDeviceSynchronize();
     CUDA_CHK();
    uint32_t* host_sub_start_pos = (uint32_t*)malloc(sizeof(uint32_t) * num_groups); 
    uint32_t* host_sub_end_pos = (uint32_t*)malloc(sizeof(uint32_t) * num_groups);
    uint32_t* host_n_kmers = (uint32_t*)malloc(sizeof(uint32_t) * num_groups); 
    uint32_t* host_n_events = (uint32_t*)malloc(sizeof(uint32_t) * num_groups);
     uint32_t* host_num_cols = (uint32_t*)malloc(sizeof(uint32_t) * num_groups);
    uint32_t* host_event_start = (uint32_t*)malloc(sizeof(uint32_t)*num_groups);
    uint32_t* host_event_stop = (uint32_t*)malloc(sizeof(uint32_t)*num_groups);

    cudaMemcpy(host_n_kmers, n_kmers, num_groups*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    CUDA_CHK();
    cudaMemcpy(host_n_events, n_events, num_groups*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    CUDA_CHK();
    cudaMemcpy(host_sub_start_pos, sub_start_pos, num_groups*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    CUDA_CHK();
    cudaMemcpy(host_sub_end_pos, sub_end_pos, num_groups*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    CUDA_CHK();
    cudaMemcpy(host_num_cols, num_cols,num_groups*sizeof(uint32_t),cudaMemcpyDeviceToHost );
    CUDA_CHK();
    cudaMemcpy(host_event_start, event_start_idx, num_groups*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    CUDA_CHK();
    cudaMemcpy(host_event_stop, event_stop_idx, num_groups*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    CUDA_CHK();



   ptr_t* host_kmer_ranks_ptr = (ptr_t*)malloc(sizeof(ptr_t)*num_groups);
   uint32_t* host_kmer_ranks = (uint32_t*)malloc(sizeof(uint32_t)*num_groups*250);

 
   uint32_t* host_mcpg_kmer_ranks = (uint32_t*)malloc(sizeof(uint32_t)*num_groups*250);

   ptr_t sum_kmer_ranks = 0;

   for(uint32_t i = 0; i < num_groups; i++)
    {
        uint32_t read_id = i/1400;

        host_kmer_ranks_ptr[i] = sum_kmer_ranks;
        if(host_sub_start_pos[i] > MIN_SEPARATION && (host_sub_end_pos[i]- host_sub_start_pos[i]-2*MIN_FLANK)<200)
       { 
        std::string subseq = core->ref_seq[read_id].substr(host_sub_start_pos[i], host_sub_end_pos[i] - host_sub_start_pos[i] + 1);
        std::string rc_subseq = reverse_complement(subseq);
        // Methylate all CpGs in the sequence and score again
        std::string mcpg_subseq = methylate(subseq);
        std::string rc_mcpg_subseq = reverse_complement_meth(mcpg_subseq);

        const char* m_seq = subseq.c_str();
        const char* m_rc_seq = rc_subseq.c_str();

        const char* m_mcpg_seq = mcpg_subseq.c_str();
        const char* m_rc_mcpg_seq = rc_mcpg_subseq.c_str();
  
        uint32_t num_kmers = host_num_cols[i]/ PSR9_NUM_STATES - 2;
        int32_t seq_len = strlen(m_seq);
        int32_t mcpg_seq_len = strlen(m_mcpg_seq);
        

        for(size_t ki = 0; ki < num_kmers; ++ki){
            const char* substring = 0;
            const char* mcpg_substring = 0;
            if(core->host_rc[read_id]==0){
                substring=m_seq+ki;
                mcpg_substring = m_mcpg_seq+ki;
            }
            else{
                substring=m_rc_seq+seq_len-ki-core->kmer_size;
                mcpg_substring = m_rc_mcpg_seq + mcpg_seq_len -ki-core->kmer_size;
            }
          
            host_kmer_ranks[host_kmer_ranks_ptr[i]+ki] = get_kmer_rank(substring,core->kmer_size);
            host_mcpg_kmer_ranks[host_kmer_ranks_ptr[i]+ki] = get_kmer_rank(mcpg_substring, core->kmer_size);

         }
        sum_kmer_ranks+= num_kmers;
      }
    }

    ptr_t* host_matrix_ptr = (ptr_t*)malloc(sizeof(ptr_t)*num_groups);
 
    uint32_t* host_trans_ptr = (uint32_t*)malloc(sizeof(uint32_t)*(num_groups+1));

    ptr_t* host_pre_flank_ptr = (ptr_t*)malloc(sizeof(ptr_t)*(num_groups+1));
    ptr_t* host_post_flank_ptr = (ptr_t*)malloc(sizeof(ptr_t)*(num_groups+1));
    

    ptr_t matrix_size = 0;
    ptr_t num_kmers = 0;
    ptr_t pre_flank_size = 0;
    ptr_t post_flank_size = 0;
    uint32_t ngroup_new = 0;

    for(uint32_t r = 0; r<num_reads;r++)
    {

        for(uint32_t g = 0; g < host_group_size[r]; g++)
        {
            uint32_t p = 1400*r+g;
            uint32_t n_events = host_n_events[p];
            uint32_t n_kmers = host_n_kmers[p];

            uint32_t n_rows = n_events + 1;
            uint32_t n_states=0;
            if((n_kmers+core->kmer_size-1)>0)
                 n_states = PSR9_NUM_STATES * (n_kmers + 2);
            else
           {
            n_states = 0;
            n_kmers = 0;
           } 
    
        host_matrix_ptr[p]=matrix_size;
        matrix_size = matrix_size + n_rows*n_states;


        host_trans_ptr[p] = num_kmers;
        num_kmers += n_kmers;


        host_pre_flank_ptr[p] = pre_flank_size;
        host_post_flank_ptr[p] = post_flank_size;   
        pre_flank_size += n_events+1;
        post_flank_size += n_events;
        ngroup_new++;   
    }

    host_trans_ptr[1400*r+host_group_size[r]] = num_kmers;
    host_pre_flank_ptr[1400*r+host_group_size[r]] = pre_flank_size;
    host_post_flank_ptr[1400*r+host_group_size[r]] = post_flank_size; 

}
  
    host_trans_ptr[ngroup_new] = num_kmers;
    host_pre_flank_ptr[ngroup_new] = pre_flank_size;
    host_post_flank_ptr[ngroup_new] = post_flank_size;


     uint32_t hmm_flags = HAF_ALLOW_PRE_CLIP | HAF_ALLOW_POST_CLIP;
     
     cudaMalloc((void**)&kmer_ranks_ptr, sizeof(ptr_t)*num_groups);
     CUDA_CHK();

     cudaMalloc((void**)&kmer_ranks, sizeof(uint32_t)*num_groups*250);
     CUDA_CHK();

  /*  cudaMalloc((void**)&mcpg_kmer_ranks, sizeof(uint32_t)*num_groups*250);
     CUDA_CHK();
*/
     cudaMalloc((void**)&matrix, matrix_size*sizeof(float));
     CUDA_CHK();
 
     cudaMalloc((void**)&matrix_ptr, num_groups*sizeof(ptr_t));
     CUDA_CHK();

     cudaMalloc((void**)&trans_ptr, sizeof(uint32_t)*(num_groups+1));
     CUDA_CHK();

     uint32_t block_size = sizeof(BlockTransitions)*num_kmers;
     
     cudaMalloc((void**)&transitions, block_size);
     CUDA_CHK();
     
     cudaMalloc((void**)&events_per_base, sizeof(double)*num_reads);
     CUDA_CHK();
  
     cudaMalloc((void**)&pre_flank, sizeof(float)*pre_flank_size);
     CUDA_CHK();
     cudaMalloc((void**)&pre_flank_ptr, sizeof(ptr_t)*(num_groups+1));
     CUDA_CHK();
     cudaMalloc((void**)&post_flank, sizeof(float)*post_flank_size);
     CUDA_CHK();
     cudaMalloc((void**)&post_flank_ptr, sizeof(ptr_t)*(num_groups+1));
     CUDA_CHK();
 
     cudaMalloc((void**)&event_ptr, sizeof(ptr_t)*num_reads);
     CUDA_CHK();
     cudaMalloc((void**)&event_table, sizeof(event_t) * core->sum_n_events);
     CUDA_CHK();
     cudaMalloc((void**)&scalings, sizeof(scalings_t) *num_reads);
     CUDA_CHK();
     cudaMalloc((void**)&cpgmodels,MAX_NUM_KMER_METH * sizeof(model_t));
     CUDA_CHK();
 
     cudaMalloc((void**)&scores, sizeof(HMMUpdateScores)*num_groups);
     CUDA_CHK();
     cudaMalloc((void**)&lp_end, sizeof(float)*num_groups);
     CUDA_CHK();
     /*cudaMalloc((void**)&mcpg_lp_end, sizeof(float)*num_groups);
     CUDA_CHK();*/


     
//     fprintf(stdout,"after cudaMalloc\n");
     int32_t cuda_device_num = core->opt.cuda_dev_id;
     cudaDeviceProp prop;
     cudaGetDeviceProperties(&prop, cuda_device_num);
     uint64_t free_mem = 0;
     free_mem=cuda_freemem(cuda_device_num);

     cudaMemcpy(kmer_ranks_ptr, host_kmer_ranks_ptr, sizeof(ptr_t)*num_groups, cudaMemcpyHostToDevice);
     CUDA_CHK();

     cudaMemcpy(kmer_ranks, host_kmer_ranks, sizeof(uint32_t)*num_groups*250, cudaMemcpyHostToDevice);
     CUDA_CHK();
 

     cudaMemcpy(trans_ptr, host_trans_ptr, sizeof(uint32_t)*(num_groups+1), cudaMemcpyHostToDevice);
     CUDA_CHK();
    
     cudaMemcpy(events_per_base, db->events_per_base,  sizeof(double)*num_reads,  cudaMemcpyHostToDevice);
     CUDA_CHK();
 
     cudaMemcpy(pre_flank_ptr, host_pre_flank_ptr,  sizeof(ptr_t)*(num_groups+1),  cudaMemcpyHostToDevice);
     CUDA_CHK();
     
     cudaMemcpy(post_flank_ptr, host_post_flank_ptr,  sizeof(ptr_t)*(num_groups+1),  cudaMemcpyHostToDevice);
     CUDA_CHK();
  
     cudaMemcpy(event_ptr, core->host_event_ptr,  sizeof(ptr_t)*num_reads, cudaMemcpyHostToDevice);
     CUDA_CHK();
     cudaMemcpy(event_table, core->host_event_table, sizeof(event_t) * core->sum_n_events, cudaMemcpyHostToDevice);
     CUDA_CHK();
     cudaMemcpy(scalings, core->host_scalings, sizeof(scalings_t) * num_reads, cudaMemcpyHostToDevice);
     CUDA_CHK();
     cudaMemcpy(cpgmodels, core->cpgmodel, MAX_NUM_KMER_METH * sizeof(model_t), cudaMemcpyHostToDevice);
     CUDA_CHK();
     cudaMemcpy(matrix_ptr, host_matrix_ptr, num_groups * sizeof(ptr_t), cudaMemcpyHostToDevice);
     CUDA_CHK();
   //  fprintf(stdout,"cudaMemcpy end\n");


    threadPerBlock = 512; 
    blockPerGrid = num_groups/threadPerBlock + 1;
    profile_initialize_kernel<<<blockPerGrid,threadPerBlock>>>(group_size, matrix, matrix_ptr, num_rows, num_cols, num_groups);
    cudaDeviceSynchronize();
    CUDA_CHK();

 //   fprintf(stdout,"profile_initialize_kernel end\n");

    threadPerBlock = 512; 
    blockPerGrid = num_groups/threadPerBlock + 1;
    calculate_transitions<<<blockPerGrid, threadPerBlock>>>(group_size, transitions,  trans_ptr, events_per_base,num_groups);
    cudaDeviceSynchronize();
    CUDA_CHK();

 //   fprintf(stdout,"calculate_transitions end\n");


    threadPerBlock = 512; 
    blockPerGrid = num_groups/threadPerBlock + 1;
    flank_fill_kernel<<<blockPerGrid, threadPerBlock>>>(group_size, pre_flank,  pre_flank_ptr, post_flank,  post_flank_ptr,
     event_start_idx,  event_stride,  event_stop_idx,  num_groups);
    cudaDeviceSynchronize();
    CUDA_CHK();

    //fprintf(stdout,"flank_fill_kernel end\n");
  //  float* host_post_flank = (float*)malloc(sizeof(float)*post_flank_size);
   // cudaMemcpy(host_post_flank, post_flank, sizeof(float)*post_flank_size, cudaMemcpyDeviceToHost);


    threadPerBlock = 512; 
    blockPerGrid = num_groups/threadPerBlock + 1; 
    profile_fill_kernel<<<blockPerGrid, threadPerBlock>>>(group_size, transitions, trans_ptr,matrix,matrix_ptr, num_rows,  num_cols, 
    kmer_ranks_ptr,  kmer_ranks, event_start_idx, event_stride, scalings, cpgmodels, event_ptr, event_table,
     scores, hmm_flags, pre_flank, pre_flank_ptr, post_flank, post_flank_ptr,
    lp_end, num_groups);
   
    cudaDeviceSynchronize();
    CUDA_CHK();

  //  fprintf(stdout,"profile_fill_kernel end\n");

   core->unmethylated_score = (float*)malloc(sizeof(float)*num_groups);
   core->methylated_score = (float*)malloc(sizeof(float)*num_groups);
   uint32_t* host_cpg_sites = (uint32_t*)malloc(sizeof(uint32_t)*num_groups);
   core->site_score_map = (std::map<int, ScoredSite> **)malloc(sizeof(std::map<int, ScoredSite> *) * core->total_num_reads);
   for (int i = 0; i < core->total_num_reads; ++i) {
      core->site_score_map[i] = new std::map<int, ScoredSite>; 
   }
    cudaMemcpy(core->unmethylated_score, lp_end, num_groups*sizeof(float),cudaMemcpyDeviceToHost);
    CUDA_CHK();
 
    cudaMemcpy(kmer_ranks, host_mcpg_kmer_ranks, sizeof(uint32_t)*num_groups*250, cudaMemcpyHostToDevice);
    CUDA_CHK();
  
    threadPerBlock = 512; 
    blockPerGrid = num_groups/threadPerBlock + 1;
    profile_initialize_kernel<<<blockPerGrid,threadPerBlock>>>(group_size, matrix, matrix_ptr, num_rows, num_cols, num_groups);
    cudaDeviceSynchronize();
    CUDA_CHK();

    threadPerBlock = 512; 
    blockPerGrid = num_groups/threadPerBlock + 1; 
    profile_fill_kernel<<<blockPerGrid, threadPerBlock>>>(group_size, transitions, trans_ptr,matrix,matrix_ptr, num_rows,  num_cols, 
    kmer_ranks_ptr,  kmer_ranks, event_start_idx, event_stride, scalings, cpgmodels, event_ptr, event_table,
    scores, hmm_flags, pre_flank, pre_flank_ptr, post_flank, post_flank_ptr,
    lp_end, num_groups);
    cudaDeviceSynchronize();
    CUDA_CHK();

    cudaMemcpy(core->methylated_score, lp_end, num_groups*sizeof(float),cudaMemcpyDeviceToHost);
    CUDA_CHK();
    cudaMemcpy(host_cpg_sites, cpg_sites, num_groups*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    CUDA_CHK();
       
    for(int r=0;r<num_reads;r++)
    {
        fprintf(stderr,"read %d group_size %d\n", r, host_group_size[r]);
        for(int g=0;g<host_group_size[r];g++)
        {
            int i =r*1400+g;
            if(r<20)
                fprintf(stderr,"read %d group %d score %f %f\n", r, g, 
                core->methylated_score[i],core->unmethylated_score[i]);
        }
    }

    // Aggregate score
    for(int r=0;r<10;r++)
    {
        for(int g=0;g<host_group_size[r];g++)
        {
        int i =r*1400+g;
        uint32_t start_idx = host_group_start[i];
        uint32_t end_idx = host_group_end[i];
        if((start_idx>0||end_idx>0)&&host_event_start[i]>0&&host_event_stop[i]>0)
        {

            uint32_t start_position = host_cpg_sites[r*1400+start_idx] + core->host_ref_start_pos[i/1400];
            auto iter = core->site_score_map[i/1400]->find(start_position);
            if (iter == core->site_score_map[i/1400]->end()) {
            // insert new score into the map
            ScoredSite ss;
            //ss.chromosome = contig;
            ss.start_position = start_position;
            ss.end_position = host_cpg_sites[r*1400+end_idx - 1] + core->host_ref_start_pos[i/1400];
            ss.n_cpg = end_idx - start_idx;

             // extract the CpG site(s) with a k-mers worth of surrounding context
             size_t site_output_start = host_cpg_sites[r*1400+start_idx] - core->kmer_size + 1;
            size_t site_output_end = host_cpg_sites[r*1400+end_idx - 1] + core->kmer_size;

         //   fprintf(stdout,"\n output_start %d output_end %d \n",site_output_start, site_output_end);

           int k =0;
           for(int j=core->host_read_ptr[i/1400]+site_output_start;
             j<core->host_read_ptr[i/1400]+site_output_start+100&&j< core->host_read_ptr[i/1400]+site_output_end; j++)
           { 
         //      fprintf(stdout,"%c",core->host_read[j]);
               ss.sequence[k] = core->host_read[j];
               k++;
            }
            ss.sequence[k]=0;
        
         //   fprintf(stdout,"\n%s\n",ss.sequence);

             // insert into the map
           iter =
              core->site_score_map[i/1400]->insert(std::make_pair(start_position, ss)).first;

            
             // set strand-specific score
             // upon output below the strand scores will be summed
             int strand_idx=0;
             //iter->second.ll_methylated[strand_idx] = methylated_score;
             iter->second.ll_unmethylated[strand_idx] =  core->unmethylated_score[i];
             iter->second.ll_methylated[strand_idx] =  core->methylated_score[i];
            iter->second.strands_scored += 1;
          }
       }
    }
    }

      //cuda data
      cudaFree(read);
      cudaFree(read_ptr);
      cudaFree(cpg_sites);
      cudaFree(group_start);
      cudaFree(group_end);
      cudaFree(group_size);
      cudaFree(ref_start_pos);
      cudaFree(alignment_ptr);
      cudaFree(alignment);
      cudaFree(num_rows); 
      cudaFree(num_cols);
      cudaFree(event_start_idx);
      cudaFree(event_stop_idx);
      cudaFree(event_stride);
      cudaFree(n_kmers);
      cudaFree(n_events);

      
      cudaFree(matrix);
      cudaFree(matrix_ptr);
  
      cudaFree(trans_ptr);
      cudaFree(transitions);
      cudaFree(events_per_base);
  
      cudaFree(pre_flank);
      cudaFree(pre_flank_ptr);
      cudaFree(post_flank);
      cudaFree(post_flank_ptr);
      cudaFree(event_start_idx);
      cudaFree(kmer_ranks);
    //  cudaFree(mcpg_kmer_ranks);
      cudaFree(event_ptr);
      cudaFree(event_table);
      cudaFree(scalings);
      cudaFree(cpgmodels);
      cudaFree(scores);
      cudaFree(lp_end);
     // cudaFree(mcpg_lp_end);

    // core->total_num_reads = 0;
      core->sum_read_len = 0;
      core->sum_alignment = 0;
      core->sum_n_events = 0;

      free(core->host_read_ptr);
      free(core->host_alignment_ptr);
      free(core->host_ref_start_pos); 
      free(core->host_event_ptr); 
      free(core->host_read);  
      free(core->host_alignment); 
      free(core->host_event_table);
      free(core->host_scalings);

      free(core->unmethylated_score);
      free(core->methylated_score);
     // free(host_cpg_sites);
      free(host_group_start);
      free(host_group_end);

      core->ref_seq.clear();
    

}
