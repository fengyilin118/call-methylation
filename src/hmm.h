#ifndef HMM_H
#define HMM_H
#include "f5c.h"

#define HMT_NUM_MOVEMENT_TYPES 6

// Pre-computed transitions from the previous block
// into the current block of states. Log-scaled.


typedef struct { float x[HMT_NUM_MOVEMENT_TYPES]; } HMMUpdateScores;

void hmm_cuda(core_t *core, db_t* db, int batch_id);

void profile_hmm_score_cuda(core_t *core, db_t* db);



#endif