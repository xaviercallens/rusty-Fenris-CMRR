/* fenris-cmrr C API. Hand-written; must match src/ffi.rs. */
#ifndef CMRR_H
#define CMRR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CMRR_CONVERGED 0
#define CMRR_NOT_CONVERGED 1   /* time limit hit; coordinates were still updated */
#define CMRR_SOLVER_FAILED 2   /* coordinates untouched */
#define CMRR_TOO_LARGE 3       /* more free nodes than the dense CVODE limit (1500); untouched */
#define CMRR_INVALID_INPUT (-1)
#define CMRR_PANIC (-2)

#define CMRR_NODE_FIXED 0
#define CMRR_NODE_FREE 1

typedef struct CmrrReport {
    int32_t code;
    uint64_t unclump_steps;
    uint64_t cvode_steps;
    uint64_t rhs_evals;
    double max_free_force_rel;   /* final max |F| on free nodes / |V'(spacing)| */
    double t_final_rel;          /* final time / nearest-neighbour relaxation time */
    uint64_t n_jittered;         /* coincident free nodes displaced before solving */
    uint64_t n_outside_fixed_bbox; /* free nodes ending outside the fixed nodes' bounding box */
    double length_scale;         /* mean node spacing used for scaling, caller units */
    uint8_t cvode_order1_fallback; /* 1 if CVODE failed at max order and an order-1 restart finished the solve */
} CmrrReport;

/* Relax free nodes in place. x, y: num_nodes doubles each, non-overlapping. node_status: CMRR_NODE_FIXED or
 * CMRR_NODE_FREE per node. kappa: screening constant in inverse caller length units. report may be NULL.
 * The library borrows the arrays for the duration of the call only. */
int32_t cmrr_optimize_mesh(double *x, double *y, const int32_t *node_status, size_t num_nodes, double kappa,
                           CmrrReport *report);

#ifdef __cplusplus
}
#endif

#endif /* CMRR_H */
