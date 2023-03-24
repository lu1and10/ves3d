#ifndef _STOKES_VELOCITY_H_
#define _STOKES_VELOCITY_H_

//#include <mpi.h>
//#include "PVFMMInterface.h"
//#include "NearSingular.h"
//#include <matrix.hpp>
#include "sctl.hpp"

template <class Real>
class StokesVelocity{

    typedef typename sctl::Vector<Real> SCTLVec;

  public:

    StokesVelocity(int sh_order, int sh_order_up, Real box_size=-1);

    ~StokesVelocity();

    void SetSrcCoord(const SCTLVec& S, int sh_order_up_self_=-1, int sh_order_up_=-1);

    template<class Vec>
    void SetSrcCoord(const Vec& S, int sh_order_up_self_=-1, int sh_order_up_=-1);

    void SetDensitySL(const SCTLVec* force_single=NULL, bool add_repul=false);

    template<class Vec>
    void SetDensitySL(const Vec* force_single=NULL, bool add_repul=false);

    void SetDensityDL(const SCTLVec* force_double=NULL);

    template<class Vec>
    void SetDensityDL(const Vec* force_double=NULL);

    void SetTrgCoord(const SCTLVec* T);
    
    const SCTLVec& SelfInteraction();

    template<class Vec>
    void SelfInteraction(Vec& vel);

    void setup_self();
  private:
    int sh_order;
    int sh_order_up_self;
    int sh_order_up;
    Real box_size;
    //MPI_Comm comm;


    SCTLVec scoord;
    SCTLVec scoord_far;
    SCTLVec scoord_norm;
    SCTLVec scoord_area;

    SCTLVec force_single;
    SCTLVec rforce_single; // force_single + repulsion
    SCTLVec qforce_single; // upsample(rforce_single) * quadrature weights * area_element
    bool add_repul;

    SCTLVec force_double;
    SCTLVec uforce_double; // upsample(force_double)
    SCTLVec qforce_double; // uforce_double * quadrature weights * area_element + normal

    bool trg_is_surf;
    SCTLVec tcoord;
    SCTLVec tcoord_repl;
    SCTLVec trg_vel;


    // Self
    SCTLVec SLMatrix, DLMatrix;
    SCTLVec S_vel, S_vel_up;


    // Near
    //NearSingular<Real> near_singular0; // Surface-to-Surface interaction
    //NearSingular<Real> near_singular1; // Surface-to-Target interaction


    // Far
    bool fmm_setup;
    //void* pvfmm_ctx;
    SCTLVec fmm_vel;

};

#include "StokesVelocity.cc"

#endif // _STOKES_VELOCITY_H_
