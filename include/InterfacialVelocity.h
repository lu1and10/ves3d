#ifndef _INTERFACIALVELOCITY_H_
#define _INTERFACIALVELOCITY_H_

#include "Logger.h"
#include "Enums.h"
#include "Error.h"

#include "InterfacialForce.h"
#include "BiCGStab.h"
#include "SHTrans.h"
#include "Device.h"
#include <queue>
#include <memory>
#include "Enums.h"
#include "BgFlowBase.h"
#include "OperatorsMats.h"
#include "ParallelLinSolverInterface.h"
#include "VesicleProps.h"
#include "MovePole.h"
#include "StokesVelocity.h"
#include "PullingBoundary.h"

template<typename SurfContainer, typename Interaction>
class InterfacialVelocity
{
  public:
    typedef typename SurfContainer::value_type value_type;
    typedef typename SurfContainer::device_type device_type;
    typedef typename SurfContainer::Arr_t Arr_t;
    typedef typename SurfContainer::Sca_t Sca_t;
    typedef typename SurfContainer::Vec_t Vec_t;
    typedef OperatorsMats<Arr_t> Mats_t;
    typedef InterfacialVelocity Matvec_t;
    typedef ParallelLinSolver<value_type> PSolver_t;
    typedef typename PSolver_t::matvec_type POp_t;
    typedef typename PSolver_t::vec_type PVec_t;
    typedef StokesVelocity<value_type> Stokes_t;
    typedef SHTrans<Sca_t, SHTMats<value_type, device_type> > SHtrans_t;
    typedef VesicleProperties<Arr_t> VProp_t;

    InterfacialVelocity(SurfContainer &S_in, const Interaction &inter,
        const Mats_t &mats, const Parameters<value_type> &params,
        const VProp_t &ves_props, const BgFlowBase<Vec_t> &bgFlow,
        PSolver_t *parallel_solver=NULL);

    ~InterfacialVelocity();

    Error_t Prepare(const SolverScheme &scheme) const;
    Error_t BgFlow(Vec_t &bg, const value_type &dt) const;

    Error_t AssembleRhsVel(PVec_t *rhs, const value_type &dt, const SolverScheme &scheme) const;
    Error_t AssembleRhsPos(PVec_t *rhs, const value_type &dt, const SolverScheme &scheme) const;
    Error_t AssembleInitial(PVec_t *u0, const value_type &dt, const SolverScheme &scheme) const;
    Error_t ImplicitMatvecPhysical(Vec_t &vox, Sca_t &ten) const;

    Error_t Solve(const PVec_t *rhs, PVec_t *u0, const value_type &dt, const SolverScheme &scheme) const;
    Error_t ConfigureSolver(const SolverScheme &scheme) const;
    Error_t ConfigurePrecond(const PrecondScheme &precond) const;
    Error_t Update(PVec_t *u0);

    Error_t updateJacobiExplicit   (const SurfContainer& S_, const value_type &dt, Vec_t& dx);
    Error_t updateJacobiGaussSeidel(const SurfContainer& S_, const value_type &dt, Vec_t& dx);
    Error_t updateJacobiImplicit   (const SurfContainer& S_, const value_type &dt, Vec_t& dx);
    Error_t updateImplicit         (const SurfContainer& S_, const value_type &dt, Vec_t& dx);

    Error_t reparam();

    Error_t getTension(const Vec_t &vel_in, Sca_t &tension) const;
    Error_t stokes(const Vec_t &force, Vec_t &vel) const;
    Error_t stokes_double_layer(const Vec_t &force, Vec_t &vel) const;
    Error_t updateFarField() const;

    Error_t EvaluateFarInteraction(const Vec_t &src, const Vec_t &fi, Vec_t &vel) const;
    Error_t CallInteraction(const Vec_t &src, const Vec_t &den, Vec_t &pot) const;

    Error_t operator()(const Vec_t &x_new, Vec_t &time_mat_vec) const;
    Error_t operator()(const Sca_t &tension, Sca_t &tension_mat_vec) const;

    value_type StokesError(const Vec_t &x) const;

    Sca_t& tension(){ return tension_;}

    mutable Sca_t density_;
    mutable Sca_t binding_probability_;
    mutable Sca_t impingement_rate_;
    mutable Vec_t pulling_force_;
    mutable Vec_t pushing_force_;
    mutable Vec_t bending_force_;
    mutable Vec_t tensile_force_;
    mutable Vec_t flux_;
    mutable value_type* centrosome_pos_;
    mutable value_type* centrosome_vel_;
    mutable Vec_t pos_vel_;
    mutable Sca_t tension_;
    mutable Vec_t centrosome_pulling_;
    mutable Vec_t bdry_centrosome_pulling_;
    mutable Vec_t centrosome_pushing_;
    mutable value_type min_dist_;
    mutable value_type bdry_min_dist_;
    mutable PullingBoundary<SurfContainer> pulling_boundary_;
  private:
    SurfContainer &S_;
    const Interaction &interaction_;
    const BgFlowBase<Vec_t> &bg_flow_;
    const Parameters<value_type> &params_;
    const VProp_t &ves_props_;

    InterfacialForce<SurfContainer> Intfcl_force_;
    BiCGStab<Sca_t, InterfacialVelocity> linear_solver_;
    BiCGStab<Vec_t, InterfacialVelocity> linear_solver_vec_;

    // parallel solver
    PSolver_t *parallel_solver_;
    mutable bool psolver_configured_;
    mutable bool precond_configured_;
    mutable POp_t *parallel_matvec_;

    mutable PVec_t *parallel_rhs_;
    mutable PVec_t *parallel_u_;

    static Error_t ImplicitApply(const POp_t *o, const value_type *x, value_type *y);
    static Error_t ImplicitPrecond(const PSolver_t *ksp, const value_type *x, value_type *y);
    size_t stokesBlockSize() const;
    size_t tensionBlockSize() const;

    value_type dt_;

    Error_t EvalFarInter_Imp(const Vec_t &src, const Vec_t &fi, Vec_t &vel) const;
    Error_t EvalFarInter_ImpUpsample(const Vec_t &src, const Vec_t &fi, Vec_t &vel) const;

    //Operators
    Sca_t w_sph_, w_sph_inv_;
    Sca_t sing_quad_weights_;
    Sca_t quad_weights_;
    Sca_t quad_weights_up_;

    SHtrans_t sht_;
    SHtrans_t sht_upsample_;

    mutable Stokes_t stokes_;
    mutable MovePole<Sca_t, Mats_t> move_pole;
    mutable Sca_t position_precond;
    mutable Sca_t tension_precond;

    //Workspace
    mutable SurfContainer* S_up_;
    mutable std::queue<Sca_t*> scalar_work_q_;
    std::auto_ptr<Sca_t> checkoutSca() const;
    void recycle(std::auto_ptr<Sca_t> scp) const;
    mutable int checked_out_work_sca_;

    mutable std::queue<Vec_t*> vector_work_q_;
    std::auto_ptr<Vec_t> checkoutVec() const;
    void recycle(std::auto_ptr<Vec_t> vcp) const;
    mutable int checked_out_work_vec_;

    void purgeTheWorkSpace() const;
};

#include "InterfacialVelocity.cc"

#endif // _INTERFACIALVELOCITY_H_
