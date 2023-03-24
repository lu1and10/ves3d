#include <omp.h>
#include <iostream>
#include <SphericalHarmonics.h>

template<class Real>
StokesVelocity<Real>::StokesVelocity(int sh_order_, int sh_order_up_, Real box_size_):
  sh_order(sh_order_), sh_order_up_self(sh_order_up_), sh_order_up(sh_order_up_), box_size(box_size_), trg_is_surf(true)
{
  add_repul=false;
  fmm_setup=false;
}

template <class Real>
StokesVelocity<Real>::~StokesVelocity(){
}


template <class Real>
void StokesVelocity<Real>::SetSrcCoord(const SCTLVec& S, int sh_order_up_self_, int sh_order_up_){
  if(sh_order_up_self_>0) sh_order_up_self=sh_order_up_self_;
  if(sh_order_up_     >0) sh_order_up     =sh_order_up_     ;
  scoord.ReInit(S.Dim(), (Real*)&S[0], true);
  fmm_setup=false;

  SLMatrix.ReInit(0);
  DLMatrix.ReInit(0);

  scoord_far.ReInit(0);
  tcoord_repl.ReInit(0);
  scoord_norm.ReInit(0);
  scoord_area.ReInit(0);

  rforce_single.ReInit(0);
  qforce_single.ReInit(0);
  uforce_double.ReInit(0);
  qforce_double.ReInit(0);

  S_vel.ReInit(0);
  S_vel_up.ReInit(0);
  fmm_vel.ReInit(0);
  trg_vel.ReInit(0);
}

template <class Real>
template <class Vec>
void StokesVelocity<Real>::SetSrcCoord(const Vec& S, int sh_order_up_self_, int sh_order_up_){
  SCTLVec tmp(S.size(), (Real*)S.begin(), false);
  SetSrcCoord(tmp, sh_order_up_self_, sh_order_up_);
}

template <class Real>
void StokesVelocity<Real>::SetDensitySL(const SCTLVec* f, bool add_repul_){
  if(f){
    if(force_single.Dim()!=f->Dim()) fmm_setup=false;
    force_single.ReInit(f->Dim(), (Real*)&f[0][0], true);
  }else if(force_single.Dim()!=0){
    fmm_setup=false;
    force_single.ReInit(0);
  }

  rforce_single.ReInit(0);
  qforce_single.ReInit(0);
  add_repul=add_repul_;

  S_vel.ReInit(0);
  S_vel_up.ReInit(0);
  fmm_vel.ReInit(0);
  trg_vel.ReInit(0);
}

template <class Real>
template <class Vec>
void StokesVelocity<Real>::SetDensitySL(const Vec* f, bool add_repul_){
  if(f){
    SCTLVec tmp(f->size(), (Real*)f->begin(), false);
    SetDensitySL(&tmp, add_repul_);
  }else SetDensitySL((const SCTLVec*)NULL, add_repul_);
}

template <class Real>
void StokesVelocity<Real>::SetDensityDL(const SCTLVec* f){
  if(f){
    if(force_double.Dim()!=f->Dim()) fmm_setup=false;
    force_double.ReInit(f->Dim(), (Real*)&f[0][0], true);
  }else if(force_double.Dim()!=0){
    fmm_setup=false;
    force_double.ReInit(0);
  }

  uforce_double.ReInit(0);
  qforce_double.ReInit(0);

  S_vel.ReInit(0);
  S_vel_up.ReInit(0);
  fmm_vel.ReInit(0);
  trg_vel.ReInit(0);
}

template <class Real>
template <class Vec>
void StokesVelocity<Real>::SetDensityDL(const Vec* f){
  if(f){
    SCTLVec tmp(f->size(), (Real*)f->begin(), false);
    SetDensityDL(&tmp);
  }else SetDensityDL(NULL);
}

template <class Real>
void StokesVelocity<Real>::SetTrgCoord(const SCTLVec* T){
  if(T){
    trg_is_surf=false;
    tcoord.ReInit(T->Dim(),const_cast<Real*>(&T[0][0]));
  }else{
    trg_is_surf=true;
    tcoord.ReInit(0);
  }

  fmm_setup=false;
  S_vel.ReInit(0);
  S_vel_up.ReInit(0);
  fmm_vel.ReInit(0);
  trg_vel.ReInit(0);
}

template <class Real>
const typename StokesVelocity<Real>::SCTLVec& StokesVelocity<Real>::SelfInteraction(){
  setup_self();

  if(!S_vel.Dim()){ // Compute self interaction
    static SCTLVec Vcoef;
    { // Compute Vcoeff
      long Ncoef =   sh_order*(sh_order+2);
      long Ngrid = 2*sh_order*(sh_order+1);
      static SCTLVec SL_vel, DL_vel;
      SL_vel.ReInit(0);
      DL_vel.ReInit(0);

      if(rforce_single.Dim()){ // Set SL_vel
        static sctl::Vector<Real> F;
        SphericalHarmonics<Real>::Grid2SHC(rforce_single,sh_order,sh_order,F);

        long nv = rforce_single.Dim()/Ngrid/VES3D_DIM;
        SL_vel.ReInit(nv*VES3D_DIM*Ncoef);
        #pragma omp parallel
        { // mat-vec
          long tid=omp_get_thread_num();
          long omp_p=omp_get_num_threads();

          long a=(tid+0)*nv/omp_p;
          long b=(tid+1)*nv/omp_p;
          for(long i=a;i<b;i++){
            sctl::Matrix<Real> Mv(1,VES3D_DIM*Ncoef,&SL_vel[i*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real> Mf(1,VES3D_DIM*Ncoef,&F     [i*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real> M(VES3D_DIM*Ncoef,VES3D_DIM*Ncoef,&SLMatrix[i*VES3D_DIM*Ncoef*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real>::GEMM(Mv,Mf,M);
          }
        }
      }
      if(force_double.Dim()){ // Set DL_vel
        static sctl::Vector<Real> F;
        SphericalHarmonics<Real>::Grid2SHC(force_double,sh_order,sh_order,F);

        long nv = force_double.Dim()/Ngrid/VES3D_DIM;
        DL_vel.ReInit(nv*VES3D_DIM*Ncoef);
        #pragma omp parallel
        { // mat-vec
          long tid=omp_get_thread_num();
          long omp_p=omp_get_num_threads();

          long a=(tid+0)*nv/omp_p;
          long b=(tid+1)*nv/omp_p;
          for(long i=a;i<b;i++){
            sctl::Matrix<Real> Mv(1,VES3D_DIM*Ncoef,&DL_vel[i*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real> Mf(1,VES3D_DIM*Ncoef,&F     [i*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real> M(VES3D_DIM*Ncoef,VES3D_DIM*Ncoef,&DLMatrix[i*VES3D_DIM*Ncoef*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real>::GEMM(Mv,Mf,M);
          }
        }
      }
      if(SL_vel.Dim() && DL_vel.Dim()){ // Vcoef=SL_vel+DL_vel
        Vcoef.ReInit(SL_vel.Dim());
        #pragma omp parallel for
        for(long i=0;i<Vcoef.Dim();i++) Vcoef[i]=SL_vel[i]+DL_vel[i];
      }else{
        if(SL_vel.Dim()) Vcoef.ReInit(SL_vel.Dim(),&SL_vel[0]);
        else if(DL_vel.Dim()) Vcoef.ReInit(DL_vel.Dim(),&DL_vel[0]);
        else Vcoef.ReInit(0);
      }
    }
    SphericalHarmonics<Real>::SHC2Grid(Vcoef, sh_order, sh_order   , S_vel);
  }

  return S_vel;
}

template <class Real>
template <class Vec>
void StokesVelocity<Real>::SelfInteraction(Vec& vel){
  SCTLVec self_vel_(vel.size(),vel.begin(),false);

  const SCTLVec& self_vel_tmp=this->SelfInteraction();
  assert(self_vel_tmp.Dim()==self_vel_.Dim());
  self_vel_=self_vel_tmp;
}

template <class Real>
void StokesVelocity<Real>::setup_self(){

  if(!SLMatrix.Dim() || !DLMatrix.Dim()){
    if(!SLMatrix.Dim() && !DLMatrix.Dim() && force_single.Dim() && force_double.Dim()){
      if(1){
        sctl::Vector<Real> tmp1; tmp1.Swap(SLMatrix);
        sctl::Vector<Real> tmp2; tmp2.Swap(DLMatrix);
      }
      SphericalHarmonics<Real>::StokesSingularInteg(scoord, sh_order, sh_order_up_self, &SLMatrix, &DLMatrix);
    }else if(!SLMatrix.Dim() && force_single.Dim()){
      if(1){
        sctl::Vector<Real> tmp1; tmp1.Swap(SLMatrix);
      }
      SphericalHarmonics<Real>::StokesSingularInteg(scoord, sh_order, sh_order_up_self, &SLMatrix, NULL);
    }else if(!DLMatrix.Dim() && force_double.Dim()){
      if(1){
        sctl::Vector<Real> tmp2; tmp2.Swap(DLMatrix);
      }
      SphericalHarmonics<Real>::StokesSingularInteg(scoord, sh_order, sh_order_up_self, NULL, &DLMatrix);
    }
  }

  if(!rforce_single.Dim() && force_single.Dim()){
    rforce_single.ReInit(force_single.Dim(),&force_single[0],false);
  }

}

