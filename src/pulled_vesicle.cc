#include <iostream>
#include "DataIO.h"
#include "Scalars.h"
#include "Vectors.h"
#include "OperatorsMats.h"
#include "Surface.h"
#include "Parameters.h"
#include "vtu_writer.h"
#include "sctl.hpp"
#include "EvolveSurface.h"

typedef Device<CPU> DevCPU;
extern const DevCPU the_cpu_device(0);
typedef double real;

typedef Scalars<real, DevCPU, the_cpu_device> ScaCPU_t;
typedef Vectors<real, DevCPU, the_cpu_device> VecCPU_t;
typedef Surface<ScaCPU_t, VecCPU_t> Surf_t;
typedef typename ScaCPU_t::array_type Arr_t;
typedef OperatorsMats<Arr_t> Mats_t;   // object that stores various spharm/quadr matrices for 1 vesicle
typedef EvolveSurface<real, DevCPU, the_cpu_device> Evolve_t;

template<typename Container>
void set_zero(Container &x){
    x.getDevice().Memset(x.begin(), 0, sizeof(typename Container::value_type)*x.size());
}

template<typename Container>
void set_one(Container &x){
    x.getDevice().Memset(x.begin(), 0, sizeof(typename Container::value_type)*x.size());
    Arr_t a_tmp(x.getNumSubFuncs());
    for(auto i = a_tmp.begin(); i!=a_tmp.end(); ++i){
        *i = 1;
    }
    x.getDevice().apx(a_tmp.begin(),x.begin(),x.getStride(),x.getNumSubFuncs(),x.begin());
}

template<typename ScalarContainer, typename VectorContainer>
void set_exp(const VectorContainer &x, const real *x0, ScalarContainer &out){
  // This writes scalar function exp(-||x-x0||) into "out", at nodes "x".
  
    // store point -x0
    Arr_t a_tmp(x.getNumSubFuncs());
    for(int i=0; i<x.getNumSubFuncs(); i++){
        a_tmp.begin()[i] = -x0[i%3];
    }

    // init v_tmp for output of apx
    VectorContainer v_tmp;
    v_tmp.replicate(x);
    set_zero(v_tmp);

    // store x - x0 in v_tmp          (apx does "a plus x" where a is const vec)
    x.getDevice().apx(a_tmp.begin(),x.begin(),x.getStride(),x.getNumSubFuncs(),v_tmp.begin());
    // The point of doing this in an abstracted way is could be CPU or GPU, real/double.
    
    // init s_tmp
    ScalarContainer s_tmp;
    s_tmp.replicate(x);
    set_zero(s_tmp);

    // store (x - x0) dot (x - x0) in s_tmp
    GeometricDot(v_tmp, v_tmp, s_tmp);

    // store sqrt(s_tmp) in s_tmp
    Sqrt(s_tmp, s_tmp);

    // [maybe exp for ScalarContainer should be added to help functions? or not and use loop]
#pragma omp parallel for
    for(int idx = 0; idx < out.size(); idx++){
      // out.begin() is C++ vec style for 1st element. a[idx] is C-style
      out.begin()[idx] = std::exp(-s_tmp.begin()[idx]);
    }
}

template<typename ScalarContainer, typename VectorContainer>
void set_exp_simple(const VectorContainer &x, const real *x0, ScalarContainer &out){
  // This writes scalar function exp(-||x-x0||) into "out", at nodes "x".
  // Would not be abstracted (not on GPU)!  

  int N = out.size();         // # nodes on surf of vesicle
#pragma omp parallel for
    for(int idx = 0; idx < N; idx++){
      real d[3];  // displacement vector x[idx]-x0
      for (int i=0; i<3; i++)
        d[i] = x.begin()[idx+N*i] - x0[i];      // note a flattened array xxx...yyy..zzz
      real d2 = d[0]*d[0]+d[1]*d[1]+d[2]*d[2];      
      out.begin()[idx] = std::exp(-std::sqrt(d2));
    }
}

//////////////////////////////////////////// LIBIN's deforming surf test
void test_evolve_surface(){
    Parameters<real> sim_par;
    sim_par.sh_order = 16;        // sph harm
    sim_par.upsample_freq = 32;   // used in various geom calcs
    sim_par.filter_freq     = 24;
    sim_par.rep_filter_freq = 6;
    sim_par.n_surfs              = 1;
    sim_par.ts                   = 0.1; // 0.005;
    sim_par.time_horizon         = 50;
    sim_par.scheme               = JacobiBlockExplicit;
    sim_par.singular_stokes      = Direct;
    sim_par.bg_flow_param        = 0.0;
    sim_par.bg_flow              = ShearFlow;
    sim_par.diffusion_rate       = 0.1;
    sim_par.pulling_rate         = 0.3;        // f_0
    sim_par.pulling_eta          = 1.0;        // eta_m
    sim_par.centrosome_position[0] = 0.0;
    sim_par.centrosome_position[1] = 0.0;
    sim_par.centrosome_position[2] = 2.16;
    sim_par.interaction_upsample = true;
    sim_par.rep_maxit            = 20;
    sim_par.checkpoint           = true;
    sim_par.checkpoint_stride    = 0.5;
    sim_par.checkpoint_file_name = "EvolveSurf.chk";
    sim_par.write_vtk = "EvolveSurftest";

    //IO
    DataIO myIO;
    VecCPU_t x0(sim_par.n_surfs, sim_par.sh_order);    // quadr nodes defining shapes, vector (x,y,z) on (th,ph)-mesh
    int fLen = x0.getStride();          // SpharmGridDim is pair order+1, 2*order. fLen is their prod.
    // this is because x00,x01,...    y00,y01,...  z...   and use ptr to conriguous RAM for eg x0.
    char fname[400];
    COUT("Loading initial shape");
    sprintf(fname, "/precomputed/ellipse_%d.txt",sim_par.sh_order);
    myIO.ReadData(FullPath(fname), x0, DataIO::ASCII, 0, fLen * 3);

    //Reading operators from file
    COUT("Loading matrices");
    bool readFromFile = true;
    Mats_t mats(readFromFile, sim_par);       // reads relevant sized mats from precomputed/

    //Setting the background flow
    BgFlowBase<VecCPU_t> *vInf(NULL);
    CHK(BgFlowFactory(sim_par, &vInf));

    typename Evolve_t::Interaction_t interaction(&StokesAlltoAll);

    //Finally, Evolve surface
    COUT("Making EvolveSurface object");
    Evolve_t Es(&sim_par, mats, vInf, NULL, &interaction, NULL, NULL, &x0);
    // this object has a F_.density which is surface concentration c.

    // how set up density = exp(-||x-x0||) :
    real point0[3] = {0.0, 0.0, 2.16};
    set_zero(Es.F_->density_);
    set_exp(Es.S_->getPosition(), point0, Es.F_->density_);   // custom routine for this particular func
    //set_one(Es.F_->density_);
    set_one(Es.F_->binding_probability_);

    GaussLegendreIntegrator<ScaCPU_t> integrator;   // setup for surf int
    ScaCPU_t int_density;
    // set size
    int_density.replicate(Es.S_->getPosition());
    set_zero(int_density);
    integrator(Es.F_->density_, Es.S_->getAreaElement(), int_density);
    real mass_before = int_density.begin()[0];
    std::cout << "mass integral before: " << std::setprecision(8) << mass_before << "\n";
    
    Es.Evolve();          // do the sim  (using InterfacialVel  & InterfacialForce)

    set_zero(int_density);
    integrator(Es.F_->density_, Es.S_->getAreaElement(), int_density);
    real mass_after = int_density.begin()[0];
    std::cout << "mass integral after: " << std::setprecision(8) << mass_after << "\n";
    std::cout << "rel err in mass cons: " << std::setprecision(3) << (mass_after-mass_before)/mass_before << "\n";
}

int main(int argc, char **argv)
{
  test_evolve_surface();   // advect c with surface vel u
  return 0;
}
