#include <iostream>
#include "DataIO.h"
#include "Scalars.h"
#include "Vectors.h"
#include "OperatorsMats.h"
#include "Surface.h"
#include "Parameters.h"

#define DIM 3

typedef Device<CPU> DevCPU;
extern const DevCPU the_cpu_device(0);
typedef double real;

typedef Scalars<real, DevCPU, the_cpu_device> ScaCPU_t;
typedef Vectors<real, DevCPU, the_cpu_device> VecCPU_t;
typedef Surface<ScaCPU_t, VecCPU_t> Surf_t;
typedef typename ScaCPU_t::array_type Arr_t;
typedef OperatorsMats<Arr_t> Mats_t;   // object that stores various spharm/quadr matrices for 1 vesicle

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
    // store point -x0
    Arr_t a_tmp(x.getNumSubFuncs());
    for(int i=0; i<x.getNumSubFuncs(); i++){
        a_tmp.begin()[i] = -x0[i%3];
    }

    // init v_tmp
    VectorContainer v_tmp;
    v_tmp.replicate(x);
    set_zero(v_tmp);

    // store x - x0 in v_tmp
    x.getDevice().apx(a_tmp.begin(),x.begin(),x.getStride(),x.getNumSubFuncs(),v_tmp.begin());


    // init s_tmp
    ScalarContainer s_tmp;
    s_tmp.replicate(x);
    set_zero(s_tmp);

    // store (x - x0) dot (x - x0) in s_tmp
    GeometricDot(v_tmp, v_tmp, s_tmp);

    // store sqrt(s_tmp) in s_tmp
    Sqrt(s_tmp, s_tmp);

    // TODO: exp for ScalarContainer should be added to help functions
#pragma omp parallel for
    for(int idx = 0; idx < out.size(); idx++){
        out.begin()[idx] = std::exp(-s_tmp.begin()[idx]);
    }
}

template<typename ScalarContainer, typename VectorContainer>
void timestep(real dt, int n_step, VectorContainer &force, ScalarContainer &density, Surf_t &S){
    // temp work space
    ScalarContainer s_tmp; s_tmp.replicate(force); set_zero(s_tmp);
    VectorContainer v_tmp; v_tmp.replicate(force); set_zero(v_tmp);
    VectorContainer v_tmp2; v_tmp2.replicate(force); set_zero(v_tmp2);

    // map force to surface tangent space
    axpy(0.0, v_tmp, force, v_tmp);
    // false means calculation without upsample
    S.mapToTangentSpace(v_tmp, false);

    // update density explicitly
    for(int i=0; i<n_step; i++){
        // scalar field times vector field point wise
        xv(density, v_tmp, v_tmp2);
        S.div(v_tmp2, s_tmp);
        axpy(-dt, s_tmp, density, density);
    }
}

int main(int argc, char **argv)
{
  int const nv(1);    // # vesicles or particles

    //IO
    DataIO myIO;

    Parameters<real> sim_par;
    sim_par.sh_order = 16;        // sph harm
    sim_par.upsample_freq = 32;   // used in various geom calcs

    // initializing vesicle positions from text file
    VecCPU_t x0(nv, sim_par.sh_order);    // quadr nodes defining shapes, vector (x,y,z) on (th,ph)-mesh
    int fLen = x0.getStride();          // SpharmGridDim is pair order+1, 2*order. fLen is their prod.
    // this is because x00,x01,...    y00,y01,...  z...   and use ptr to conriguous RAM for eg x0.
    char fname[400];
    COUT("Loading initial shape");
    sprintf(fname, "/precomputed/ellipse_%d.txt",sim_par.sh_order);
    myIO.ReadData(FullPath(fname), x0, DataIO::ASCII, 0, fLen * DIM);

    //Reading operators from file
    COUT("Loading matrices");
    bool readFromFile = true;
    Mats_t mats(readFromFile, sim_par);       // reads relevant sized mats from precomputed/

    //Creating objects
    COUT("Creating the surface object");
    Surface<ScaCPU_t, VecCPU_t> S(x0.getShOrder(),mats, &x0);  // x0 stores coords

    // define some vector scalar field on surface
    ScaCPU_t density;
    VecCPU_t force;
    force.replicate(x0);                         // alloc force to have same sh_order as x0, data is 0
    density.replicate(x0);

    // some machinery
    S.div(force, density);          // see Surface.h
    xv(density, force, force);    // (in,in,out).   see HelperFUns.h : pointwise scalar-vector mult
    S.grad(density, force);

    // integrate over surf?
    GaussLegendreIntegrator<ScaCPU_t> integrator;
    // container to store integral result
    ScaCPU_t int_density;
    // set size
    int_density.replicate(x0);
    // init to int_density to zero
    set_zero(int_density);

    // init density field to constant one
    set_one(density);
    COUT(density);

    // integrate density over surf and store in int_density
    integrator(density, S.getAreaElement(), int_density);
    // print out content of int_density
    COUT(int_density);

    // how set up density = exp(-||x-x0||) ?
    real point0[3] = {0.0, 0.0, 2.16};
    set_zero(density);
    set_exp(S.getPosition(),point0,density);
    COUT(density);

    //check div_S grad_S u = 0
    ScaCPU_t s_tmp; s_tmp.replicate(x0); set_zero(s_tmp);
    VecCPU_t v_tmp; v_tmp.replicate(x0); set_zero(v_tmp);
    S.grad(density, v_tmp);
    S.div(v_tmp, s_tmp);
    COUT(s_tmp);

    // integrate density over surf and store in int_density
    integrator(density, S.getAreaElement(), int_density);
    // print out content of int_density
    COUT(int_density);
    COUT(density);

    // timestep ?
    set_one(force);
    real dt = 0.01;
    int n_step = 100;
    timestep(dt, n_step, force, density, S);

    // is mass conserved?
    // integrate density over surf and store in int_density
    integrator(density, S.getAreaElement(), int_density);
    // print out content of int_density
    COUT(int_density);
    COUT(density);

    return 0;
}
