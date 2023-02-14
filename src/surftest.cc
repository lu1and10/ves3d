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
    
    // how set up density = exp(-||x-x0||) ?

    //check div_S grad_S u = 0

    // timestep ?

    // is mass conserved?
    
    return 0;
}
