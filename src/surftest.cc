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
typedef OperatorsMats<Arr_t> Mats_t;

int main(int argc, char **argv)
{
    int const nv(1);

    //IO
    DataIO myIO;

    Parameters<real> sim_par;
    sim_par.sh_order = 16;
    sim_par.upsample_freq = 32;

    // initializing vesicle positions from text file
    VecCPU_t x0(nv, sim_par.sh_order);
    int fLen = x0.getStride();
    char fname[400];
    COUT("Loading initial shape");
    sprintf(fname, "/precomputed/ellipse_%d.txt",sim_par.sh_order);
    myIO.ReadData(FullPath(fname), x0, DataIO::ASCII, 0, fLen * DIM);

    //Reading operators from file
    COUT("Loading matrices");
    bool readFromFile = true;
    Mats_t mats(readFromFile, sim_par);

    //Creating objects
    COUT("Creating the surface object");
    Surface<ScaCPU_t, VecCPU_t> S(x0.getShOrder(),mats, &x0);

    return 0;
}
