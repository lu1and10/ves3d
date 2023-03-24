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


template<typename ScalarContainer, typename VectorContainer>
void timestep(const real &dt, const int &n_step, const Surf_t &S, const VectorContainer &force, ScalarContainer &density){
  // n_step advection timesteps using "force" as fluid velocity; no source term.
  // fwd Euler, 1st-order
  
    // temp work space
    ScalarContainer s_tmp; s_tmp.replicate(force); set_zero(s_tmp);
    ScalarContainer s_tmp2; s_tmp2.replicate(force); set_zero(s_tmp2);
    VectorContainer v_tmp; v_tmp.replicate(force); set_zero(v_tmp);
    VectorContainer v_tmp2; v_tmp2.replicate(force); set_zero(v_tmp2);

    // diffusion constant
    real D = 3.0;

    // map force to surface tangent space
    axpy(0.0, v_tmp, force, v_tmp);               // easy way to memcpy of force -> v_tmp
    // here false means calculation without upsample...
    S.mapToTangentSpace(v_tmp, false);            // overwrites v_tmp

    std::string filename_prefix = "timestep";
    // update density explicitly
    for(int i=0; i<n_step; i++){         // time steps
      char suffix[7];
      sprintf(suffix, "%06d", i);
      std::string filename;
      filename.append(filename_prefix);
      filename.append(std::string(suffix));


      // scalar field times vector field point wise
      xv(density, v_tmp, v_tmp2);      // v_tmp2 = density.v_tmp   (pointwise)
      test_vtu_surface_writer(S.getPosition(), v_tmp2, density, filename);
      filename.append(".vtu");
      test_vtu_point_cloud_writer(S.getPosition(), force, v_tmp, density, filename);
      // surface div
      S.div(v_tmp2, s_tmp);           // s_tmp = div_Gamma (density force_tang)
      // diffusion
      S.grad(density, v_tmp2);
      S.div(v_tmp2, s_tmp2);
      // advect update
      axpy(-dt, s_tmp, density, density);  // density -= dt * s_tmp
      // diffusion update
      //axpy(dt*D, s_tmp2, density, density);  // density += dt * s_tmp2
    }

}

template<typename ScalarContainer, typename VectorContainer>
void test_vtu_point_cloud_writer(const VectorContainer &X, const VectorContainer &v1, const VectorContainer &v2, const ScalarContainer &s, const std::string &filename){
    int N = s.size();         // # nodes on surf of vesicle
    const int dim = 3;
    vtu_writer::VTUWriter writer;

    std::vector<double> points(X.size(), 0.0);
    std::vector<double> vector_field1(v1.size(), 0.0);
    std::vector<double> vector_field2(v2.size(), 0.0);
    std::vector<double> scalar_field(s.size(), 0.0);

    for(int idx=0; idx<N; idx++){
        for(int idim=0; idim<dim; idim++){
            points[dim*idx + idim] = X.begin()[idx+N*idim];
            vector_field1[dim*idx + idim] = v1.begin()[idx+N*idim];
            vector_field2[dim*idx + idim] = v2.begin()[idx+N*idim];
        }
        scalar_field[idx] = s.begin()[idx];
    }


    writer.add_scalar_field("scalar_field", scalar_field);
    writer.add_vector_field("vector_field1", vector_field1, dim);
    writer.add_vector_field("vector_field2", vector_field2, dim);

    writer.write_point_cloud(filename, dim,  points);
}

template<typename ScalarContainer, typename VectorContainer>
void test_vtu_surface_writer(const VectorContainer &X, const VectorContainer &v, const ScalarContainer &s, const std::string &filename){
    // test write vector field

    char* cstr = const_cast<char*>(filename.c_str());

    // sh order
    int p0 = X.getShOrder();
    typedef sctl::Vector<typename VectorContainer::value_type> SCTL_Vec;

    SCTL_Vec Xgrid(X.size(), (real*)X.begin(), false);
    SCTL_Vec Xgrid_shc;
    sctl::SphericalHarmonics<real>::Grid2SHC(Xgrid, p0+1, 2*p0, p0, Xgrid_shc, sctl::SHCArrange::ROW_MAJOR);

    SCTL_Vec V(v.size(), (real*)v.begin(), false);
    SCTL_Vec V_shc;
    sctl::SphericalHarmonics<real>::Grid2SHC(V, p0+1, 2*p0, p0, V_shc, sctl::SHCArrange::ROW_MAJOR);

    sctl::SphericalHarmonics<real>::WriteVTK(cstr, &Xgrid_shc, &V_shc, sctl::SHCArrange::ROW_MAJOR, p0, 2*p0);

}

void test_evolve_surface(){
    Parameters<real> sim_par;
    sim_par.sh_order = 16;        // sph harm
    sim_par.upsample_freq = 32;   // used in various geom calcs
    sim_par.filter_freq     = 24;
    sim_par.rep_filter_freq = 6;
    sim_par.n_surfs              = 1;
    sim_par.ts                   = 0.005;
    sim_par.time_horizon         = 1.;
    sim_par.scheme               = JacobiBlockExplicit;
    sim_par.singular_stokes      = Direct;
    sim_par.bg_flow_param        = 0.1;
    sim_par.bg_flow              = ShearFlow;
    sim_par.interaction_upsample = true;
    sim_par.rep_maxit            = 20;
    sim_par.checkpoint           = true;
    sim_par.checkpoint_stride    = 0.1;
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

    // how set up density = exp(-||x-x0||) :
    real point0[3] = {0.0, 0.0, 2.16};
    //set_zero(Es.F_->density_);
    //set_exp(Es.S_->getPosition(), point0, Es.F_->density_);   // custom routine for this particular func
    set_one(Es.F_->density_);

    GaussLegendreIntegrator<ScaCPU_t> integrator;
    ScaCPU_t int_density;
    // set size
    int_density.replicate(Es.S_->getPosition());
    // init to int_density to zero
    set_zero(int_density);
    integrator(Es.F_->density_, Es.S_->getAreaElement(), int_density);
    // print out content of int_density
    COUT(int_density);


    Es.Evolve();
    set_zero(int_density);
    integrator(Es.F_->density_, Es.S_->getAreaElement(), int_density);
    COUT(int_density);
}

void test_utils(){
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
    myIO.ReadData(FullPath(fname), x0, DataIO::ASCII, 0, fLen * 3);

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

    // how to integrate over surf:
    GaussLegendreIntegrator<ScaCPU_t> integrator;
    // container to store integral result (one number, same type)
    ScaCPU_t int_density;
    // set size
    int_density.replicate(x0);
    // init to int_density to zero
    set_zero(int_density);

    // init density field to constant one
    set_one(density);
    COUT(density);

    // integrate density over surf and store in int_density[0] (a slight waste to use whole array, part of legacy *** maybe change)
    integrator(density, S.getAreaElement(), int_density);
    // print out content of int_density
    COUT(int_density);

    // how set up density = exp(-||x-x0||) :
    real point0[3] = {0.0, 0.0, 2.16};
    set_zero(density);
    set_exp(S.getPosition(),point0,density);   // custom routine for this particular func
    // set_exp_simple(S.getPosition(),point0,density);   // custom routine for this particular func, dumb version
    COUT(density);

    //check div_S grad_S u = 0
    ScaCPU_t s_tmp; s_tmp.replicate(x0); set_zero(s_tmp);
    VecCPU_t v_tmp; v_tmp.replicate(x0); set_zero(v_tmp);
    S.grad(density, v_tmp);
    S.div(v_tmp, s_tmp);
    COUT(s_tmp);                // surf Laplacian (density)

    // MASS CONSERVATION TEST (ADVECTION ONLY) --------------------
    // integrate density over surf and store in int_density
    integrator(density, S.getAreaElement(), int_density);
    real mass_before = int_density.begin()[0];
    // print out content of int_density
    std::cout << "mass integral before: " << std::setprecision(8) << mass_before << "\n";
    //COUT(int_density);
    //COUT(density);

    // timestep:
    set_one(force);     // vec field = (1,1,1) at all nodes .. is not tangential
    real dt = 0.01;
    int n_step = 100;
    timestep(dt, n_step, S, force, density);       // changes density

    // is mass conserved?
    // integrate density over surf and store in int_density
    integrator(density, S.getAreaElement(), int_density);
    // print out content of int_density
    std::cout << "mass integral after: " << std::setprecision(8) << int_density.begin()[0] << "\n";
    std::cout << "mass error: " << std::setprecision(8) << std::abs(mass_before-int_density.begin()[0]) << "\n";

}

int main(int argc, char **argv)
{
    test_evolve_surface();
    return 0;
}
