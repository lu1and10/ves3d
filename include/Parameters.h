#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

#include "Enums.h"
#include "Error.h"
#include "Logger.h"
#include "anyoption.h"
#include "Streamable.h"
#include "ves3d_common.h"

#include <typeinfo>
#include <iostream>
#include <string>
#include <sstream>
#include <map>


typedef std::map<std::string,std::string> DictString_t;

template <typename T>
class Parameters : public Streamable
{
  public:
    Parameters();
    Parameters(int argc, char** argv);
    Parameters(std::istream &is, Format format);

    ~Parameters();

    //Surface
    int n_surfs;
    int sh_order;
    int filter_freq;
    int upsample_freq;

    T bending_modulus;
    T viscosity_contrast;
    T permeability_coeff;
    T excess_density;
    T diffusion_rate;
    T fg_pulling_force;
    T fg_detachment_rate;
    T fg_radius;
    T fg_drag_coeff;
    T centrosome_position[3];
    T mt_growth_velocity;
    T mt_nucleation_rate;
    T mt_catastrophe_rate;

    //Time stepper
    T time_horizon;
    T ts;
    T time_tol;
    int time_iter_max;
    bool time_adaptive;
    bool solve_for_velocity;
    bool pseudospectral;

    enum SolverScheme scheme;
    enum PrecondScheme time_precond;
    enum BgFlowType bg_flow;
    enum SingularStokesRot singular_stokes;

    //Reparametrization
    enum ReparamType rep_type;
    int  rep_maxit;
    int  rep_filter_freq;
    bool rep_upsample;
    T    rep_ts;
    T    rep_tol;
    T    rep_exponent;

    //Repulsion
    T    repul_dist;

    //Background flow
    T bg_flow_param;
    bool interaction_upsample;
    T periodic_length;
    T gravity_field[3];

    //Startup and Monitoring
    bool checkpoint;
    T checkpoint_stride;
    std::string write_vtk;
    std::string shape_gallery_file;
    std::string vesicle_props_file;
    std::string vesicle_geometry_file;
    std::string checkpoint_file_name;
    std::string load_checkpoint;
    T error_factor;
    int num_threads;

    //parsing
    Error_t parseInput(int argc, char** argv, const DictString_t *dict=NULL);
    Error_t expand_templates(const DictString_t *dict=NULL);

    //Adjust the frequencies according to the sh_order
    void adjustFreqs();

    // From streamable class --------------------------------------------------
    // ------------------------------------------------------------------------
    virtual Error_t pack(std::ostream &os, Format format) const;
    virtual Error_t unpack(std::istream &is, Format format);

  private:
    Parameters(Parameters<T> &rhs);
    Parameters<T>& operator=(Parameters<T> &rhs);

    void setUsage(AnyOption* opt);
    void setOptions(AnyOption* opt);
    void getOptionValues(AnyOption* opt);
    void init();
};

template<typename T>
std::ostream& operator<<(std::ostream& output, const Parameters<T>& par);

//! Expands the templates in pattern marked by their key in dict to
//! the value in dict. For example user_{{name}}} with dict
//! {name:'JohnDoe'} expands to user_JohnDoe.
Error_t expand_template(std::string *pattern, const DictString_t &dict);

#include "Parameters.cc"

#endif //_PARAMETERS_H_
