#include "IntersectionCheck.h"

template<typename SurfContainer>
PullingBoundary<SurfContainer>::
PullingBoundary(const Params_t &params, const Mats_t &mats) :
    params_(params)
{
    // init PullingBoundary surface, sphere at (0,0,0)
    DataIO io;
    std::string fname("precomputed/sphere_"+std::to_string(params_.upsample_freq)+".txt");
    INFO("Reading PullingBoundary from file: "<<fname);
    std::vector<value_type> sphere_xyz;
    io.ReadDataStl(fname, sphere_xyz, DataIO::ASCII);
    Vec_t x0(1, params_.upsample_freq);
    #pragma omp parallel for
    for (int i=0; i<x0.size(); i++) {
        x0.begin()[i] = sphere_xyz[i] * params_.boundary_radius;
    }
    S_ = new SurfContainer(params_.upsample_freq, mats, &x0, params_.filter_freq*params_.upsample_freq/params_.sh_order, params_.rep_filter_freq, params_.rep_type, params_.rep_exponent);

    // set member variables memory size 
    Fpull_.replicate(x0);
    binding_probability_.replicate(x0);
    impingement_rate_.replicate(x0);
    visible_zone_.replicate(x0);

    // init member variables
    binding_probability_.getDevice().Memset(binding_probability_.begin(), 0, sizeof(value_type)*binding_probability_.size());
    impingement_rate_.getDevice().Memset(impingement_rate_.begin(), 0, sizeof(value_type)*impingement_rate_.size());
    Fpull_.getDevice().Memset(Fpull_.begin(), 0, sizeof(value_type)*Fpull_.size());
    visible_zone_.getDevice().Memset(visible_zone_.begin(), 0, sizeof(value_type)*visible_zone_.size());

    // calculate pulling boundary area
    Sca_t area;
    area.resize(1,1);
    S_->area(area);
    area_ = area.begin()[0];
    INFO("boundary area: "<<area_);

    // calculate pulling boundary M0_(max number of force generator)
    M0_ = area_/(M_PI*params_.fg_radius*params_.fg_radius);
    INFO("boundary M0: "<<M0_);
}

template<typename SurfContainer>
PullingBoundary<SurfContainer>::
~PullingBoundary()
{
    // clean SurfContainer
    delete S_;
}

// solve for double layer density
template<typename SurfContainer>
void PullingBoundary<SurfContainer>::
Solve()
{

}

template<typename SurfContainer>
void PullingBoundary<SurfContainer>::
GetCentrosomePulling(const value_type* centrosome_position, const value_type* centrosome_velocity, const Vec_t &vesicle_position, Vec_t *Fcpull, value_type *min_dist)
{
    // compute visible_zone_
    GetVisibleZone(centrosome_position, vesicle_position);

    // init work space
    // TODO: Pre-allocate
    Sca_t s_wrk[2];
    s_wrk[0].replicate(S_->getPosition());
    s_wrk[1].replicate(S_->getPosition());
    Vec_t v_wrk[2];
    v_wrk[0].replicate(S_->getPosition());
    v_wrk[1].replicate(S_->getPosition());

    // pulling force direction
    // TODO: FIXME
    // need to put this loop(each 3D-vector of vector field minus const 3D-vector) to help functions
    int N = Fpull_.size()/VES3D_DIM;
    for(int idim=0; idim<VES3D_DIM; idim++){
      #pragma omp parallel for
      for(int i=0; i<N; i++){
        Fpull_.begin()[idim*N+i] = centrosome_position[idim] - S_->getPosition().begin()[idim*N+i];
      }
    }
    // TODO: \hat\xi here has different sign compared to \hat\xi in the writeup
    // normalizing  to get \hat\xi
    GeometricDot(Fpull_, Fpull_, s_wrk[0]);
    // s_wrk[0] stores D, distance between centrosome and membrane surface points
    Sqrt(s_wrk[0], s_wrk[0]);
    uyInv(Fpull_, s_wrk[0], Fpull_);
    *min_dist = MinAbs(s_wrk[0]);
    // at this point
    // Fpull_ stores the unit direction of pullingForce
    // s_wrk[0] stores the distance between centrosome and membrane points ,i.e., D

    // calculate impingement_rate_ without geometric factor for pulling force
    axpy(-params_.mt_catastrophe_rate/params_.mt_growth_velocity, s_wrk[0], s_wrk[1]); // s_wrk[1] = -k_cat/V_g * D
    Exp(s_wrk[1], s_wrk[1]); // s_wrk[1] = exp(-k_cat/V_g*D)
    axpy(params_.mt_nucleation_rate/params_.mt_growth_velocity, s_wrk[1], s_wrk[1]); // s_wrk[1] = \dot{n}/V_g * exp(-k_cat/V_g*D)
    xv(s_wrk[1], S_->getNormal(), v_wrk[0]); // v_wrk[0] = \dot{n}/V_g * exp(-k_cat/V_g*D) * n
    axpy(-params_.mt_growth_velocity, Fpull_, v_wrk[1]); // v_wrk[1] = -V_g * \xi
    // FIXME: subtract the last time step centrosome velocity from v_wrk[1]
    // need to put this loop(each 3D-vector of vector field minus const 3D-vector) to help functions
    for(int idim=0; idim<VES3D_DIM; idim++){
      #pragma omp parallel for
      for(int i=0; i<N; i++){
        v_wrk[1].begin()[idim*N+i] += centrosome_velocity[idim]; // v_wrk[1] = -Vg*\xi + Vc
      }
    }
    GeometricDot(v_wrk[0], v_wrk[1], impingement_rate_); // impingement_rate_ = (\dot{n}/V_g * exp(-k_cat/V_g*D) * n)*(-Vg*\xi + Vc)
    // TODO: test if membrane is in the way between centrosome and boundary
    // only consider the surface points which centrosome can directly connent to
    // by now, impingement_rate without geometric chi factor has been calculated

    // calculate geometric factor
    set_one(s_wrk[1]); // till end of this function s_wrk[0] is alwarys one
    axpy(1.0/params_.fg_radius, s_wrk[0], s_wrk[0]); // s_wrk[0] = D/r_m, now D is stored no where
    xInv(s_wrk[0], s_wrk[0]); // s_wrk[0] = r_m/D
    chi(s_wrk[0], s_wrk[1], s_wrk[0]); // s_wrk[0] = 0.5*(1 - 1/sqrt(1+(r_m/D)^2))

    // calculate impingement_rate_ with geometric factor
    xy(s_wrk[0], impingement_rate_, impingement_rate_); // impingement_rate_ = 0.5*(1 - 1/sqrt(1+(r_m/D)^2)) * (\dot{n}/V_g * exp(-k_cat/V_g*D) * n)*(-Vg*\xi + Vc)
    xy(visible_zone_, impingement_rate_, impingement_rate_); // impingement_rate_ = 0.5*(1 - 1/sqrt(1+(r_m/D)^2)) * (\dot{n}/V_g * exp(-k_cat/V_g*D) * n)*(-Vg*\xi + Vc) * visible_zone_

    // cap impingement_rate
    #pragma omp parallel for
    for(int i=0; i<impingement_rate_.size(); i++){
      if(impingement_rate_.begin()[i] < 0) impingement_rate_.begin()[i] = 0;
    }

    // begin to update binding probability P, (1-P) is the limit if M/M0 is very small
    axpy(-1.0, binding_probability_, s_wrk[1], s_wrk[0]); // s_wrk[0] = 1-P
    // if M/M0 is not small enough, don't take limit and calculate (1 - e^{-M/M0 * (1-P)})/(M/M0)
    if(params_.boundary_M/M0_ > 1e-10) {
        axpy(-params_.boundary_M/M0_, s_wrk[0], s_wrk[0]); // s_wrk[0] = -M/M0 * (1-P)
        Exp(s_wrk[0], s_wrk[0]); // s_wrk[0] = e^{-M/M0 * (1-P)}
        axpy(-1.0, s_wrk[0], s_wrk[1], s_wrk[0]);// 1 - e^{-M/M0 * (1-P)}
        axpy(M0_/params_.boundary_M, s_wrk[0], s_wrk[0]); // (1 - e^{-M/M0 * (1-P)}) / (M/M0)
    }
    // by now s_wrk[0] stores (1 - e^{-M/M0 * (1-P)}) / (M/M0)
    xy(impingement_rate_, s_wrk[0], s_wrk[0]); // R*(1 - e^{-M/M0 * (1-P)})/(M/M0)
    axpy(-params_.fg_detachment_rate, binding_probability_, s_wrk[0], s_wrk[0]); // R*(1 - e^{-c*(1-P)})/c - k *P
    axpy(params_.ts, s_wrk[0], binding_probability_, binding_probability_);// dt * (R*(1 - e^{-c*(1-P)})/c - k *P) + P

    // cap binding_probability_ to be [0,1]
    #pragma omp parallel for
    for(int i=0; i<binding_probability_.size(); i++){
      if(binding_probability_.begin()[i] < 0)
        binding_probability_.begin()[i] = 0;
      if(binding_probability_.begin()[i] > 1)
        binding_probability_.begin()[i] = 1;
    }

    // calculate Fpull_
    // scale Fpull_ by fg_pulling_force*M/A
    axpy(params_.fg_pulling_force*params_.boundary_M/area_, Fpull_, Fpull_);
    // scale Fpull_ by binding_probability
    xv(binding_probability_, Fpull_, Fpull_);
    // scale Fpull_ by visible_zone_
    xv(visible_zone_, Fpull_, Fpull_);

    // calculate force on centrosome
    if(Fcpull){
        integrator_(Fpull_, S_->getAreaElement(), *Fcpull);
        axpy(static_cast<value_type>(-1.0), *Fcpull, *Fcpull);
    }

}

template<typename SurfContainer>
void PullingBoundary<SurfContainer>::
GetVisibleZone(const value_type* centrosome_position, const Vec_t &vesicle_position)
{
    int p0 = vesicle_position.getShOrder();
    int p1 = 2*p0;
    int ves_stride_dim = 2*p1*(p1+1);
    int bdry_stride_dim = S_->getPosition().getStride();

    sctl::Vector<value_type> X0(vesicle_position.size(), (value_type*)vesicle_position.begin(), false);
    sctl::Vector<value_type> X, Xpole, Xcoef;

    SphericalHarmonics<value_type>::Grid2SHC(X0,p0,p0,Xcoef);
    SphericalHarmonics<value_type>::SHC2Grid(Xcoef,p0,p1,X);
    SphericalHarmonics<value_type>::SHC2Pole(Xcoef,p0,Xpole);

    // calculate vesicle axis aligned bounding box
    value_type ves_bb_min[VES3D_DIM], ves_bb_max[VES3D_DIM];
    for(size_t k=0; k<VES3D_DIM; k++){
        ves_bb_min[k] = std::min(Xpole[0+2*k],Xpole[1+2*k]);
        ves_bb_max[k] = std::max(Xpole[0+2*k],Xpole[1+2*k]);
        #pragma omp parallel for
        for(size_t i=0; i<ves_stride_dim; i++){
            ves_bb_min[k] = std::min(ves_bb_min[k], X[i+k*ves_stride_dim]);
            ves_bb_max[k] = std::max(ves_bb_max[k], X[i+k*ves_stride_dim]);
        }
    }

    // set visible zone to all 1
    set_one(visible_zone_);

    value_type origin[VES3D_DIM] = {centrosome_position[0], centrosome_position[1], centrosome_position[2]};
    // calculate visible zone
    #pragma omp parallel for
    for(size_t i=0; i<bdry_stride_dim; i++){
        // do vesicle bounding box and ray intersection check
        value_type dir[VES3D_DIM] = { S_->getPosition().begin()[0*bdry_stride_dim+i] - origin[0],
                                      S_->getPosition().begin()[1*bdry_stride_dim+i] - origin[1],
                                      S_->getPosition().begin()[2*bdry_stride_dim+i] - origin[2] };
        value_type coord[VES3D_DIM];
        bool ray_box_intersect = RayBoxCheck(ves_bb_min, ves_bb_max, origin, dir, coord);
        if(ray_box_intersect)
        {
            bool ray_tri_intersect = false;
            value_type tri[3][VES3D_DIM];

            // north pole triangles
            for(size_t k=0; k<VES3D_DIM; k++){
                tri[0][k] = Xpole[0+2*k];
            }
            for(size_t j=0; j<2*p1; j++){
                size_t i0 = 0;
                size_t j0 = ((j+0)       );
                size_t j1 = ((j+1)%(2*p1));
                for(size_t k=0; k<VES3D_DIM; k++){
                   tri[1][k] = X[j0 + 2*p1*(i0+(p1+1)*k)];
                   tri[2][k] = X[j1 + 2*p1*(i0+(p1+1)*k)];
                }
                // do triangle and ray intersection check
                ray_tri_intersect = RayTriCheck(tri, origin, dir);
                if(ray_tri_intersect){
                    visible_zone_.begin()[i] = 0;
                    break;
                }
            }
            if(ray_tri_intersect) continue;

            // south pole triangles
            for(size_t k=0; k<VES3D_DIM; k++){
                tri[0][k] = Xpole[1+2*k];
            }
            for(size_t j=0; j<2*p1; j++){
                size_t i0 = p1;
                size_t j0 = ((j+0)       );
                size_t j1 = ((j+1)%(2*p1));
                for(size_t k=0; k<VES3D_DIM; k++){
                   tri[1][k] = X[j0 + 2*p1*(i0+(p1+1)*k)];
                   tri[2][k] = X[j1 + 2*p1*(i0+(p1+1)*k)];
                }
                // do triangle and ray intersection check
                ray_tri_intersect = RayTriCheck(tri, origin, dir);
                if(ray_tri_intersect){
                    visible_zone_.begin()[i] = 0;
                    break;
                }
            }
            if(ray_tri_intersect) continue;

            // middle quads
            for(size_t j=0; j<p1; j++){
                size_t j0 = (j+0);
                size_t j1 = (j+1);
                bool ray_quad_intersect = false;
                for(size_t k=0; k<2*p1; k++){
                    size_t k0 = ((k+0)       );
                    size_t k1 = ((k+1)%(2*p1));
                    // do quad and ray intersection check
                    for(size_t l=0; l<VES3D_DIM; l++){
                        tri[0][l] = X[k0 + 2*p1*(j0+(p1+1)*l)];
                        tri[1][l] = X[k0 + 2*p1*(j1+(p1+1)*l)];
                        tri[2][l] = X[k1 + 2*p1*(j0+(p1+1)*l)];
                    }
                    ray_quad_intersect = RayTriCheck(tri, origin, dir);
                    if(ray_quad_intersect){
                        visible_zone_.begin()[i] = 0;
                        break;
                    }
                    for(size_t l=0; l<VES3D_DIM; l++){
                        tri[0][l] = X[k1 + 2*p1*(j1+(p1+1)*l)];
                        tri[1][l] = X[k1 + 2*p1*(j0+(p1+1)*l)];
                        tri[2][l] = X[k0 + 2*p1*(j1+(p1+1)*l)];
                    }
                    ray_quad_intersect = RayTriCheck(tri, origin, dir);
                    if(ray_quad_intersect){
                        visible_zone_.begin()[i] = 0;
                        break;
                    }
                }
                if(ray_quad_intersect) break;
            }
        }
    }
}
