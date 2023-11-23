template<typename SurfContainer>
InterfacialForce<SurfContainer>::InterfacialForce(
    const Parameters<value_type> &params,
    const VProp_t &ves_props,
    const OperatorsMats<Arr_t> &mats) :
    params_(params),
    ves_props_(ves_props),
    sht_   (mats.p_   , mats.mats_p_   ),
    sht_up_(mats.p_up_, mats.mats_p_up_),
    cen(1, 1, std::make_pair(1,1)),
    S_up(NULL)
{}

template<typename SurfContainer>
InterfacialForce<SurfContainer>::~InterfacialForce()
{
  if(S_up) delete S_up;
}

template<typename SurfContainer>
void InterfacialForce<SurfContainer>::bendingForce(const SurfContainer &S,
    Vec_t &Fb) const
{
    S.resample(params_.upsample_freq, &S_up); // upsample

    Vec_t Fb_up;
    Fb_up.replicate(S_up->getPosition());
    s1.replicate(S_up->getPosition());
    s2.replicate(S_up->getPosition());

    xy(S_up->getMeanCurv(), S_up->getMeanCurv(), s2);
    axpy(static_cast<typename SurfContainer::value_type>(-1.0),
         S_up->getGaussianCurv(), s2, s2);
    xy(S_up->getMeanCurv(), s2, s2);

    S_up->grad(S_up->getMeanCurv(), Fb_up);
    S_up->div(Fb_up, s1);

    axpy(static_cast<typename SurfContainer::value_type>(2), s2, s1, s1);
    xv(s1, S_up->getNormal(), Fb_up);
    av(ves_props_.bending_coeff,Fb_up, Fb_up);

    { // downsample Fb
      Vec_t wrk[2]; // TODO: Pre-allocate
      wrk[0].resize(Fb_up.getNumSubs(), params_.upsample_freq);
      wrk[1].resize(Fb_up.getNumSubs(), params_.upsample_freq);
      Fb.replicate(S.getPosition());
      Resample(Fb_up, sht_up_, sht_, wrk[0], wrk[1], Fb);
    }
}

template<typename SurfContainer>
void InterfacialForce<SurfContainer>::linearBendingForce(const SurfContainer &S,
    const Vec_t &x_new, Vec_t &Fb) const
{
    S.resample(params_.upsample_freq, &S_up); // upsample
    Vec_t x_new_up;
    { // upsample x_new
      Vec_t wrk[2]; // TODO: Pre-allocate
      wrk[0]  .resize(x_new.getNumSubs(), params_.upsample_freq);
      wrk[1]  .resize(x_new.getNumSubs(), params_.upsample_freq);
      x_new_up.resize(x_new.getNumSubs(), params_.upsample_freq);
      Resample(x_new, sht_, sht_up_, wrk[0], wrk[1], x_new_up);
    }

    Vec_t Fb_up;
    Fb_up.replicate(S_up->getPosition());
    s1.replicate(S_up->getPosition());
    s2.replicate(S_up->getPosition());

    S_up->linearizedMeanCurv(x_new_up, s1);

    xy(S_up->getMeanCurv(), S_up->getMeanCurv(), s2);
    axpy(static_cast<typename SurfContainer::value_type>(-1.0),
        S_up->getGaussianCurv(), s2, s2);
    xy(s1, s2, s2);

    S_up->grad(s1, Fb_up);
    S_up->div(Fb_up, s1);
    axpy(static_cast<typename SurfContainer::value_type>(2), s2, s1, s1);

    xv(s1, S_up->getNormal(), Fb_up);
    av(ves_props_.bending_coeff, Fb_up, Fb_up);

    { // downsample Fb
      Vec_t wrk[2]; // TODO: Pre-allocate
      wrk[0].resize(Fb_up.getNumSubs(), params_.upsample_freq);
      wrk[1].resize(Fb_up.getNumSubs(), params_.upsample_freq);
      Fb.replicate(S.getPosition());
      Resample(Fb_up, sht_up_, sht_, wrk[0], wrk[1], Fb);
    }
}

template<typename SurfContainer>
void InterfacialForce<SurfContainer>::tensileForce(const SurfContainer &S,
    const Sca_t &tension, Vec_t &Fs) const
{
    S.resample(params_.upsample_freq, &S_up); // upsample
    Sca_t tension_up;
    { // upsample tension
      Sca_t wrk[2]; // TODO: Pre-allocate
      wrk[0]    .resize(tension.getNumSubs(), params_.upsample_freq);
      wrk[1]    .resize(tension.getNumSubs(), params_.upsample_freq);
      tension_up.resize(tension.getNumSubs(), params_.upsample_freq);
      Resample(tension, sht_, sht_up_, wrk[0], wrk[1], tension_up);
    }

    Vec_t Fs_up;
    Fs_up.replicate(S_up->getPosition());
    v1.replicate(S_up->getPosition());

    xv(S_up->getMeanCurv(), S_up->getNormal(), Fs_up);
    xv(tension_up, Fs_up, Fs_up);
    S_up->grad(tension_up, v1);
    axpy(static_cast<typename SurfContainer::value_type>(2), Fs_up, v1, Fs_up);

    { // downsample Fs
      Vec_t wrk[2]; // TODO: Pre-allocate
      wrk[0].resize(Fs_up.getNumSubs(), params_.upsample_freq);
      wrk[1].resize(Fs_up.getNumSubs(), params_.upsample_freq);
      Fs.replicate(S.getPosition());
      Resample(Fs_up, sht_up_, sht_, wrk[0], wrk[1], Fs);
    }
}

template<typename SurfContainer>
void InterfacialForce<SurfContainer>::gravityForce(const SurfContainer &S, const Vec_t &x, Vec_t &Fg) const
{
    value_type zero(0);
    value_type one(1.0);

    if(!ves_props_.has_excess_density){
        Vec_t::getDevice().Memset(Fg.begin(), zero, Fg.mem_size());
        return;
    }

    /*
     * The traction jump due to gravity is (excess_density)*(g.x) n
     */

    // g.x
    Sca_t s;
    Vec_t w;
    s.replicate(x);
    w.replicate(x);

    ShufflePoints(x,w);
    int m(1), k(VES3D_DIM), n(x.size()/VES3D_DIM);

    Vec_t::getDevice().gemm("N","N", &m, &n, &k,
        &one, params_.gravity_field, &m,
        w.begin(), &k,
        &zero, s.begin(), &m);

    // (g.x) n
    xv(s,S.getNormal(),Fg);

    // scale by excess density
    av(ves_props_.excess_density,Fg,Fg);
}

template<typename SurfContainer>
void InterfacialForce<SurfContainer>::pullingForce(const SurfContainer &S, const value_type* centrosome_position, const value_type* centrosome_velocity, Sca_t &binding_probability, Sca_t &density, Vec_t &Fpull, Vec_t &Fpush, Sca_t &impingement_rate, Vec_t *Fcpull, Vec_t *Fcpush, value_type *min_dist) const
// returns 3d force vector per microtubule (not force density), ie f_0 \hat{\xi}(y) p(y,t)
// at all surface points y.
{
    // upsample everything
    S.resample(params_.upsample_freq, &S_up); // upsample

    // TODO: Pre-allocate
    Sca_t s_wrk[3];
    s_wrk[0].replicate(S_up->getPosition());
    s_wrk[1].replicate(S_up->getPosition());
    s_wrk[2].replicate(S_up->getPosition());
    Vec_t v_wrk[2];
    v_wrk[0].replicate(S_up->getPosition());
    v_wrk[1].replicate(S_up->getPosition());

    Sca_t binding_probability_up;
    Sca_t density_up;
    Sca_t impingement_rate_up;
    Vec_t Fpull_up;
    Vec_t Fpush_up;
    { // upsample binding_probability
      binding_probability_up.resize(binding_probability.getNumSubs(), params_.upsample_freq);
      Resample(binding_probability, sht_, sht_up_, s_wrk[0], s_wrk[1], binding_probability_up);

      // upsample density
      density_up.resize(density.getNumSubs(), params_.upsample_freq);
      Resample(density, sht_, sht_up_, s_wrk[0], s_wrk[1], density_up);

      impingement_rate_up.resize(impingement_rate.getNumSubs(), params_.upsample_freq);
      Fpull_up.resize(Fpull.getNumSubs(), params_.upsample_freq);
      Fpush_up.resize(Fpush.getNumSubs(), params_.upsample_freq);
    }

    // pulling force direction
    // FIXME
    // need to put this loop(each 3D-vector of vector field minus const 3D-vector) to help functions
    int N = Fpull_up.size()/VES3D_DIM;
    for(int idim=0; idim<VES3D_DIM; idim++){
      #pragma omp parallel for
      for(int i=0; i<N; i++){
        Fpull_up.begin()[idim*N+i] = centrosome_position[idim] - S_up->getPosition().begin()[idim*N+i];
      }
    }
    // normalizing  to get \hat\xi
    GeometricDot(Fpull_up, Fpull_up, s_wrk[0]);
    // s_wrk[0] stores D, distance between centrosome and membrane surface points
    Sqrt(s_wrk[0], s_wrk[0]);
    uyInv(Fpull_up, s_wrk[0], Fpull_up);
    *min_dist = MinAbs(s_wrk[0]);
    // at this point Fpull_up stores the unit direction of pullingForce
    // s_wrk[0] stores the distance between centrosome and membrane points

    // calculate the impingement rate
    set_one(s_wrk[1]); // 1
    axpy(1.0/params_.fg_radius, s_wrk[0], s_wrk[2]); // s_wrk2 = D/r_m
    xyInv(s_wrk[1], s_wrk[2], s_wrk[2]); // s_wrk2 = r_m/D
    xy(s_wrk[2], s_wrk[2], s_wrk[2]); // s_wrk2 = (r_m/D)^2
    axpy(1.0, s_wrk[1], s_wrk[2], s_wrk[2]); // s_wrk2 = 1 + (r_m/D)^2
    Sqrt(s_wrk[2], s_wrk[2]); // s_wrk2 = sqrt(1+(r_m/D)^2)
    xyInv(s_wrk[1], s_wrk[2], s_wrk[2]); // s_wrk2 = 1/sqrt(1+(r_m/D)^2)
    axpy(-1.0, s_wrk[2], s_wrk[1], s_wrk[2]); //s_wrk2 = 1 - 1/sqrt(1+(r_m/D)^2)
    axpy(0.5, s_wrk[2], s_wrk[2]); // s_wrk[2] = 0.5*(1 - 1/sqrt(1+(r_m/D)^2))
    axpy(-params_.mt_catastrophe_rate/params_.mt_growth_velocity, s_wrk[0], s_wrk[1]); // s_wrk[1] = -k_cat/V_g * D
    Exp(s_wrk[1], s_wrk[1]); // s_wrk[1] = exp(-k_cat/V_g*D)
    axpy(params_.mt_nucleation_rate/params_.mt_growth_velocity, s_wrk[1], s_wrk[1]); // s_wrk[1] = \dot{n}/V_g * exp(-k_cat/V_g*D)
    xv(s_wrk[1], S_up->getNormal(), v_wrk[0]); // v_wrk[0] = \dot{n}/V_g * exp(-k_cat/V_g*D) * n
    axpy(params_.mt_growth_velocity, Fpull_up, v_wrk[1]); // v_wrk[1] = V_g * \xi
    // FIXME: subtract the last time step centrosome velocity from v_wrk[1]
    // need to put this loop(each 3D-vector of vector field minus const 3D-vector) to help functions
    for(int idim=0; idim<VES3D_DIM; idim++){
      #pragma omp parallel for
      for(int i=0; i<N; i++){
        v_wrk[1].begin()[idim*N+i] -= centrosome_velocity[idim]; // v_wrk[1] = Vg*\xi - Vc
      }
    }
    GeometricDot(v_wrk[0], v_wrk[1], impingement_rate_up); // impingement_rate_up = (\dot{n}/V_g * exp(-k_cat/V_g*D) * n)*(Vg*\xi - Vc)
    // only consider the surface points which centrosome can directly connent to
    // TODO: use ray tracing for complex shapes and get number of collisions to the surface
    // now only consider force direction dot with outward normal
    GeometricDot(Fpull_up, S_up->getNormal(), s_wrk[0]);
    #pragma omp parallel for
    for(int i=0; i<s_wrk[0].size(); i++){
        value_type smooth_factor = (1.0+tanh(s_wrk[0].begin()[i]*40.0))/2;
        S_up->contact_indicator_.begin()[i] = smooth_factor;
        s_wrk[0].begin()[i] = smooth_factor * params_.fg_pulling_force;
        impingement_rate_up.begin()[i] *= smooth_factor;
        if(impingement_rate_up.begin()[i] < 0) impingement_rate_up.begin()[i] = 0;
    }
    // now, impingement_rate has been updated in upsampled space
    // calaculate pushing force
    xv(impingement_rate_up, Fpull_up, Fpush_up);
    axpy(-params_.mt_pushing_force, Fpush_up, Fpush_up);

    // get impingement_rate_up
    xy(s_wrk[2], impingement_rate_up, impingement_rate_up); // impingement_rate_up = 0.5*(1 - 1/sqrt(1+(r_m/D)^2)) * (\dot{n}/V_g * exp(-k_cat/V_g*D) * n)*(Vg*\xi - Vc)

    // scale Fpull_up by smoothed indicator function times fg_pulling_force
    xv(s_wrk[0], Fpull_up, Fpull_up);

    // update binding probability P
    set_one(s_wrk[1]);
    axpy(-1.0, binding_probability_up, s_wrk[1], s_wrk[1]); //1-P
    axpy(-1.0, density_up, s_wrk[2]); // -c
    xy(s_wrk[1], s_wrk[2], s_wrk[1]);// -c*(1-P)
    Exp(s_wrk[1], s_wrk[1]); // e^{-c*(1-P)}
    set_one(s_wrk[2]);
    axpy(-1.0, s_wrk[1], s_wrk[2], s_wrk[1]);// 1 - e^{-c*(1-P)}
    xyInv(s_wrk[1], density_up, s_wrk[1]); // (1 - e^{-c*(1-P)})/c
    // in case density is near zero, take the limit
    #pragma omp parallel for
    for(int i=0; i<density_up.size(); i++){
      if(abs(density_up.begin()[i]) < 1e-10)
        s_wrk[1].begin()[i] = 1.0 - binding_probability_up.begin()[i];
    }
    xy(impingement_rate_up, s_wrk[1], s_wrk[1]); // R*(1 - e^{-c*(1-P)})/c
    axpy(-params_.fg_detachment_rate, binding_probability_up, s_wrk[1], s_wrk[1]); // R*(1 - e^{-c*(1-P)})/c - k *P
    axpy(params_.ts, s_wrk[1], binding_probability_up, binding_probability_up);// dt * (R*(1 - e^{-c*(1-P)})/c - k *P) + P

    // cap binding_probability_ to be [0,1]
    #pragma omp parallel for
    for(int i=0; i<binding_probability_up.size(); i++){
      if(binding_probability_up.begin()[i] < 0)
        binding_probability_up.begin()[i] = 0;
      if(binding_probability_up.begin()[i] > 1)
        binding_probability_up.begin()[i] = 1;
    }

    // calculate fp
    // scale Fpull_up by binding_probability
    xv(binding_probability_up, Fpull_up, Fpull_up);
    // scale Fpull_up by density
    xv(density_up, Fpull_up, Fpull_up);

    // calculate force on centrosome
    if(Fcpull){
        integrator_(Fpull_up, S_up->getAreaElement(), *Fcpull);
        axpy(static_cast<value_type>(-1.0), *Fcpull, *Fcpull);
    }
    // calculate force on centrosome
    if(Fcpush){
        integrator_(Fpush_up, S_up->getAreaElement(), *Fcpush);
        axpy(static_cast<value_type>(-1.0), *Fcpush, *Fcpush);
    } 

    // begin to update density
    // note that f_push does not advect density as in the model
    // if no advection apart from membrane vel, that's it, since for a vesicle w/ local area-conservation, div_s u = 0!  (not divs u_s = 0 !)
    // Lagrangian:    (d/dt)c + c (div_s u) = 0
    // f_p = pulling force vector per unit force generator,  eta_m = drag
    // v_p = pulling-related advection vel relative to u.
    // D = diffusion const
    // Lagrangian w/ extra transport (pulling) v_p = (I-nn).f_p / eta_m
    //      (d/dt)c + div_s (v_p c)   + c (div_s u)          =    D \Delta_s c
    //                pulling advec    stretching of dS           diffusion
    // Test (d/dt) \int_surf c(x,t) dS = 0  "mass conservation"
    // Simulation works in lagrangian!
    /* Discussion from 3/31/23:
      in R2: transport eqn  Eulerian  (par/par t)c + div(uc) = 0    u = advec vel
                                       ^ means sit at fixed x, rate of change of c
                            Lagrangian  (d/dt)c = 0
                            d/dt = (par/par t) + u.grad  + (div u)
                            d/dt = (par/par t) + div (u .)
                                                  if div u nonzero
      on moving Gamma surf that may leave x, what does (par/par t)c(x,t) mean ???
     */
    // advect density on surface: fill wrk with -dc/dt...
    // sht_.lowPassFilter(density_up, s_wrk[0], s_wrk[1], density_up);
    // to do: add stretching term in case it's not const in some future setting?
    // need: compute v_p         ... add eta_m, D, f_p (???) to Parameter struct
    axpy(1.0/params_.fg_drag_coeff, Fpull_up, v_wrk[0]);  // f_p/eta
    S_up->mapToTangentSpace(v_wrk[0], false);   // overwrites u1, now v_p, tangential
    S_up->div(v_wrk[0], s_wrk[0]);                // wrk = div_s.(c v_p)
    // add D \Delta_s c = div_s (D grad_s c), and subtract from -dc/dt
    S_up->grad(density_up, v_wrk[0]);
    set_zero(v_wrk[1]);
    axpy(-params_.diffusion_rate, v_wrk[0], v_wrk[1], v_wrk[0]);      // u1 <-  -D grad_s c
    S_up->div(v_wrk[0], s_wrk[1]);              // wrk2 <-  div_s (-D grad_s c)
    axpy(1.0, s_wrk[0], s_wrk[1], s_wrk[0]);     // wrk = add advection plus diffusion
    axpy(-params_.ts, s_wrk[0], density_up, density_up);          // den -= dt*wrk
    /*
    // cap density to be non-negative
    #pragma omp parallel for
    for(int ii=0; ii<density_.size(); ii++){
      if(density_.begin()[ii] < 0)
        density_.begin()[ii] = 0;
    }
    */

    //sht_.lowPassFilter(Fpull_up, v_wrk[0], v_wrk[1], Fpull_up);  //filter high frequency
    //sht_.lowPassFilter(Fpush_up, v_wrk[0], v_wrk[1], Fpush_up);  //filter high frequency
    //sht_.lowPassFilter(impingement_rate_up, s_wrk[0], s_wrk[1], impingement_rate_up);  //filter high frequency
    //sht_.lowPassFilter(binding_probability_up, s_wrk[0], s_wrk[1], binding_probability_up);  //filter high frequency
    //sht_.lowPassFilter(density_up, s_wrk[0], s_wrk[1], density_up);  //filter high frequency

    { // downsample Fpull
      Fpull.replicate(S.getPosition());
      Resample(Fpull_up, sht_up_, sht_, v_wrk[0], v_wrk[1], Fpull);
      // downsample Fpush
      Fpush.replicate(S.getPosition());
      Resample(Fpush_up, sht_up_, sht_, v_wrk[0], v_wrk[1], Fpush);
      // downsample impingement_rate
      impingement_rate.replicate(S.getPosition());
      Resample(impingement_rate_up, sht_up_, sht_, s_wrk[0], s_wrk[1], impingement_rate);
      Resample(binding_probability_up, sht_up_, sht_, s_wrk[0], s_wrk[1], binding_probability);
      Resample(density_up, sht_up_, sht_, s_wrk[0], s_wrk[1], density);
      Resample(S_up->contact_indicator_, sht_up_, sht_, s_wrk[0], s_wrk[1], S.contact_indicator_);
    }

    // cap impingement_rate
    #pragma omp parallel for
    for(int i=0; i<impingement_rate.size(); i++){
      if(impingement_rate.begin()[i] < 0) impingement_rate.begin()[i] = 0;
    }

    // cap binding_probability_ to be [0,1]
    #pragma omp parallel for
    for(int i=0; i<binding_probability.size(); i++){
      if(binding_probability.begin()[i] < 0)
        binding_probability.begin()[i] = 0;
      if(binding_probability.begin()[i] > 1)
        binding_probability.begin()[i] = 1;
    }

}

template<typename SurfContainer>
void InterfacialForce<SurfContainer>::explicitTractionJump(const SurfContainer &S, Vec_t &F) const
{
    bendingForce(S, F);

    ftmp.replicate(S.getPosition());
    gravityForce(S, S.getPosition(), ftmp);
    axpy((value_type) 1.0, F, ftmp, F);
}

template<typename SurfContainer>
void InterfacialForce<SurfContainer>::implicitTractionJump(const SurfContainer &S, const Vec_t &x,
    const Sca_t &tension, Vec_t &F) const
{
    linearBendingForce(S, x, F);

    ftmp.replicate(x);
    gravityForce(S, x, ftmp);
    axpy((value_type) 1.0, F, ftmp, F);

    tensileForce(S, tension, ftmp);
    axpy( (value_type) 1.0, F, ftmp, F);
}
