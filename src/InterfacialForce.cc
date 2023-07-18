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
void InterfacialForce<SurfContainer>::pullingForce(const SurfContainer &S, const value_type* centrosome_position, const Sca_t &binding_probability, const Sca_t &density, Vec_t &Fp, Sca_t &impingement_rate, Vec_t *Fc, value_type *min_dist) const
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
    Vec_t Fp_up;
    { // upsample binding_probability
      binding_probability_up.resize(binding_probability.getNumSubs(), params_.upsample_freq);
      Resample(binding_probability, sht_, sht_up_, s_wrk[0], s_wrk[1], binding_probability_up);

      // upsample density
      density_up.resize(density.getNumSubs(), params_.upsample_freq);
      Resample(density, sht_, sht_up_, s_wrk[0], s_wrk[1], density_up);

      impingement_rate_up.resize(impingement_rate.getNumSubs(), params_.upsample_freq);
      Fp_up.resize(Fp.getNumSubs(), params_.upsample_freq);
    }

    // pulling force direction
    int N = Fp_up.size()/VES3D_DIM;
    for(int idim=0; idim<VES3D_DIM; idim++){
      #pragma omp parallel for
      for(int i=0; i<N; i++){
        Fp_up.begin()[idim*N+i] = centrosome_position[idim] - S_up->getPosition().begin()[idim*N+i];
      }
    }
    // normalizing  to get \hat\xi
    GeometricDot(Fp_up, Fp_up, s_wrk[0]);
    // s_wrk[0] stores D, distance between centrosome and membrane surface points
    Sqrt(s_wrk[0], s_wrk[0]);
    uyInv(Fp_up, s_wrk[0], Fp_up);
    *min_dist = MinAbs(s_wrk[0]);

    set_one(s_wrk[1]);
    axpy(1.0/params_.fg_radius, s_wrk[0], s_wrk[2]);
    xy(s_wrk[2], s_wrk[2], s_wrk[2]);
    axpy(1.0, s_wrk[1], s_wrk[2], s_wrk[2]);
    Sqrt(s_wrk[2], s_wrk[2]);
    xyInv(s_wrk[1], s_wrk[2], s_wrk[2]);
    axpy(-1.0, s_wrk[2], s_wrk[1], s_wrk[2]);
    axpy(0.5*params_.mt_nucleation_rate/params_.mt_growth_velocity, s_wrk[2], s_wrk[2]);
    axpy(-params_.mt_catastrophe_rate/params_.mt_growth_velocity, s_wrk[0], s_wrk[1]);
    Exp(s_wrk[1], s_wrk[1]);
    xy(s_wrk[1], s_wrk[2], s_wrk[1]);
    xv(s_wrk[1], S_up->getNormal(), v_wrk[0]);
    axpy(params_.mt_growth_velocity, Fp_up, v_wrk[1]);
    GeometricDot(v_wrk[0], v_wrk[1], impingement_rate_up);

    // only consider the surface points which centrosome can directly connent to
    // TODO: use ray tracing for complex shapes and get number of collisions to the surface
    // now only consider force direction dot with outward normal
    GeometricDot(Fp_up, S_up->getNormal(), s_wrk[0]);
    #pragma omp parallel for
    for(int i=0; i<s_wrk[0].size(); i++){
        value_type smooth_factor = (1.0+tanh(s_wrk[0].begin()[i]*40.0))/2;
        S_up->contact_indicator_.begin()[i] = smooth_factor;
        s_wrk[0].begin()[i] = smooth_factor * params_.fg_pulling_force;
        impingement_rate_up.begin()[i] *= smooth_factor;
        /*
        if(s_wrk[0].begin()[i] >=0){
            s_wrk[0].begin()[i] = params_.fg_pulling_force;     // f_0
        }
        else{
            s_wrk[0].begin()[i] = 0.0;
            impingement_rate_up.begin()[i] = 0;
        }
        */
    }
    COUT("max indicator upsampled: "<<MaxAbs(S_up->contact_indicator_));
    xv(s_wrk[0], Fp_up, Fp_up);
    xv(binding_probability_up, Fp_up, Fp_up);
    xv(density_up, Fp_up, Fp_up);

    //sht_.lowPassFilter(Fp_up, v_wrk[0], v_wrk[1], Fp_up);  //filter high frequency
    //sht_.lowPassFilter(impingement_rate_up, s_wrk[0], s_wrk[1], impingement_rate_up);  //filter high frequency

    { // downsample Fp
      Fp.replicate(S.getPosition());
      Resample(Fp_up, sht_up_, sht_, v_wrk[0], v_wrk[1], Fp);
      // downsample impingement_rate
      impingement_rate.replicate(S.getPosition());
      Resample(impingement_rate_up, sht_up_, sht_, s_wrk[0], s_wrk[1], impingement_rate);
      Resample(S_up->contact_indicator_, sht_up_, sht_, s_wrk[0], s_wrk[1], S.contact_indicator_);
      COUT("max indicator downsampled: "<<MaxAbs(S.contact_indicator_));
    }

    /*
    // cap impingement_rate
    #pragma omp parallel for
    for(int i=0; i<impingement_rate.size(); i++){
      if(impingement_rate.begin()[i] < 0) impingement_rate.begin()[i] = 0;
    }
    */

    // calculate force on centrosome
    if(Fc){
        integrator_(Fp, S.getAreaElement(), *Fc);
        axpy(static_cast<value_type>(-1.0), *Fc, *Fc);
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
