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
void InterfacialForce<SurfContainer>::pullingForce(const SurfContainer &S, const Vec_t &centrosome_position, const Sca_t &binding_probability, const Sca_t &density, Vec_t &Fp) const
// returns 3d force vector per microtubule (not force density), ie f_0 \hat{\xi}(y) p(y,t)
// at all surface points y.
{

    // TODO: Pre-allocate
    Sca_t stmp;
    stmp.replicate(Fp);
    Vec_t wrk[2];
    wrk[0].replicate(Fp);
    wrk[1].replicate(Fp);

    // pulling force direction
    axpy(-1.0, S.getPosition(), centrosome_position, Fp);
    // normalizing  to get \hat\xi
    GeometricDot(Fp, Fp, stmp);
    Sqrt(stmp, stmp);
    uyInv(Fp, stmp, Fp);

    // only consider the surface points which centrosome can directly connent to
    // TODO: use ray tracing for complex shapes and get number of collisions to the surface
    // now only consider force direction dot with outward normal
    GeometricDot(Fp, S.getNormal(), stmp);
    #pragma omp parallel for
    for(int i=0; i<stmp.size(); i++){
        if(stmp.begin()[i] >=0)
            stmp.begin()[i] = params_.pulling_rate;     // f_0
        else
            stmp.begin()[i] = 0.0;
    }
    xv(stmp, Fp, Fp);
    xv(binding_probability, Fp, Fp);
    xv(density, Fp, Fp);

    sht_.lowPassFilter(Fp, wrk[0], wrk[1], Fp);  //filter high frequency
    /*
    // TODO: calculate in upsample space
    S.resample(params_.upsample_freq, &S_up); // upsample
    Vec_t Fp_up;
    Fp_up.replicate(S_up->getPosition());
    { // downsample Fs
      Vec_t wrk[2]; // TODO: Pre-allocate
      wrk[0].resize(Fp_up.getNumSubs(), params_.upsample_freq);
      wrk[1].resize(Fp_up.getNumSubs(), params_.upsample_freq);
      Fp.replicate(S.getPosition());
      Resample(Fp_up, sht_up_, sht_, wrk[0], wrk[1], Fp);
    }
    */
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
