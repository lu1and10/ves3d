template<typename SurfContainer, typename Interaction>
InterfacialVelocity<SurfContainer, Interaction>::
InterfacialVelocity(SurfContainer &S_in, const Interaction &Inter,
    const OperatorsMats<Arr_t> &mats,
    const Parameters<value_type> &params, const VProp_t &ves_props,
    const BgFlowBase<Vec_t> &bgFlow, PSolver_t *parallel_solver) :
    S_(S_in),
    interaction_(Inter),
    bg_flow_(bgFlow),
    params_(params),
    ves_props_(ves_props),
    Intfcl_force_(params,ves_props_,mats),
    //
    parallel_solver_(parallel_solver),
    psolver_configured_(false),
    precond_configured_(false),
    parallel_matvec_(NULL),
    parallel_rhs_(NULL),
    parallel_u_(NULL),
    //
    dt_(params_.ts),
    sht_(mats.p_, mats.mats_p_),
    sht_upsample_(mats.p_up_, mats.mats_p_up_),
    checked_out_work_sca_(0),
    checked_out_work_vec_(0),
    stokes_(params_.sh_order,params_.upsample_freq,params_.periodic_length,params_.repul_dist),
    S_up_(NULL),
    pos_vel_(S_in.velocity_),
    tension_(S_in.tension_)
{
    //fixed_bd = new FixedBoundary();
    el_force = new ElectroForce<SurfContainer>(S_, params_);
    
    int p = S_.getPosition().getShOrder();
    int np = S_.getPosition().getStride();

    //W_spherical
    w_sph_.resize(1, p);
    w_sph_inv_.resize(1, p);
    w_sph_.getDevice().Memcpy(w_sph_.begin(), mats.w_sph_,
        np * sizeof(value_type), device_type::MemcpyDeviceToDevice);
    xInv(w_sph_,w_sph_inv_);

    //Singular quadrature weights
    sing_quad_weights_.resize(1,p);
    sing_quad_weights_.getDevice().Memcpy(sing_quad_weights_.begin(),
        mats.sing_quad_weights_, sing_quad_weights_.size() *
        sizeof(value_type),
        device_type::MemcpyDeviceToDevice);

    //quadrature weights
    quad_weights_.resize(1,p);
    quad_weights_.getDevice().Memcpy(quad_weights_.begin(),
        mats.quad_weights_,
        quad_weights_.size() * sizeof(value_type),
        device_type::MemcpyDeviceToDevice);

    int p_up = sht_upsample_.getShOrder();
    quad_weights_up_.resize(1, p_up);
    quad_weights_up_.getDevice().Memcpy(quad_weights_up_.begin(),
        mats.quad_weights_p_up_,
        quad_weights_up_.size() * sizeof(value_type),
        device_type::MemcpyDeviceToDevice);

    //spectrum in harmonic space, diagonal
    if (params_.time_precond == DiagonalSpectral){
        position_precond.resize(1,p);
        tension_precond.resize(1,p);
    }
    
    // *init Contact Interface*
    ASSERT(params_.upsample_freq >= params_.sh_order, "Bad collision upsample freq size");
    int np_up = 2*params_.sh_order*(params_.sh_order+1);
    if(params_.col_upsample)
        np_up = 2*params_.upsample_freq*(params_.upsample_freq+1);
    nv_ = S_.getPosition().getNumSubs();
    
    std::vector<value_type> x_s(np_up*COORD_DIM*2, 0.0);
    pvfmm::Vector<value_type> x_s_pole;
    Vec_t x_pair(2, params_.sh_order);
    // assumption: S_.getPosition() has at least one vesicle
    // init x_pair
    x_pair.getDevice().Memcpy(&(x_pair.begin()[0]), S_.getPosition().begin(), 
            x_pair.size()*sizeof(value_type)/2, device_type::MemcpyDeviceToDevice);
    x_pair.getDevice().Memcpy(&(x_pair.begin()[x_pair.size()/2]), S_.getPosition().begin(), 
            x_pair.size()*sizeof(value_type)/2, device_type::MemcpyDeviceToDevice);
    
    //init sphericalharmonics for per vesicle transform
    SphericalHarmonics<value_type>::SetB0B1();

    // get upsampled position of x_pair
    // init contact object
    GetColPos(x_pair, x_s, x_s_pole);
    if(params_.col_upsample)
        CI_pair_.generateMesh(x_s, &x_s_pole[0], params_.upsample_freq, 2);
    else
        CI_pair_.generateMesh(x_s, &x_s_pole[0], params_.sh_order, 2);
    CI_pair_.init(SURF_SIMPLE);

    int omp_p=omp_get_max_threads();
    //COUT("we have number of omp threads: "<<omp_p);
    for(int i=0; i<omp_p; ++i)
        CI_pairs_.push_back(new ContactInterface());
    #pragma omp parallel for
    for(int i=0; i<omp_p; ++i)
    {
        //#pragma omp critical
        //COUT("number of omp threads: "<<omp_get_num_threads()<<". thread #: "<<omp_get_thread_num());
        if(params_.col_upsample)
            CI_pairs_[i]->generateMesh(x_s, &x_s_pole[0], params_.upsample_freq, 2);
        else
            CI_pairs_[i]->generateMesh(x_s, &x_s_pole[0], params_.sh_order, 2);
        CI_pairs_[i]->init(SURF_SIMPLE);
    }

    std::vector<value_type> pos_s(np_up*COORD_DIM*nv_, 0.0);
    pvfmm::Vector<value_type> pos_s_pole;
    GetColPosAll(S_.getPosition(), pos_s, pos_s_pole);
    /*
    if(params_.col_upsample)
        CI_bd_.generateMesh(pos_s, &pos_s_pole[0], params_.upsample_freq, S_.getPosition().getNumSubs(), 
                fixed_bd->tri_vertices, fixed_bd->num_vertices_per_patch_1d, fixed_bd->num_patches);
    else
        CI_bd_.generateMesh(pos_s, &pos_s_pole[0], params_.sh_order, S_.getPosition().getNumSubs(), 
                fixed_bd->tri_vertices, fixed_bd->num_vertices_per_patch_1d, fixed_bd->num_patches);
    CI_bd_.init(SURF_SIMPLE);
    */
    /*
    static std::vector<value_type> pos_s(np_up*COORD_DIM*nv_, 0.0);
    static pvfmm::Vector<value_type> pos_s_pole;
    GetColPos(S_.getPosition(), pos_s, pos_s_pole);
    CI_.generateMesh(pos_s, &pos_s_pole[0], params_.upsample_freq, S_.getPosition().getNumSubs());
    CI_.writeOFF();
    CI_.init(SURF_SUBDIVISION);
    */
    
    // init contact variables
    // init vgrad_ and vgrad_ind_
    if(params_.col_upsample)
        vgrad_.resize(S_.getPosition().getNumSubs(), params_.upsample_freq);
    else
        vgrad_.resize(S_.getPosition().getNumSubs(), params_.sh_order);
    vgrad_.getDevice().Memset(vgrad_.begin(), 0, sizeof(value_type)*vgrad_.size());
    vgrad_ind_.resize(vgrad_.size(), 0);
    // init ghost_vgrad_ and ghost_vgrad_ind_
    ghost_vgrad_.clear();
    ghost_vgrad_ind_.clear();
    // init single vesicle surface
    Vec_t xi_tmp(1, params_.sh_order);
    xi_tmp.getDevice().Memset(xi_tmp.begin(), 0, sizeof(value_type)*xi_tmp.size());
    S_i_ = new SurfContainer(params_.sh_order, mats, &xi_tmp, params_.filter_freq,
            params_.rep_filter_freq, params_.rep_type, params_.rep_exponent);
    S_i_->set_name("subsurface_i");
    for(int i=0; i<omp_p; ++i)
    {
        S_is_.push_back(new SurfContainer(params_.sh_order, mats, &xi_tmp, params_.filter_freq,
                params_.rep_filter_freq, params_.rep_type, params_.rep_exponent));
        //S_is_[i]->set_name("subsurface_i");
    }
    for(int i=0; i<omp_p; ++i)
    {
        S_is_[i]->resample(params_.upsample_freq, &(Intfcl_force_.S_ups[i]));
    }
    // parallel bounding box intersection checker
    VBBI_ = new VesBoundingBox<value_type>(params_.periodic_length);
    BDVBBI_ = new BDVesBoundingBox<value_type>(params_.periodic_length);
    // init the alltoallv send counter for lambda communication
    int npros;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &npros);
    s_ind_cnt_.ReInit(npros); s_ind_dsp_.ReInit(npros); r_ind_cnt_.ReInit(npros); r_ind_dsp_.ReInit(npros);
    // init other variables
    PA_.clear();
    num_cvs_ = 0;
    sum_num_cvs_ = 0;
    IV_.clear();
    parallel_lcp_matrix_.clear();
    contact_vesicle_list_.clear();
    lcp_parallel_linear_solver_ = NULL;
    // *end of init Contact Interface*

    // MKL GMRES solver
    CHK(linear_solver_gmres_.SetContext(static_cast<const void*>(this)));
    for(int i=0; i<omp_p; ++i)
    {
        linear_solver_gmress_.push_back(GMRESLinSolver<value_type>());
    }
    for(int i=0; i<omp_p; ++i)
    {
        CHK(linear_solver_gmress_[i].SetContext(static_cast<const void*>(this)));
    }

    // init x and rhs storage
    size_t vsz(stokesBlockSize()), tsz(tensionBlockSize());
    ASSERT(S_.getPosition().size()==vsz,"Bad sizes");
    size_t N_size = vsz+tsz;
    x_host_ = new value_type[N_size];
    rhs_host_ = new value_type[N_size];
    CI_ves_bd_ = nullptr;

    /*
    // expand vesicle
    ves_stop.resize(nv_, false);
    expand_by.resize(nv_, 0);
    S_.getCenters(pos_vel_);
    Arr_t cm_tmp(nv_*COORD_DIM);
    for(int i=0; i<nv_; i++)
        for(int j=0; j<COORD_DIM;j++)
            cm_tmp.begin()[i*COORD_DIM+j] = -pos_vel_.begin()[i*COORD_DIM+j];
    Arr_t::getDevice().apx(cm_tmp.begin(),
            S_.getPosition().begin(),
            S_.getPosition().getStride(),
            S_.getPosition().getNumSubFuncs(),
            pos_vel_.begin());
    */
}

template<typename SurfContainer, typename Interaction>
InterfacialVelocity<SurfContainer, Interaction>::
~InterfacialVelocity()
{
    COUTDEBUG("Destroying an instance of interfacial velocity");
    assert(!checked_out_work_sca_);
    assert(!checked_out_work_vec_);

    purgeTheWorkSpace();

    COUTDEBUG("Deleting parallel matvec and containers");
    delete parallel_matvec_;
    delete parallel_rhs_;
    delete parallel_u_;
    delete VBBI_;
    delete BDVBBI_;
    delete lcp_parallel_linear_solver_;

    for(typename std::map<int, Vec_t*>::iterator i=ghost_vgrad_.begin(); i!=ghost_vgrad_.end(); i++)
        delete i->second;
    ghost_vgrad_.clear();
    for(std::map<int, std::vector<int>*>::iterator i=ghost_vgrad_ind_.begin(); i!=ghost_vgrad_ind_.end(); i++)
        delete i->second;
    ghost_vgrad_ind_.clear();

    if(S_up_) delete S_up_;
    if(S_i_) delete S_i_;
    
    int omp_p=omp_get_max_threads();
    for(int i=0; i<omp_p; ++i)
    {
        delete S_is_[i];
        delete CI_pairs_[i];
    }

    if(x_host_)
        delete x_host_;
    if(rhs_host_)
        delete rhs_host_;

    if(CI_ves_bd_)
        delete CI_ves_bd_;
    
    linear_solver_gmres_.clear();
}

// Performs the following computation:
// velocity(n+1) = updateFarField( bending(n), tension(n) );    // Far-field
// velocity(n+1)+= stokes( bending(n) );                        // Add near-field due to bending
// tension(n+1)  = getTension( velocity(n+1) )                  // Linear solve to compute new tension
// velocity(n+1)+= stokes( tension(n+1) )                       // Add near-field due to tension
// position(n+1) = position(n) + dt*velocity(n+1)
//
// Notes: tension solve is block implicit.
template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
updateJacobiExplicit(const SurfContainer& S_, const value_type &dt, Vec_t& dx)
{
    this->dt_ = dt;
    SolverScheme scheme(JacobiBlockExplicit);
    INFO("Taking a time step using "<<scheme<<" scheme");
    CHK(Prepare(scheme));

    std::auto_ptr<Vec_t> u1 = checkoutVec();
    std::auto_ptr<Vec_t> u2 = checkoutVec();

    // puts u_inf and interaction in pos_vel_
    this->updateFarField();

    // add S[f_b]
    Intfcl_force_.bendingForce(S_, *u1);
    CHK(stokes(*u1, *u2));
    axpy(static_cast<value_type>(1.0), *u2, pos_vel_, pos_vel_);

    // compute tension
    CHK(getTension(pos_vel_, tension_));

    // add S[f_sigma]
    Intfcl_force_.tensileForce(S_, tension_, *u1);
    CHK(stokes(*u1, *u2));
    axpy(static_cast<value_type>(1.0), *u2, pos_vel_, pos_vel_);

    //axpy(dt_, pos_vel_, S_.getPosition(), S_.getPositionModifiable());
    dx.replicate(S_.getPosition());
    axpy(dt_, pos_vel_, dx);

    recycle(u1);
    recycle(u2);

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
updateJacobiGaussSeidel(const SurfContainer& S_, const value_type &dt, Vec_t& dx)
{
    this->dt_ = dt;
    SolverScheme scheme(JacobiBlockGaussSeidel);
    INFO("Taking a time step using "<<scheme<<" scheme");
    CHK(Prepare(scheme));

    std::auto_ptr<Vec_t> u1 = checkoutVec();
    std::auto_ptr<Vec_t> u2 = checkoutVec();
    std::auto_ptr<Vec_t> u3 = checkoutVec();

    // put far field in pos_vel_ and the sum with S[f_b] in u1
    this->updateFarField();
    Intfcl_force_.bendingForce(S_, *u2);
    CHK(stokes(*u2, *u1));
    axpy(static_cast<value_type>(1.0), pos_vel_, *u1, *u1);

    // tension
    CHK(getTension(*u1, tension_));
    Intfcl_force_.tensileForce(S_, tension_, *u1);
    CHK(stokes(*u1, *u2));

    // position rhs
    axpy(static_cast<value_type>(1.0), pos_vel_, *u2, *u1);
    axpy(dt_, *u1, S_.getPosition(), *u1);

    // initial guess
    u2->getDevice().Memcpy(u2->begin(), S_.getPosition().begin(),
        S_.getPosition().size() * sizeof(value_type),
        u2->getDevice().MemcpyDeviceToDevice);

    int iter(params_.time_iter_max);
    int rsrt(params_.time_iter_max);
    value_type tol(params_.time_tol),relres(params_.time_tol);

    enum BiCGSReturn solver_ret;
    Error_t ret_val(ErrorEvent::Success);

    COUTDEBUG("Solving for position");
    solver_ret = linear_solver_vec_(*this, *u2, *u1, rsrt, iter, relres);
    if ( solver_ret  != BiCGSSuccess )
        ret_val = ErrorEvent::DivergenceError;

    COUTDEBUG("Position solve: Total iter = "<<iter<<", relres = "<<tol);
    COUTDEBUG("Checking true relres");
    ASSERT(((*this)(*u2, *u3),
            axpy(static_cast<value_type>(-1), *u3, *u1, *u3),
            relres = sqrt(AlgebraicDot(*u3, *u3))/sqrt(AlgebraicDot(*u1,*u1)),
            relres<tol
           ),
           "relres ("<<relres<<")<tol("<<tol<<")"
           );

    //u2->getDevice().Memcpy(S_.getPositionModifiable().begin(), u2->begin(),
    //    S_.getPosition().size() * sizeof(value_type),
    //    u2->getDevice().MemcpyDeviceToDevice);
    dx.replicate(S_.getPosition());
    axpy(-1, S_.getPosition(), *u2, dx);

    recycle(u1);
    recycle(u2);
    recycle(u3);

    return ret_val;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
updateJacobiImplicit(const SurfContainer& S_, const value_type &dt, Vec_t& dx)
{
    PROFILESTART();
        
    this->dt_ = dt;
    SolverScheme scheme(JacobiBlockImplicit);
    INFO("Taking a time step using "<<scheme<<" scheme\n");
    CHK(Prepare(scheme));
    
    // check out working vec, sca
    std::auto_ptr<Vec_t> x1 = checkoutVec();
    std::auto_ptr<Vec_t> f1 = checkoutVec();
    std::auto_ptr<Vec_t> b1 = checkoutVec();
    std::auto_ptr<Vec_t> xtmp = checkoutVec();
    std::auto_ptr<Vec_t> xstart = checkoutVec();
    std::auto_ptr<Sca_t> b2 = checkoutSca();
    
    // resize
    x1->replicate(S_.getPosition());
    f1->replicate(S_.getPosition());
    b1->replicate(S_.getPosition());
    xtmp->replicate(S_.getPosition());
    xstart->replicate(S_.getPosition());
    b2->replicate(tension_);
    
    // initial guess
    x1->getDevice().Memcpy(x1->begin(), pos_vel_.begin(),
        pos_vel_.size() * sizeof(value_type),
        x1->getDevice().MemcpyDeviceToDevice);

    // farfield velocity
    this->updateFarField();

    // position rhs
    Intfcl_force_.explicitTractionJump(S_, *f1);
    //el_force->updateJacobiImplicit(*xtmp);
    //axpy(static_cast<value_type>(1.0), S_.fc_, *f1, *f1);
    //axpy(static_cast<value_type>(-1.0), fe_, *f1, *f1);
    stokes_.SetDensitySL(f1.get());
    stokes_.SetDensityDL(NULL);
    stokes_.SelfInteraction(*b1);
    axpy(static_cast<value_type>(1.0), pos_vel_, *b1, *b1);
    
    // tension rhs
    S_.div(*b1, *b2);
    //axpy(static_cast<value_type>(-1.0),*b2,*b2);

    // parameters for gmres
    int iter(params_.time_iter_max);
    int rsrt(params_.time_iter_max);
    value_type tol(params_.time_tol),relres(params_.time_tol);

    Error_t ret_val(ErrorEvent::Success);
    INFO("Solving for velocity");
        
    // new code using GMRES Solver
    size_t vsz(stokesBlockSize()), tsz(tensionBlockSize());
    ASSERT(S_.getPosition().size()==vsz,"Bad sizes");
    size_t N_size = vsz+tsz;
    // copy device type to value_type array to call GMRES
    // copy to unknown solution
    x1->getDevice().Memcpy(x_host_, x1->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    tension_.getDevice().Memcpy(x_host_+vsz, tension_.begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    
    // copy to rhs
    b1->getDevice().Memcpy(rhs_host_    , b1->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    b2->getDevice().Memcpy(rhs_host_+vsz, b2->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        
    // solve the linear system using gmres
    MPI_Comm comm=MPI_COMM_WORLD;
    pvfmm::Profile::Tic("LinearSolverVelocity", &comm, true);
    int solver_ret = linear_solver_gmres_(JacobiImplicitApply, JacobiImplicitPrecond, x_host_, rhs_host_, 
            relres, relres*0, N_size, iter, rsrt);
    MKL_Free_Buffers();
    pvfmm::Profile::Toc();
    
    // copy host to device
    x1->getDevice().Memcpy(x1->begin(), x_host_    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    tension_.getDevice().Memcpy(tension_.begin(), x_host_+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    // end of new code using GMRES Solver

if(true){
    // developing code for col
    INFO("Begin of contact resolving steps.\n");
    // reset contact force
    axpy(static_cast<value_type>(-0), S_.fc_, S_.fc_);
     
    //if(params_.min_sep_dist>0)
    //{
        /* new parallel code
        // the candidate position
        axpy(dt_, *x1, S_.getPosition(), *xtmp);
        // get collision
        pvfmm::Profile::Tic("GetContactVolume", &comm, true);
	GetStartPosAll(S_.getPosition(), *xtmp, *f1);
        ParallelGetVolumeAndGradient(*f1, *xtmp);
        //ParallelGetVolumeAndGradient(S_.getPosition(), *xtmp);
        pvfmm::Profile::Toc();
        */
   
        // old sequential code
        // pos_s stores the start configuration
        // pos_e stores the end configuration
        // vGrad stores the gradient of contact volumes
        int np_up = 2*params_.sh_order*(params_.sh_order+1);
        if(params_.col_upsample)
            np_up = 2*params_.upsample_freq*(params_.upsample_freq+1);
        static std::vector<value_type> pos_s(np_up*COORD_DIM*nv_, 0.0);
        static std::vector<value_type> pos_e(np_up*COORD_DIM*nv_, 0.0);
        static std::vector<value_type> vGrad(np_up*COORD_DIM*nv_, 0.0);
        static pvfmm::Vector<value_type> pos_s_pole, pos_e_pole;

        
        //GetColPosAll(S_.getPosition(), pos_s, pos_s_pole);
        if (params_.col_upsample)
        {
            S_.resample(params_.upsample_freq, &S_up_);
            
            std::auto_ptr<Vec_t> u1 = checkoutVec();
            std::auto_ptr<Vec_t> u2 = checkoutVec();
            u1->replicate(S_up_->getPosition());
            u2->replicate(S_up_->getPosition());

            axpy(static_cast<value_type>(-0.0001), S_up_->getNormal(), S_up_->getPosition(), S_up_->getPositionModifiable());
            Resample(S_up_->getPositionModifiable(), sht_upsample_, sht_, *u1, *u2, *xstart);

            recycle(u1);
            recycle(u2);
        }
        else
        {
            axpy(static_cast<value_type>(-0.0001), S_.getNormal(), S_.getPosition(), *xstart);
        }
        GetColPosAll(*xstart, pos_s, pos_s_pole);
        axpy(dt_, *x1, S_.getPosition(), *xtmp);
        GetColPosAll(*xtmp, pos_e, pos_e_pole);

        sum_num_cvs_ = 0;
        num_cvs_ = 0;
        IV_.clear();
        IS_BDV_.clear();
        vgrad_.getDevice().Memset(vgrad_.begin(), 0, sizeof(value_type)*vgrad_.size());
        std::memset(&vgrad_ind_[0], 0, vgrad_ind_.size()*sizeof(vgrad_ind_[0]));

        if(params_.periodic_length > 0)
        {
            TransferVesicle(pos_s, pos_e, pos_s_pole, pos_e_pole);
        }
        /*
        CI_bd_.getVolumeAndGradient(IV_, IS_BDV_, num_cvs_, vGrad, vgrad_ind_, pos_s, pos_e, &pos_s_pole[0], &pos_e_pole[0],
                params_.min_sep_dist, fixed_bd->tri_vertices, nv_, params_.periodic_length);
        sum_num_cvs_ = num_cvs_;
        vgrad_.getDevice().Memcpy(vgrad_.begin(), &vGrad.front(), 
                vgrad_.size() * sizeof(value_type),
                vgrad_.getDevice().MemcpyHostToDevice);
        */
        // end of sequential code
        
        // new parallel code with boundary
        //ParallelGetVolumeAndGradientWithBoundary(*xstart, *xtmp, 
        //        fixed_bd->tri_vertices, fixed_bd->num_vertices_per_patch_1d, fixed_bd->num_patches);
        ParallelGetVolumeAndGradientWithBoundary(*xstart, *xtmp, 
                NULL, 0, 0);
        // end of new parallel code with boundary
    //}
        
    // prepare
    std::auto_ptr<Vec_t> col_dx = checkoutVec();
    std::auto_ptr<Vec_t> col_f = checkoutVec();
    std::auto_ptr<Sca_t> col_tension = checkoutSca();
    col_dx->replicate(*x1);
    col_f->replicate(*x1);
    col_tension->replicate(tension_);
    int resolveCount = 0;
    // begin of developing code for col 
    pvfmm::Profile::Tic("ContactResolveIters", &comm, true);
    while(sum_num_cvs_ >0 && params_.min_sep_dist>0)
    {
        //COUT("IV_.size(): "<<IV_.size());
        //COUT("num_cvs_: "<<num_cvs_);
        INFO("sum_num_cvs_: "<<sum_num_cvs_);
        //COUT("Norm of vgrad: " << AlgebraicDot(vgrad_, vgrad_));

        // copy contact volume gradient to vgrad_
        ///vgrad_.getDevice().Memcpy(vgrad_.begin(), &vGrad.front(), 
        ///        vgrad_.size() * sizeof(value_type),
        ///        vgrad_.getDevice().MemcpyHostToDevice);

        // col_lambda stores the force magnitude
        Arr_t col_lambda(num_cvs_);
        // cvs stores the contact volumes
        Arr_t cvs(num_cvs_);
        // copy contact volumes to cvs
        cvs.getDevice().Memcpy(cvs.begin(), &IV_.front(), num_cvs_ * sizeof(value_type), cvs.getDevice().MemcpyHostToDevice);

        // solve for u, tension, lambda using minmap LCP solver
        //INFO("Begin of SolveLCP.");
        //SolveLCP(*col_dx, *col_tension, col_lambda, cvs);
        //INFO("End of SolveLCP.");
        // get contact force
        //CVJacobianTrans(col_lambda, *col_f);
        // accumulate contact force to S_.fc_
        //axpy(static_cast<value_type>(1.0), *col_f, S_.fc_, S_.fc_);
 
        // form lcp matrix
        INFO("before formlcp");
        pvfmm::Profile::Tic("FormLCPMatrix", &comm, true);
        ParallelFormLCPMatrixSparse(parallel_lcp_matrix_);
        pvfmm::Profile::Toc();
        INFO("after formlcp");
        typename std::map<std::pair<size_t, size_t>, value_type>::iterator got_lcp_matrix_ij;

        //COUT("size of lcp matrix: "<<parallel_lcp_matrix_.size());
        //for(got_lcp_matrix_ij = parallel_lcp_matrix_.begin(); got_lcp_matrix_ij!=parallel_lcp_matrix_.end(); got_lcp_matrix_ij++)
        //{
        //    COUT("entry("<<got_lcp_matrix_ij->first.first<<", "<<got_lcp_matrix_ij->first.second<<") = "<<
        //            got_lcp_matrix_ij->second);
        //}

        ///lcp_matrix_.resize(num_cvs_*num_cvs_);
        ///FormLCPMatrixSparse(lcp_matrix_);
        ///INFO("lcp_matrix sparse: "<<lcp_matrix_);
        
        // solve lcp
        INFO("before solvelcp");
        pvfmm::Profile::Tic("SolveLCP", &comm, true);
        ParallelSolveLCPSmall(col_lambda, cvs);
        pvfmm::Profile::Toc();
        INFO("after solvelcp");
        ///SolveLCPSmall(col_lambda, cvs);
        
        // get contact force
        //COUT("getting contact force.");
        pvfmm::Profile::Tic("CVJacobianTrans", &comm, true);
        ParallelCVJacobianTrans(col_lambda, *col_f);
        pvfmm::Profile::Toc();
        ///CVJacobianTrans(col_lambda, *col_f);
                
        // get displacement in u and tension due to contact force
        //COUT("getting dx dtension update.");
        pvfmm::Profile::Tic("GetDx", &comm, true);
        GetDx(*col_dx,*col_tension,*col_f);
        pvfmm::Profile::Toc();
        
        // update velocity
        axpy(static_cast<value_type>(1.0), *col_dx, *x1, *x1);
      
        resolveCount++;
        INFO("Col iter#: "<<resolveCount);
        //COUT("cvs: "<<cvs);
        //COUT("lambda: "<<col_lambda);
      
        // accumulate contact force to S_.fc_ and update tension
        //COUT("accumulating contact force.");
        //axpy(static_cast<value_type>(1.0), *col_f, S_.fc_, S_.fc_);
        value_type is_bdv = 1;
        value_type is_bdv_glb = 1;
        for(int iv=0; iv<col_lambda.size(); iv++)
        {
            col_lambda.begin()[iv] *= IS_BDV_[iv];
            is_bdv *= IS_BDV_[iv];
        }
        MPI_Allreduce(&is_bdv, &is_bdv_glb, 1, MPI_DOUBLE, MPI_MIN, comm);
        if(is_bdv_glb)
        {
            //COUT("no boundary vesicle contact.");
            //axpy(static_cast<value_type>(1.0), *col_tension, tension_, tension_);
        }
        else
        {
            //COUT("there is boundary vesicle contact.");
            //ParallelCVJacobianTrans(col_lambda, *col_f);
            //COUT("boundary contact force ignored");
            //GetDx(*col_dx,*col_tension,*col_f);
            //COUT("new tension update");
            //axpy(static_cast<value_type>(1.0), *col_tension, tension_, tension_);
        }
        axpy(static_cast<value_type>(1.0), *col_f, S_.fc_, S_.fc_);

        // test if still have contact
        //

        /*
        // get new candidate position
        axpy(dt_, *x1, S_.getPosition(), *xtmp);
        INFO("before getvolume");
        pvfmm::Profile::Tic("GetContactVolume", &comm, true);
	GetStartPosAll(S_.getPosition(), *xtmp, *f1);
        ParallelGetVolumeAndGradient(*f1, *xtmp);
        //ParallelGetVolumeAndGradient(S_.getPosition(), *xtmp);
        pvfmm::Profile::Toc();
        // end of new parallel 
        */
        
        
        // old code sequential
        axpy(dt_, *x1, S_.getPosition(), *xtmp);
        GetColPosAll(*xtmp, pos_e, pos_e_pole);

        sum_num_cvs_ = 0;
        num_cvs_ = 0;
        IV_.clear();
        IS_BDV_.clear();
        vgrad_.getDevice().Memset(vgrad_.begin(), 0, sizeof(value_type)*vgrad_.size());
        std::memset(&vgrad_ind_[0], 0, vgrad_ind_.size()*sizeof(vgrad_ind_[0]));

        if(params_.periodic_length > 0)
        {
            // TODO: add GetColPosAll(X_s, pos_s, pos_s_pole);
            TransferVesicle(pos_s, pos_e, pos_s_pole, pos_e_pole);
        }
        /*
        CI_bd_.getVolumeAndGradient(IV_, IS_BDV_, num_cvs_, vGrad, vgrad_ind_, pos_s, pos_e, &pos_s_pole[0], &pos_e_pole[0],
                params_.min_sep_dist, fixed_bd->tri_vertices, nv_, params_.periodic_length);
        sum_num_cvs_ = num_cvs_;
        vgrad_.getDevice().Memcpy(vgrad_.begin(), &vGrad.front(), 
                vgrad_.size() * sizeof(value_type),
                vgrad_.getDevice().MemcpyHostToDevice);
        */

        ///GetColPos(*xtmp, pos_e, pos_e_pole);
        ///if(params_.periodic_length > 0)
        ///{
            ///GetColPos(S_.getPosition(), pos_s, pos_s_pole);
            ///TransferVesicle(pos_s, pos_e, pos_s_pole, pos_e_pole);
        ///}
        ///CI_.getVolumeAndGradient(IV, num_cvs_, vGrad, vgrad_ind_, pos_s, pos_e, &pos_s_pole[0], &pos_e_pole[0],
        ///        params_.min_sep_dist, params_.periodic_length);
        // end of old code sequential
        
        // new parallel contact 
        ParallelGetVolumeAndGradientWithBoundary(*xstart, *xtmp, 
                NULL, 0, 0);
                //fixed_bd->tri_vertices, fixed_bd->num_vertices_per_patch_1d, fixed_bd->num_patches);
        // end of new parallel contact

        INFO("after getvolume");
    }
    pvfmm::Profile::Toc();
    
    recycle(col_dx);
    recycle(col_tension);
    recycle(col_f);
    // end of developing code for col 
}
        
    COUTDEBUG("Position solve: Total iter = "<<iter<<", relres = "<<tol);
    COUTDEBUG("Checking true relres");
    /*
    ASSERT(((*this)(*u2, *u3),
            axpy(static_cast<value_type>(-1), *u3, *u1, *u3),
            relres = sqrt(AlgebraicDot(*u3, *u3))/sqrt(AlgebraicDot(*u1,*u1)),
            relres<tol
           ),
           "relres ("<<relres<<")<tol("<<tol<<")"
           );
    */

    // set dx
    dx.replicate(S_.getPosition());
    axpy(dt_, *x1, dx);
    // set pos_vel_
    axpy(static_cast<value_type>(1.0), *x1, pos_vel_);
    // TODO: update pos_vel_ so that pos_vel_ .* S_.fc_ = 0;

    // filter fc
    //sht_filter_high(S_.fc_, S_.fc_, &sht_, params_.rep_exponent);

    // filter velocity
    //sht_filter_high(pos_vel_, pos_vel_, &sht_, params_.rep_exponent);
    
    // filter tension_
    //sht_filter_high_sca(tension_, tension_, &sht_, params_.rep_exponent);
    
    //sht_filter_high_sca(el_force->Vm_, el_force->Vm_, &sht_, params_.rep_exponent);
    
    // print the max abs vel
    INFO("vel maxabs: "<<MaxAbs(pos_vel_)<<"\n");

    // clear memory
    recycle(x1);
    recycle(f1);
    recycle(b1);
    recycle(b2);
    recycle(xtmp);
    recycle(xstart);

    PROFILEEND("",0);
    return ret_val;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
updateExpand(const SurfContainer& S_, const value_type &dt, Vec_t& dx)
{
    PROFILESTART();
        
    this->dt_ = dt;
    SolverScheme scheme(JacobiBlockImplicit);
    INFO("Taking a time step using "<<scheme<<" scheme\n");
    CHK(Prepare(scheme));
    
    // check out working vec, sca
    std::auto_ptr<Vec_t> x1 = checkoutVec();
    std::auto_ptr<Vec_t> f1 = checkoutVec();
    std::auto_ptr<Vec_t> b1 = checkoutVec();
    std::auto_ptr<Vec_t> xtmp = checkoutVec();
    std::auto_ptr<Vec_t> xstart = checkoutVec();
    std::auto_ptr<Sca_t> b2 = checkoutSca();
    
    // resize
    x1->replicate(S_.getPosition());
    f1->replicate(S_.getPosition());
    b1->replicate(S_.getPosition());
    xtmp->replicate(S_.getPosition());
    xstart->replicate(S_.getPosition());
    b2->replicate(tension_);
    
    // initial guess
    x1->getDevice().Memcpy(x1->begin(), pos_vel_.begin(),
        pos_vel_.size() * sizeof(value_type),
        x1->getDevice().MemcpyDeviceToDevice);
    
    int np_up = 2*params_.sh_order*(params_.sh_order+1);
    if(params_.col_upsample)
        np_up = 2*params_.upsample_freq*(params_.upsample_freq+1);
        
    static std::vector<value_type> pos_s(np_up*COORD_DIM*nv_, 0.0);
    static std::vector<value_type> pos_e(np_up*COORD_DIM*nv_, 0.0);
    static pvfmm::Vector<value_type> pos_s_pole, pos_e_pole;

    if (params_.col_upsample)
    {
        S_.resample(params_.upsample_freq, &S_up_);
            
        std::auto_ptr<Vec_t> u1 = checkoutVec();
        std::auto_ptr<Vec_t> u2 = checkoutVec();
        u1->replicate(S_up_->getPosition());
        u2->replicate(S_up_->getPosition());

        axpy(static_cast<value_type>(-0.0001), S_up_->getNormal(), S_up_->getPosition(), S_up_->getPositionModifiable());
        Resample(S_up_->getPositionModifiable(), sht_upsample_, sht_, *u1, *u2, *xstart);

        recycle(u1);
        recycle(u2);
    }
    else
    {
        axpy(static_cast<value_type>(-0.0001), S_.getNormal(), S_.getPosition(), *xstart);
    }
        
    // start config
    GetColPosAll(*xstart, pos_s, pos_s_pole);

    // expand vesicle
    Arr_t expand_step;
    expand_step.resize(nv_);
    for(int i=0; i<nv_; i++)
    {
        if(ves_stop[i]==false)
            expand_step.begin()[i] = 0.02;
        else
            expand_step.begin()[i] = 0.0;
    }
    av(expand_step, pos_vel_, *x1);
    axpy(1.0, *x1, S_.getPosition(), *xtmp);
    GetColPosAll(*xtmp, pos_e, pos_e_pole);

    // test collision
    sum_num_cvs_ = 0;
    num_cvs_ = 0;
    IV_.clear();
    IS_BDV_.clear();
    vgrad_.getDevice().Memset(vgrad_.begin(), 0, sizeof(value_type)*vgrad_.size());
    std::memset(&vgrad_ind_[0], 0, vgrad_ind_.size()*sizeof(vgrad_ind_[0]));
    if(params_.periodic_length > 0)
    {
        TransferVesicle(pos_s, pos_e, pos_s_pole, pos_e_pole);
    }
    // new parallel code with boundary
    ParallelGetVolumeAndGradientWithBoundary(*xstart, *xtmp, 
            NULL, 0, 0);
            //fixed_bd->tri_vertices, fixed_bd->num_vertices_per_patch_1d, fixed_bd->num_patches);

    // update expand coefficient
    for(int i=0; i<nv_; i++)
    {
        for(int j=0; j<np_up*COORD_DIM; j++)
        {
            if(vgrad_ind_[np_up*COORD_DIM*i+j]>0)
            {
                ves_stop[i] = true;
                expand_step.begin()[i] = 0.0;
                break;
            }
        }
    }
    for(int i=0; i<nv_; i++)
        expand_by[i] += expand_step.begin()[i];
            
    // set dx
    dx.replicate(S_.getPosition());
    av(expand_step, pos_vel_, dx);
    
    
    // clear memory
    recycle(x1);
    recycle(f1);
    recycle(b1);
    recycle(b2);
    recycle(xtmp);
    recycle(xstart);

    PROFILEEND("",0);
    Error_t ret_val(ErrorEvent::Success);
    return ret_val;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
updateImplicit(const SurfContainer& S_, const value_type &dt, Vec_t& dx)
{
    PROFILESTART();
    this->dt_ = dt;
    SolverScheme scheme(GloballyImplicit);
    INFO("Taking a time step using "<<scheme<<" scheme");
    CHK(Prepare(scheme));

    if (params_.solve_for_velocity) {
        CHK(AssembleRhsVel(parallel_rhs_, dt_, scheme));
    } else {
        CHK(AssembleRhsPos(parallel_rhs_, dt_, scheme));
    }

    Error_t err=ErrorEvent::Success;
    if(err==ErrorEvent::Success) err=AssembleInitial(parallel_u_, dt_, scheme);
    if(err==ErrorEvent::Success) err=Solve(parallel_rhs_, parallel_u_, dt_, scheme);
    if(err==ErrorEvent::Success) err=Update(parallel_u_);
    
    INFO("vel maxabs: "<<MaxAbs(pos_vel_)<<"\n");

    if(0)
    if (params_.solve_for_velocity && !params_.pseudospectral){ // Save velocity field to VTK
      std::auto_ptr<Vec_t> vel_ = checkoutVec();
      std::auto_ptr<Sca_t> ten_ = checkoutSca();
      { // Set vel_, ten_
          typename PVec_t::iterator i(NULL);
          typename PVec_t::size_type rsz;
          CHK(parallel_u_->GetArray(i, rsz));
          size_t vsz(stokesBlockSize()), tsz(tensionBlockSize());
          ASSERT(rsz==vsz+tsz,"Bad sizes");

          vel_->replicate(pos_vel_);
          ten_->replicate(tension_);
          {
              std::auto_ptr<Vec_t> voxSh = checkoutVec();
              std::auto_ptr<Sca_t> tSh   = checkoutSca();
              std::auto_ptr<Vec_t> wrk   = checkoutVec();

              voxSh->replicate(*vel_);
              tSh->replicate(*ten_);
              wrk->replicate(*vel_);

              voxSh->getDevice().Memcpy(voxSh->begin(), i    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
              tSh  ->getDevice().Memcpy(tSh  ->begin(), i+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);

              sht_.backward(*voxSh, *wrk, *vel_);
              sht_.backward(*tSh  , *wrk, *ten_);

              recycle(voxSh);
              recycle(tSh);
              recycle(wrk);
          }
          CHK(parallel_u_->RestoreArray(i));
      }

      { // Set DensitySL
          std::auto_ptr<Vec_t> f   = checkoutVec();
          Intfcl_force_.explicitTractionJump(S_, *f);
          { // Add implicit traction jump
              std::auto_ptr<Vec_t> Du  = checkoutVec();
              std::auto_ptr<Vec_t> fi  = checkoutVec();
              axpy(dt_, *vel_, *Du);
              Intfcl_force_.implicitTractionJump(S_, *Du, *ten_, *fi);
              axpy(static_cast<value_type>(1.0), *fi, *f, *f);
              recycle(Du);
              recycle(fi);
          }
          stokes_.SetDensitySL(f.get());
          recycle(f);
      }

      if( ves_props_.has_contrast ){ // Set DensityDL
          std::auto_ptr<Vec_t> lcoeff_vel  = checkoutVec();
          av(ves_props_.dl_coeff, *vel_, *lcoeff_vel);
          stokes_.SetDensityDL(lcoeff_vel.get());
          recycle(lcoeff_vel);
      } else {
          stokes_.SetDensityDL(NULL);
      }

      if(0){ // Print error
        std::auto_ptr<Vec_t> Sf = checkoutVec();
        stokes_(*Sf);

        { // Add bg_vel
          std::auto_ptr<Vec_t> bg_vel = checkoutVec();
          bg_vel->replicate(S_.getPosition());
          CHK(BgFlow(*bg_vel, dt));
          axpy(static_cast<value_type>(1.0), *bg_vel, *Sf, *Sf);
          recycle(bg_vel);
        }
        if( ves_props_.has_contrast ){
          Arr_t* inv_vel_coeff = new Arr_t;
          inv_vel_coeff->resize(ves_props_.vel_coeff.size());
          //xInv(ves_props_.vel_coeff,*inv_vel_coeff);
          { // inv_vel_coeff = 1.0/ves_props_.vel_coeff
            const value_type* in=ves_props_.vel_coeff .begin();
            value_type*      out=       inv_vel_coeff->begin();
            for(long i=0;i<inv_vel_coeff->size();i++) out[i]=1.0/in[i];
          }
          av(*inv_vel_coeff, *Sf, *Sf);
          delete inv_vel_coeff;
        }

        axpy(static_cast<value_type>(-1.0), *vel_, *Sf, *Sf);
        pvfmm::Matrix<value_type> dv(1,Sf->size(),Sf->begin());
        pvfmm::Matrix<value_type> dv2=dv*dv.Transpose();

        pvfmm::Matrix<value_type> v(1, vel_->size(),vel_->begin());
        pvfmm::Matrix<value_type> v2=v*v.Transpose();

        std::cout<<"GMRES Error = "<<sqrt(dv2[0][0])<<"/"<<sqrt(v2[0][0])<<'\n';
        recycle(Sf);
      }else{ // Write VTK
        int myrank, np;
        MPI_Comm comm=MPI_COMM_WORLD;
        MPI_Comm_rank(comm,&myrank);
        MPI_Comm_size(comm,&np);

        typedef float VTKReal;
        struct VTKData{
          std::vector<VTKReal> coord;
          std::vector<VTKReal> value;

          std::vector<int32_t> connect;
          std::vector<int32_t> offset ;
          std::vector<uint8_t> types  ;
        };
        VTKData vtk_data;

        { // Set vtk_data
          value_type range[6];
          range[0]=-4; range[1]=-4; range[2]=-4;
          range[3]= 4; range[4]= 4; range[5]= 4;
          long gridpt_cnt=40;
          long data_dof=3;

          std::vector<VTKReal>& coord=vtk_data.coord;
          std::vector<VTKReal>& value=vtk_data.value;

          std::vector<int32_t>& connect=vtk_data.connect;
          std::vector<int32_t>& offset =vtk_data.offset ;
          std::vector<uint8_t>& types  =vtk_data.types  ;

          { // Set coord, connect, offset, types
            long Ngrid=(gridpt_cnt-1)*(gridpt_cnt-1)*(gridpt_cnt-1);
            long idx_start=(Ngrid*(myrank+0))/np;
            long idx_end  =(Ngrid*(myrank+1))/np;

            long grid_idx=0;
            for(int i0=0;i0<(gridpt_cnt-1);i0++)
            for(int i1=0;i1<(gridpt_cnt-1);i1++)
            for(int i2=0;i2<(gridpt_cnt-1);i2++){
              if(idx_start<=grid_idx && grid_idx<idx_end){
                for(int j0=0;j0<2;j0++)
                for(int j1=0;j1<2;j1++)
                for(int j2=0;j2<2;j2++){
                  connect.push_back(coord.size()/3);
                  coord.push_back((i0+j0)/(gridpt_cnt-1.0)*(range[3]-range[0])+range[0]);
                  coord.push_back((i1+j1)/(gridpt_cnt-1.0)*(range[4]-range[1])+range[1]);
                  coord.push_back((i2+j2)/(gridpt_cnt-1.0)*(range[5]-range[2])+range[2]);
                  for(int j=0;j<data_dof;j++) value.push_back(0.0);
                }
                offset.push_back(connect.size());
                types.push_back(11);
              }
              grid_idx++;
            }
          }

          { // Set vtk_data.value
            pvfmm::Vector<value_type> coord(vtk_data.coord.size()), value(vtk_data.value.size());
            for(long i=0;i<coord.Dim();i++) coord[i]=vtk_data.coord[i];
            stokes_.SetSrcCoord(S_.getPosition(),100,100);
            stokes_.SetTrgCoord(&coord);
            value=stokes_();
            { // Add BgVel
              long ntrg=coord.Dim()/COORD_DIM;
              long nv=(ntrg+4-1)/4;
              long mv=4;

              Vec_t c(nv,1), bgvel(nv,1);
              value_type* c_=c.begin();
              for(long i0=0;i0<nv;i0++){
                for(long i1=0;i1<mv;i1++){
                  long i=i0*mv+i1;
                  if(i<ntrg){
                    c_[(i0*COORD_DIM+0)*mv+i1]=coord[i*COORD_DIM+0];
                    c_[(i0*COORD_DIM+1)*mv+i1]=coord[i*COORD_DIM+1];
                    c_[(i0*COORD_DIM+2)*mv+i1]=coord[i*COORD_DIM+2];
                  }
                }
              }

              bg_flow_(c, 0, bgvel);
              value_type* bgvel_=bgvel.begin();
              for(long i0=0;i0<nv;i0++){
                for(long i1=0;i1<mv;i1++){
                  long i=i0*mv+i1;
                  if(i<ntrg){
                    value[i*COORD_DIM+0]+=bgvel_[(i0*COORD_DIM+0)*mv+i1];
                    value[i*COORD_DIM+1]+=bgvel_[(i0*COORD_DIM+1)*mv+i1];
                    value[i*COORD_DIM+2]+=bgvel_[(i0*COORD_DIM+2)*mv+i1];
                  }
                }
              }
            }
            stokes_.SetTrgCoord(NULL);
            for(long i=0;i<value.Dim();i++) vtk_data.value[i]=value[i];
          }
        }

        const char* fname="vis/vel";
        { // WriteVTK
          std::vector<VTKReal>& coord=vtk_data.coord;
          std::vector<VTKReal>& value=vtk_data.value;

          std::vector<int32_t>& connect=vtk_data.connect;
          std::vector<int32_t>& offset =vtk_data.offset;
          std::vector<uint8_t>& types  =vtk_data.types;

          int pt_cnt=coord.size()/3;
          int cell_cnt=types.size();

          std::vector<int32_t> mpi_rank;  //MPI_Rank at points.
          int new_myrank=myrank;//rand();
          mpi_rank.resize(pt_cnt,new_myrank);

          bool isLittleEndian;
          { // Set isLittleEndian
            uint16_t number = 0x1;
            uint8_t *numPtr = (uint8_t*)&number;
            isLittleEndian=(numPtr[0] == 1);
          }

          //Open file for writing.
          std::stringstream vtufname;
          vtufname<<fname<<std::setfill('0')<<std::setw(6)<<myrank<<".vtu";
          std::ofstream vtufile;
          vtufile.open(vtufname.str().c_str());
          if(!vtufile.fail()){ // write .vtu
            size_t data_size=0;
            vtufile<<"<?xml version=\"1.0\"?>\n";
            if(isLittleEndian) vtufile<<"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
            else               vtufile<<"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n";
            //===========================================================================
            vtufile<<"  <UnstructuredGrid>\n";
            vtufile<<"    <Piece NumberOfPoints=\""<<pt_cnt<<"\" NumberOfCells=\""<<cell_cnt<<"\">\n";

            //---------------------------------------------------------------------------
            vtufile<<"      <Points>\n";
            vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<3<<"\" Name=\"Position\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
            data_size+=sizeof(uint32_t)+coord.size()*sizeof(VTKReal);
            vtufile<<"      </Points>\n";
            //---------------------------------------------------------------------------
            vtufile<<"      <PointData>\n";
            if(value.size()){ // value
              vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value.size()/pt_cnt<<"\" Name=\""<<"value"<<"\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
              data_size+=sizeof(uint32_t)+value.size()*sizeof(VTKReal);
            }
            { // mpi_rank
              vtufile<<"        <DataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
              data_size+=sizeof(uint32_t)+mpi_rank.size()*sizeof(int32_t);
            }
            vtufile<<"      </PointData>\n";
            //---------------------------------------------------------------------------
            //---------------------------------------------------------------------------
            vtufile<<"      <Cells>\n";
            vtufile<<"        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
            data_size+=sizeof(uint32_t)+connect.size()*sizeof(int32_t);
            vtufile<<"        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
            data_size+=sizeof(uint32_t)+offset.size() *sizeof(int32_t);
            vtufile<<"        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
            data_size+=sizeof(uint32_t)+types.size()  *sizeof(uint8_t);
            vtufile<<"      </Cells>\n";
            //---------------------------------------------------------------------------
            //vtufile<<"      <CellData>\n";
            //vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" Name=\"Velocity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
            //vtufile<<"      </CellData>\n";
            //---------------------------------------------------------------------------

            vtufile<<"    </Piece>\n";
            vtufile<<"  </UnstructuredGrid>\n";
            //===========================================================================
            vtufile<<"  <AppendedData encoding=\"raw\">\n";
            vtufile<<"    _";

            int32_t block_size;
            block_size=coord   .size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&coord   [0], coord   .size()*sizeof(VTKReal));
            if(value.size()){ // value
              block_size=value .size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&value   [0], value   .size()*sizeof(VTKReal));
            }
            block_size=mpi_rank.size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&mpi_rank[0], mpi_rank.size()*sizeof(int32_t));

            block_size=connect .size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&connect [0], connect .size()*sizeof(int32_t));
            block_size=offset  .size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&offset  [0], offset  .size()*sizeof(int32_t));
            block_size=types   .size()*sizeof(uint8_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&types   [0], types   .size()*sizeof(uint8_t));

            vtufile<<"\n";
            vtufile<<"  </AppendedData>\n";
            //===========================================================================
            vtufile<<"</VTKFile>\n";
            vtufile.close();
          }
          if(!myrank){ // write .pvtu
            std::stringstream pvtufname;
            pvtufname<<fname<<".pvtu";
            std::ofstream pvtufile;
            pvtufile.open(pvtufname.str().c_str());
            if(!pvtufile.fail()){
              pvtufile<<"<?xml version=\"1.0\"?>\n";
              pvtufile<<"<VTKFile type=\"PUnstructuredGrid\">\n";
              pvtufile<<"  <PUnstructuredGrid GhostLevel=\"0\">\n";
              pvtufile<<"      <PPoints>\n";
              pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<3<<"\" Name=\"Position\"/>\n";
              pvtufile<<"      </PPoints>\n";
              pvtufile<<"      <PPointData>\n";
              if(value.size()){ // value
                pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value.size()/pt_cnt<<"\" Name=\""<<"value"<<"\"/>\n";
              }
              { // mpi_rank
                pvtufile<<"        <PDataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\"/>\n";
              }
              pvtufile<<"      </PPointData>\n";
              {
                // Extract filename from path.
                std::stringstream vtupath;
                vtupath<<'/'<<fname;
                std::string pathname = vtupath.str();
                unsigned found = pathname.find_last_of("/\\");
                std::string fname_ = pathname.substr(found+1);
                //char *fname_ = (char*)strrchr(vtupath.str().c_str(), '/') + 1;
                //std::string fname_ = boost::filesystem::path(fname).filename().string().
                for(int i=0;i<np;i++) pvtufile<<"      <Piece Source=\""<<fname_<<std::setfill('0')<<std::setw(6)<<i<<".vtu\"/>\n";
              }
              pvtufile<<"  </PUnstructuredGrid>\n";
              pvtufile<<"</VTKFile>\n";
              pvtufile.close();
            }
          }
        }
      }

      recycle(vel_);
      recycle(ten_);
    }

    dx.replicate(S_.getPosition());
    if (params_.solve_for_velocity){
        axpy(dt, pos_vel_, dx);
    } else {
        axpy(-1.0, S_.getPosition(), pos_vel_, dx);
    }

    PROFILEEND("",0);
    return err;
}

template<typename SurfContainer, typename Interaction>
size_t InterfacialVelocity<SurfContainer, Interaction>::stokesBlockSize() const{

    return (params_.pseudospectral ?
        S_.getPosition().size() :
        S_.getPosition().getShOrder()*(S_.getPosition().getShOrder()+2)*S_.getPosition().getNumSubFuncs() ); /* (p+1)^2-1 (last freq doesn't have a cosine */
}

template<typename SurfContainer, typename Interaction>
size_t InterfacialVelocity<SurfContainer, Interaction>::tensionBlockSize() const{
    return stokesBlockSize()/3;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::Prepare(const SolverScheme &scheme) const
{
    PROFILESTART();

    if (pos_vel_.size() != S_.getPosition().size()){
      COUT(pos_vel_.size());
      COUT(S_.getPosition().size());
      COUT(tension_.size());
      COUT("Zeroing pos_vel_ and tension_!!!!");
      COUTDEBUG("Resizing the containers");
      pos_vel_.replicate(S_.getPosition());
      tension_.replicate(S_.getPosition());

      COUTDEBUG("zeroing content of velocity and tension arrays");
      pos_vel_.getDevice().Memset(pos_vel_.begin(), 0, sizeof(value_type)*pos_vel_.size());
      tension_.getDevice().Memset(tension_.begin(), 0, sizeof(value_type)*tension_.size());
    }

    ASSERT(pos_vel_.size() == S_.getPosition().size(), "inccorrect size");
    ASSERT(3*tension_.size() == S_.getPosition().size(), "inccorrect size");
    ASSERT(ves_props_.dl_coeff.size() == S_.getPosition().getNumSubs(), "inccorrect size");
    ASSERT(ves_props_.vel_coeff.size() == S_.getPosition().getNumSubs(), "inccorrect size");
    ASSERT(ves_props_.bending_modulus.size() == S_.getPosition().getNumSubs(), "inccorrect size");

    stokes_.SetSrcCoord(S_.getPosition());
    stokes_.SetTrgCoord(NULL);
    
    if (!precond_configured_ && params_.time_precond!=NoPrecond)
        ConfigurePrecond(params_.time_precond);
    
    //!@bug doesn't support repartitioning
    if (!psolver_configured_ && scheme==GloballyImplicit){
        ASSERT(parallel_solver_ != NULL, "need a working parallel solver");
        CHK(ConfigureSolver(scheme));
    }

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ConfigureSolver(const SolverScheme &scheme) const
{
    PROFILESTART();
    ASSERT(scheme==GloballyImplicit, "Unsupported scheme");
    COUTDEBUG("Configuring the parallel solver");

    typedef typename PSolver_t::matvec_type POp;
    typedef typename PSolver_t::vec_type PVec;
    typedef typename PVec::size_type size_type;

    // Setting up the operator
    size_type sz(stokesBlockSize() + tensionBlockSize());
    CHK(parallel_solver_->LinOpFactory(&parallel_matvec_));
    CHK(parallel_matvec_->SetSizes(sz,sz));
    CHK(parallel_matvec_->SetName("Vesicle interaction"));
    CHK(parallel_matvec_->SetContext(static_cast<const void*>(this)));
    CHK(parallel_matvec_->SetApply(ImplicitApply));
    CHK(parallel_matvec_->Configure());

    // setting up the rhs
    CHK(parallel_solver_->VecFactory(&parallel_rhs_));
    CHK(parallel_rhs_->SetSizes(sz));
    CHK(parallel_rhs_->SetName("rhs"));
    CHK(parallel_rhs_->Configure());

    CHK(parallel_rhs_->ReplicateTo(&parallel_u_));
    CHK(parallel_u_->SetName("solution"));

    // setting up the solver
    CHK(parallel_solver_->SetOperator(parallel_matvec_));
    CHK(parallel_solver_->SetTolerances(params_.time_tol,
            PSolver_t::PLS_DEFAULT,
            PSolver_t::PLS_DEFAULT,
            params_.time_iter_max));

    CHK(parallel_solver_->Configure());

    // setting up the preconditioner
    if (params_.time_precond != NoPrecond){
        ASSERT(precond_configured_, "The preconditioner isn't configured yet");
        CHK(parallel_solver_->SetPrecondContext(static_cast<const void*>(this)));
        CHK(parallel_solver_->UpdatePrecond(ImplicitPrecond));
    }
    psolver_configured_ = true;

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ConfigurePrecond(const PrecondScheme &precond) const{

    PROFILESTART();
    if (precond!=DiagonalSpectral)
        return ErrorEvent::NotImplementedError; /* Unsupported preconditioner scheme */

    INFO("Setting up the diagonal preceonditioner");
    value_type *buffer = new value_type[position_precond.size() * sizeof(value_type)];

    { //bending precond
        int idx(0), N(0);
        // The sh coefficients are ordered by m and then n
        //COUT("second dim: "<<position_precond.getGridDim().second<<", first dim: "<<position_precond.getGridDim().first);
        for(int iM=0; iM<position_precond.getGridDim().second; ++iM){
            for(int iN=++N/2; iN<position_precond.getGridDim().first; ++iN){
                value_type bending_precond(1.0/fabs(1.0-dt_*iN*iN*iN));
                bending_precond = fabs(bending_precond) < 1e3   ? bending_precond : 1.0;
                buffer[idx]     = fabs(bending_precond) > 1e-10 ? bending_precond : 1.0;
                ++idx;
            }
        }
        //COUT("idx: "<<idx);
        position_precond.getDevice().Memcpy(position_precond.begin(), buffer,
            position_precond.size() * sizeof(value_type),
            device_type::MemcpyHostToDevice);
    }
    { // tension precond
        int idx(0), N(0);
        for(int iM=0; iM<tension_precond.getGridDim().second; ++iM){
            for(int iN=++N/2; iN<tension_precond.getGridDim().first; ++iN){
                value_type eig(4*iN*iN-1);
                eig *= 2*iN+3;
                eig /= iN+1;
                eig /= 2*iN*iN+2*iN-1;
                eig  = iN==0 ? 1.0 : eig/iN;
                buffer[idx] = fabs(eig) > 1e-10 ? eig : 1.0;
                ++idx;
            }
        }
        tension_precond.getDevice().Memcpy(tension_precond.begin(), buffer,
            tension_precond.size() * sizeof(value_type),
            device_type::MemcpyHostToDevice);
    }

    delete[] buffer;
    precond_configured_=true;
    PROFILEEND("",0);

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
AssembleRhsVel(PVec_t *rhs, const value_type &dt, const SolverScheme &scheme) const
{
    PROFILESTART();
    ASSERT(scheme==GloballyImplicit, "Unsupported scheme");
    INFO("Assembling RHS to solve for velocity");

    // rhs=[u_inf+Bx;div(u_inf+Bx)]
    COUTDEBUG("Evaluate background flow");
    std::auto_ptr<Vec_t> vRhs = checkoutVec();
    vRhs->replicate(S_.getPosition());
    CHK(BgFlow(*vRhs, dt));

    COUTDEBUG("Computing the far-field interaction due to explicit traction jump");
    std::auto_ptr<Vec_t> f  = checkoutVec();
    std::auto_ptr<Vec_t> Sf = checkoutVec();
    Intfcl_force_.explicitTractionJump(S_, *f);
    stokes_.SetDensitySL(f.get(),true);
    stokes_.SetDensityDL(NULL);
    stokes_(*Sf);
    axpy(static_cast<value_type>(1.0), *Sf, *vRhs, *vRhs);
    stokes_.GetRepulsionForce(S_.fc_);

    COUTDEBUG("Computing rhs for div(u)");
    std::auto_ptr<Sca_t> tRhs = checkoutSca();
    S_.div(*vRhs, *tRhs);

    ASSERT( vRhs->getDevice().isNumeric(vRhs->begin(), vRhs->size()), "Non-numeric rhs");
    ASSERT( tRhs->getDevice().isNumeric(tRhs->begin(), tRhs->size()), "Non-numeric rhs");

    // copy data
    size_t xsz(stokesBlockSize()), tsz(tensionBlockSize());
    typename PVec_t::iterator i(NULL);
    typename PVec_t::size_type rsz;
    CHK(parallel_rhs_->GetArray(i, rsz));
    ASSERT(rsz==xsz+tsz,"Bad sizes");

    if (params_.pseudospectral){
        COUTDEBUG("Copy data to parallel rhs array");
        vRhs->getDevice().Memcpy(i    , vRhs->begin(), xsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        tRhs->getDevice().Memcpy(i+xsz, tRhs->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    } else {  /* Galerkin */
        COUTDEBUG("Project RHS to spectral coefficient");
        std::auto_ptr<Vec_t> vRhsSh  = checkoutVec();
        std::auto_ptr<Sca_t> tRhsSh  = checkoutSca();
        std::auto_ptr<Vec_t> wrk     = checkoutVec();

        vRhsSh->replicate(*vRhs);
        tRhsSh->replicate(*tRhs);
        wrk->replicate(*vRhs);

        sht_.forward(*vRhs, *wrk, *vRhsSh);
        sht_.forward(*tRhs, *wrk, *tRhsSh);

        COUTDEBUG("Copy data to parallel RHS array (size="<<xsz<<"+"<<tsz<<")");
        vRhs->getDevice().Memcpy(i    , vRhsSh->begin(), xsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        tRhs->getDevice().Memcpy(i+xsz, tRhsSh->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

        recycle(vRhsSh);
        recycle(tRhsSh);
        recycle(wrk);
    }

    CHK(parallel_rhs_->RestoreArray(i));

    recycle(vRhs);
    recycle(f);
    recycle(Sf);
    recycle(tRhs);

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
AssembleRhsPos(PVec_t *rhs, const value_type &dt, const SolverScheme &scheme) const
{
    PROFILESTART();
    ASSERT(scheme==GloballyImplicit, "Unsupported scheme");
    INFO("Assembling RHS to solve for position");

    COUTDEBUG("Evaluate background flow");
    std::auto_ptr<Vec_t> pRhs = checkoutVec();
    std::auto_ptr<Vec_t> pRhs2 = checkoutVec();
    pRhs->replicate(S_.getPosition());
    pRhs2->replicate(S_.getPosition());
    CHK(BgFlow(*pRhs, dt));

    if( ves_props_.has_contrast ){
        COUTDEBUG("Computing the rhs due to viscosity contrast");
        std::auto_ptr<Vec_t> x  = checkoutVec();
        std::auto_ptr<Vec_t> Dx = checkoutVec();
        av(ves_props_.dl_coeff, S_.getPosition(), *x);
        stokes_.SetDensitySL(NULL, true);
        stokes_.SetDensityDL(x.get());
        stokes_(*Dx);
        axpy(-dt, *pRhs, *Dx, *pRhs);
        axpy(static_cast<value_type>(-1.0), *pRhs, *pRhs);

        recycle(x);
        recycle(Dx);
    } else
        axpy(dt, *pRhs, *pRhs);

    COUTDEBUG("Computing rhs for div(u)");
    std::auto_ptr<Sca_t> tRhs = checkoutSca();
    S_.div(*pRhs, *tRhs);

    av(ves_props_.vel_coeff, S_.getPosition(), *pRhs2);
    axpy(static_cast<value_type>(1.0), *pRhs, *pRhs2, *pRhs);

    ASSERT( pRhs->getDevice().isNumeric(pRhs->begin(), pRhs->size()), "Non-numeric rhs");
    ASSERT( tRhs->getDevice().isNumeric(tRhs->begin(), tRhs->size()), "Non-numeric rhs");

    // copy data
    size_t xsz(stokesBlockSize()), tsz(tensionBlockSize());
    typename PVec_t::iterator i(NULL);
    typename PVec_t::size_type rsz;
    CHK(parallel_rhs_->GetArray(i, rsz));
    ASSERT(rsz==xsz+tsz,"Bad sizes");

    if (params_.pseudospectral){
        COUTDEBUG("Copy data to parallel rhs array");
        pRhs->getDevice().Memcpy(i    , pRhs->begin(), xsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        tRhs->getDevice().Memcpy(i+xsz, tRhs->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    } else {  /* Galerkin */
        COUTDEBUG("Project RHS to spectral coefficient");
        std::auto_ptr<Vec_t> pRhsSh  = checkoutVec();
        std::auto_ptr<Sca_t> tRhsSh  = checkoutSca();
        std::auto_ptr<Vec_t> wrk     = checkoutVec();

        pRhsSh->replicate(*pRhs);
        tRhsSh->replicate(*tRhs);
        wrk->replicate(*pRhs);

        sht_.forward(*pRhs, *wrk, *pRhsSh);
        sht_.forward(*tRhs, *wrk, *tRhsSh);

        COUTDEBUG("Copy data to parallel rhs array");
        pRhs->getDevice().Memcpy(i    , pRhsSh->begin(), xsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        tRhs->getDevice().Memcpy(i+xsz, tRhsSh->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

        recycle(pRhsSh);
        recycle(tRhsSh);
        recycle(wrk);
    }

    CHK(parallel_rhs_->RestoreArray(i));

    recycle(pRhs);
    recycle(pRhs2);
    recycle(tRhs);

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
AssembleInitial(PVec_t *u0, const value_type &dt, const SolverScheme &scheme) const
{
    PROFILESTART();
    COUTDEBUG("Using current position/tension as initial guess");
    size_t vsz(stokesBlockSize()), tsz(tensionBlockSize());
    typename PVec_t::iterator i(NULL);
    typename PVec_t::size_type rsz;

    ASSERT( pos_vel_.getDevice().isNumeric(pos_vel_.begin(), pos_vel_.size()), "Non-numeric velocity");
    ASSERT( tension_.getDevice().isNumeric(tension_.begin(), tension_.size()), "Non-numeric tension");

    CHK(parallel_u_->GetArray(i, rsz));
    ASSERT(rsz==vsz+tsz,"Bad sizes");

    if (params_.pseudospectral){
        COUTDEBUG("Copy initial guess to parallel solution array");
        pos_vel_.getDevice().Memcpy(i    , pos_vel_.begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        tension_.getDevice().Memcpy(i+vsz, tension_.begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    } else {  /* Galerkin */
            COUTDEBUG("Project initial guess to spectral coefficient");
        std::auto_ptr<Vec_t> voxSh  = checkoutVec();
        std::auto_ptr<Sca_t> tSh    = checkoutSca();
        std::auto_ptr<Vec_t> wrk    = checkoutVec();

        voxSh->replicate(pos_vel_);
        tSh->replicate(tension_);
        wrk->replicate(pos_vel_);

        sht_.forward(pos_vel_, *wrk, *voxSh);
        sht_.forward(tension_, *wrk, *tSh);

        COUTDEBUG("Copy initial guess to parallel solution array");
        voxSh->getDevice().Memcpy(i    , voxSh->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        tSh->getDevice().Memcpy(  i+vsz, tSh->begin()  , tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

        recycle(voxSh);
        recycle(tSh);
        recycle(wrk);
    }

    CHK(parallel_u_->RestoreArray(i));
    CHK(parallel_solver_->InitialGuessNonzero(true));
    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::ImplicitMatvecPhysical(Vec_t &vox, Sca_t &ten) const
{
    PROFILESTART();

    std::auto_ptr<Vec_t> f   = checkoutVec();
    std::auto_ptr<Vec_t> Sf  = checkoutVec();
    std::auto_ptr<Vec_t> Du  = checkoutVec();
    f->replicate(vox);
    Sf->replicate(vox);
    Du->replicate(vox);

    COUTDEBUG("Computing the interfacial forces and setting single-layer density");
    if (params_.solve_for_velocity){
        // Bending of dt*u + tension of sigma
        axpy(dt_, vox, *Du);
        Intfcl_force_.implicitTractionJump(S_, *Du, ten, *f);
    } else {
        // dt*(Bending of x + tension of sigma)
        Intfcl_force_.implicitTractionJump(S_, vox, ten, *f);
        axpy(dt_, *f, *f);
    }
    stokes_.SetDensitySL(f.get());

    if( ves_props_.has_contrast ){
        COUTDEBUG("Setting the double-layer density");
        av(ves_props_.dl_coeff, vox, *Du);
        stokes_.SetDensityDL(Du.get());
    } else {
        stokes_.SetDensityDL(NULL);
    }

    COUTDEBUG("Calling stokes");
    stokes_(*Sf);

    COUTDEBUG("Computing the div term");
    //! @note For some reason, doing the linear algebraic manipulation
    //! and writing the constraint as -\div{S[f_b+f_s]} = \div{u_inf
    //! almost halves the number of gmres iterations. Also having the
    //! minus sign in the matvec is tangibly better (1-2
    //! iterations). Need to investigate why.
    S_.div(*Sf, ten);
    axpy((value_type) -1.0, ten, ten);

    if( ves_props_.has_contrast )
        av(ves_props_.vel_coeff, vox, vox);

    axpy((value_type) -1.0, *Sf, vox, vox);

    ASSERT(vox.getDevice().isNumeric(vox.begin(), vox.size()), "Non-numeric velocity");
    ASSERT(ten.getDevice().isNumeric(ten.begin(), ten.size()), "Non-numeric divergence");

    recycle(f);
    recycle(Sf);
    recycle(Du);

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ImplicitApply(const POp_t *o, const value_type *x, value_type *y)
{
    PROFILESTART();
    const InterfacialVelocity *F(NULL);
    o->Context((const void**) &F);
    size_t vsz(F->stokesBlockSize()), tsz(F->tensionBlockSize());

    std::auto_ptr<Vec_t> vox = F->checkoutVec();
    std::auto_ptr<Sca_t> ten = F->checkoutSca();
    vox->replicate(F->pos_vel_);
    ten->replicate(F->tension_);

    COUTDEBUG("Unpacking the input from parallel vector");
    if (F->params_.pseudospectral){
        vox->getDevice().Memcpy(vox->begin(), x    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
        ten->getDevice().Memcpy(ten->begin(), x+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    } else {  /* Galerkin */
        std::auto_ptr<Vec_t> voxSh = F->checkoutVec();
        std::auto_ptr<Sca_t> tSh   = F->checkoutSca();
        std::auto_ptr<Vec_t> wrk   = F->checkoutVec();

        voxSh->replicate(*vox);
        tSh->replicate(*ten);
        wrk->replicate(*vox);
        voxSh->getDevice().Memcpy(voxSh->begin(), x    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
        tSh  ->getDevice().Memcpy(tSh->begin()  , x+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);

        COUTDEBUG("Mapping the input to physical space");
        F->sht_.backward(*voxSh, *wrk, *vox);
        F->sht_.backward(*tSh  , *wrk, *ten);

        F->recycle(voxSh);
        F->recycle(tSh);
        F->recycle(wrk);
    }

    F->ImplicitMatvecPhysical(*vox, *ten);

    if (F->params_.pseudospectral){
        COUTDEBUG("Packing the matvec into parallel vector");
        vox->getDevice().Memcpy(y    , vox->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        ten->getDevice().Memcpy(y+vsz, ten->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    } else {  /* Galerkin */
        COUTDEBUG("Mapping the matvec to physical space");
        std::auto_ptr<Vec_t> voxSh = F->checkoutVec();
        std::auto_ptr<Sca_t> tSh   = F->checkoutSca();
        std::auto_ptr<Vec_t> wrk   = F->checkoutVec();

        voxSh->replicate(*vox);
        tSh->replicate(*ten);
        wrk->replicate(*vox);

        F->sht_.forward(*vox, *wrk, *voxSh);
        F->sht_.forward(*ten, *wrk, *tSh);

        COUTDEBUG("Packing the matvec into parallel vector");
        voxSh->getDevice().Memcpy(y    , voxSh->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        tSh  ->getDevice().Memcpy(y+vsz, tSh->begin()  , tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

        F->recycle(voxSh);
        F->recycle(tSh);
        F->recycle(wrk);
    }

    F->recycle(vox);
    F->recycle(ten);

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ImplicitPrecond(const PSolver_t *ksp, const value_type *x, value_type *y)
{
    PROFILESTART();
    const InterfacialVelocity *F(NULL);
    ksp->PrecondContext((const void**) &F);

    size_t vsz(F->stokesBlockSize()), tsz(F->tensionBlockSize());

    std::auto_ptr<Vec_t> vox = F->checkoutVec();
    std::auto_ptr<Vec_t> vxs = F->checkoutVec();
    std::auto_ptr<Vec_t> wrk = F->checkoutVec();
    vox->replicate(F->pos_vel_);
    vxs->replicate(F->pos_vel_);
    wrk->replicate(F->pos_vel_);

    std::auto_ptr<Sca_t> ten = F->checkoutSca();
    std::auto_ptr<Sca_t> tns = F->checkoutSca();
    ten->replicate(F->tension_);
    tns->replicate(F->tension_);

    COUTDEBUG("Unpacking the input parallel vector");
    if (F->params_.pseudospectral){
        vox->getDevice().Memcpy(vox->begin(), x    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
        ten->getDevice().Memcpy(ten->begin(), x+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
        F->sht_.forward(*vox, *wrk, *vxs);
        F->sht_.forward(*ten, *wrk, *tns);
    } else {  /* Galerkin */
        vxs->getDevice().Memcpy(vxs->begin(), x    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
        tns->getDevice().Memcpy(tns->begin(), x+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    }

    COUTDEBUG("Applying diagonal preconditioner");
    F->sht_.ScaleFreq(vxs->begin(), vxs->getNumSubFuncs(), F->position_precond.begin(), vxs->begin());
    F->sht_.ScaleFreq(tns->begin(), tns->getNumSubFuncs(), F->tension_precond.begin() , tns->begin());

    if (F->params_.pseudospectral){
        F->sht_.backward(*vxs, *wrk, *vox);
        F->sht_.backward(*tns, *wrk, *ten);
        vox->getDevice().Memcpy(y    , vox->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        ten->getDevice().Memcpy(y+vsz, ten->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    } else {  /* Galerkin */
        vxs->getDevice().Memcpy(y    , vxs->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        tns->getDevice().Memcpy(y+vsz, tns->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    }

    F->recycle(vox);
    F->recycle(vxs);
    F->recycle(wrk);
    F->recycle(ten);
    F->recycle(tns);

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
JacobiImplicitApply(const GMRESLinSolver<value_type> *o, const value_type *x, value_type *y, int tmp_trash)
{
    //PROFILESTART();
    const InterfacialVelocity *F(NULL);
    o->Context((const void**) &F);
    size_t vsz(F->stokesBlockSize()), tsz(F->tensionBlockSize());

    std::auto_ptr<Vec_t> vox = F->checkoutVec();
    std::auto_ptr<Sca_t> ten = F->checkoutSca();
    vox->replicate(F->pos_vel_);
    ten->replicate(F->tension_);
    
    std::auto_ptr<Vec_t> vox_y = F->checkoutVec();
    std::auto_ptr<Sca_t> ten_y = F->checkoutSca();
    vox_y->replicate(F->pos_vel_);
    ten_y->replicate(F->tension_);

    vox->getDevice().Memcpy(vox->begin(), x    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    ten->getDevice().Memcpy(ten->begin(), x+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);

    F->operator()(*vox, *ten, *vox_y, *ten_y);

    vox_y->getDevice().Memcpy(y    , vox_y->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    ten_y->getDevice().Memcpy(y+vsz, ten_y->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

    F->recycle(vox);
    F->recycle(ten);
    
    F->recycle(vox_y);
    F->recycle(ten_y);

    //PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
JacobiImplicitApplyPerVesicle(const GMRESLinSolver<value_type> *o, const value_type *x, value_type *y, int current_vesicle)
{
    //PROFILESTART();
    const InterfacialVelocity *F(NULL);
    o->Context((const void**) &F);
    size_t vsz(F->stokesBlockSize()/F->nv_), tsz(F->tensionBlockSize()/F->nv_);

    Vec_t vox(1, F->params_.sh_order);
    Sca_t ten(1, F->params_.sh_order);
    
    Vec_t vox_y(1, F->params_.sh_order);
    Vec_t ten_y(1, F->params_.sh_order);

    vox.getDevice().Memcpy(vox.begin(), x    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    ten.getDevice().Memcpy(ten.begin(), x+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);

    F->operator()(vox, ten, vox_y, ten_y, current_vesicle);

    vox_y.getDevice().Memcpy(y    , vox_y.begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    ten_y.getDevice().Memcpy(y+vsz, ten_y.begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

    //PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
JacobiImplicitPrecond(const GMRESLinSolver<value_type> *o, const value_type *x, value_type *y)
{
    PROFILESTART();
    const InterfacialVelocity *F(NULL);
    o->Context((const void**) &F);
    
    size_t vsz(F->stokesBlockSize()), tsz(F->tensionBlockSize());
    
    //Arr_t::getDevice().Memcpy(y, x, (vsz+tsz)*sizeof(value_type), device_type::MemcpyHostToHost);

    std::auto_ptr<Vec_t> vox = F->checkoutVec();
    std::auto_ptr<Vec_t> vxs = F->checkoutVec();
    std::auto_ptr<Vec_t> wrk = F->checkoutVec();
    vox->replicate(F->pos_vel_);
    vxs->replicate(F->pos_vel_);
    wrk->replicate(F->pos_vel_);

    std::auto_ptr<Sca_t> ten = F->checkoutSca();
    std::auto_ptr<Sca_t> tns = F->checkoutSca();
    ten->replicate(F->tension_);
    tns->replicate(F->tension_);

    COUTDEBUG("Unpacking the input parallel vector");
    vox->getDevice().Memcpy(vox->begin(), x    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    ten->getDevice().Memcpy(ten->begin(), x+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    
    F->sht_.forward(*vox, *wrk, *vxs);
    F->sht_.forward(*ten, *wrk, *tns);

    //COUTDEBUG("Applying diagonal preconditioner");
    F->sht_.ScaleFreq(vxs->begin(), vxs->getNumSubFuncs(), F->position_precond.begin(), vxs->begin());
    F->sht_.ScaleFreq(tns->begin(), tns->getNumSubFuncs(), F->tension_precond.begin() , tns->begin());

    F->sht_.backward(*vxs, *wrk, *vox);
    F->sht_.backward(*tns, *wrk, *ten);
    
    vox->getDevice().Memcpy(y    , vox->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    ten->getDevice().Memcpy(y+vsz, ten->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

    F->recycle(vox);
    F->recycle(vxs);
    F->recycle(wrk);
    F->recycle(ten);
    F->recycle(tns);

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
JacobiImplicitPrecondPerVesicle(const GMRESLinSolver<value_type> *o, const value_type *x, value_type *y)
{
    // TODO for each vesicle
    //PROFILESTART();
    const InterfacialVelocity *F(NULL);
    o->Context((const void**) &F);
    
    size_t vsz(F->stokesBlockSize()/F->nv_), tsz(F->tensionBlockSize()/F->nv_);
    
    Arr_t::getDevice().Memcpy(y, x, (vsz+tsz) * sizeof(value_type), device_type::MemcpyHostToHost);

    /*
    std::auto_ptr<Vec_t> vox = F->checkoutVec();
    std::auto_ptr<Vec_t> vxs = F->checkoutVec();
    std::auto_ptr<Vec_t> wrk = F->checkoutVec();
    vox->replicate(F->pos_vel_);
    vxs->replicate(F->pos_vel_);
    wrk->replicate(F->pos_vel_);

    std::auto_ptr<Sca_t> ten = F->checkoutSca();
    std::auto_ptr<Sca_t> tns = F->checkoutSca();
    ten->replicate(F->tension_);
    tns->replicate(F->tension_);

    COUTDEBUG("Unpacking the input parallel vector");
    vox->getDevice().Memcpy(vox->begin(), x    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    ten->getDevice().Memcpy(ten->begin(), x+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    
    F->sht_.forward(*vox, *wrk, *vxs);
    F->sht_.forward(*ten, *wrk, *tns);

    //COUTDEBUG("Applying diagonal preconditioner");
    F->sht_.ScaleFreq(vxs->begin(), vxs->getNumSubFuncs(), F->position_precond.begin(), vxs->begin());
    F->sht_.ScaleFreq(tns->begin(), tns->getNumSubFuncs(), F->tension_precond.begin() , tns->begin());

    F->sht_.backward(*vxs, *wrk, *vox);
    F->sht_.backward(*tns, *wrk, *ten);
    
    vox->getDevice().Memcpy(y    , vox->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    ten->getDevice().Memcpy(y+vsz, ten->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

    F->recycle(vox);
    F->recycle(vxs);
    F->recycle(wrk);
    F->recycle(ten);
    F->recycle(tns);
    */

    //PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
JacobiImplicitLCPApply(const GMRESLinSolver<value_type> *o, const value_type *x, value_type *y)
{
    PROFILESTART();
    const InterfacialVelocity *F(NULL);
    o->Context((const void**) &F);
    size_t vsz(F->stokesBlockSize()), tsz(F->tensionBlockSize()), lsz(F->num_cvs_);

    std::auto_ptr<Vec_t> vox = F->checkoutVec();
    std::auto_ptr<Sca_t> ten = F->checkoutSca();
    vox->replicate(F->pos_vel_);
    ten->replicate(F->tension_);
    Arr_t lam(lsz);
    
    std::auto_ptr<Vec_t> vox_y = F->checkoutVec();
    std::auto_ptr<Sca_t> ten_y = F->checkoutSca();
    vox_y->replicate(F->pos_vel_);
    ten_y->replicate(F->tension_);
    Arr_t lam_y(lsz);

    vox->getDevice().Memcpy(vox->begin(), x    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    ten->getDevice().Memcpy(ten->begin(), x+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    lam.getDevice().Memcpy(lam.begin(), x+vsz+tsz, lsz * sizeof(value_type), device_type::MemcpyHostToDevice);

    F->operator()(*vox, *ten, lam, *vox_y, *ten_y, lam_y);

    vox_y->getDevice().Memcpy(y    , vox_y->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    ten_y->getDevice().Memcpy(y+vsz, ten_y->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    lam_y.getDevice().Memcpy(y+vsz+tsz, lam_y.begin(), lsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

    F->recycle(vox);
    F->recycle(ten);
    
    F->recycle(vox_y);
    F->recycle(ten_y);

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
JacobiImplicitLCPPrecond(const GMRESLinSolver<value_type> *o, const value_type *x, value_type *y)
{
    PROFILESTART();
    const InterfacialVelocity *F(NULL);
    o->Context((const void**) &F);
    
    size_t vsz(F->stokesBlockSize()), tsz(F->tensionBlockSize()), lsz(F->num_cvs_);

    Arr_t::getDevice().Memcpy(y, x, (vsz+tsz+lsz) * sizeof(value_type), device_type::MemcpyHostToHost);
    
    /* 
    std::auto_ptr<Vec_t> vox = F->checkoutVec();
    std::auto_ptr<Vec_t> vxs = F->checkoutVec();
    std::auto_ptr<Vec_t> wrk = F->checkoutVec();
    vox->replicate(F->pos_vel_);
    vxs->replicate(F->pos_vel_);
    wrk->replicate(F->pos_vel_);
    Arr_t lam(lsz);

    std::auto_ptr<Sca_t> ten = F->checkoutSca();
    std::auto_ptr<Sca_t> tns = F->checkoutSca();
    ten->replicate(F->tension_);
    tns->replicate(F->tension_);

    COUTDEBUG("Unpacking the input parallel vector");
    vox->getDevice().Memcpy(vox->begin(), x    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    ten->getDevice().Memcpy(ten->begin(), x+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    lam.getDevice().Memcpy(lam.begin(), x+vsz+tsz, lsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    
    axpy(static_cast<value_type>(1.0),*vox,*vox);
    axpy(static_cast<value_type>(1.0),*ten,*ten);
    
    F->sht_.forward(*vox, *wrk, *vxs);
    F->sht_.forward(*ten, *wrk, *tns);

    COUTDEBUG("Applying diagonal preconditioner");
    F->sht_.ScaleFreq(vxs->begin(), vxs->getNumSubFuncs(), F->position_precond.begin(), vxs->begin());
    F->sht_.ScaleFreq(tns->begin(), tns->getNumSubFuncs(), F->tension_precond.begin() , tns->begin());

    F->sht_.backward(*vxs, *wrk, *vox);
    F->sht_.backward(*tns, *wrk, *ten);
    
    vox->getDevice().Memcpy(y    , vox->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    ten->getDevice().Memcpy(y+vsz, ten->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    lam.getDevice().Memcpy(y+vsz+tsz, lam.begin(), lsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

    F->recycle(vox);
    F->recycle(vxs);
    F->recycle(wrk);

    F->recycle(ten);
    F->recycle(tns);
    */

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
LCPApply(const GMRESLinSolver<value_type> *o, const value_type *x, value_type *y)
{
    PROFILESTART();
    const InterfacialVelocity *F(NULL);
    o->Context((const void**) &F);
    size_t lsz(F->num_cvs_);

    Arr_t lam(lsz);
    Arr_t lam_y(lsz);

    lam.getDevice().Memcpy(lam.begin(), x, lsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    F->operator()(lam, lam_y);
    lam_y.getDevice().Memcpy(y, lam_y.begin(), lsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
LCPPrecond(const GMRESLinSolver<value_type> *o, const value_type *x, value_type *y)
{
    PROFILESTART();
    const InterfacialVelocity *F(NULL);
    o->Context((const void**) &F);
    
    size_t lsz(F->num_cvs_);

    Arr_t::getDevice().Memcpy(y, x, lsz * sizeof(value_type), device_type::MemcpyHostToHost);

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ParallelLCPApply(const POp_t *o, const value_type *x, value_type *y)
{
    PROFILESTART();
    const InterfacialVelocity *F(NULL);
    o->Context((const void**) &F);
    size_t sz(F->num_cvs_);
    Arr_t lambda(sz);
        
    lambda.getDevice().Memcpy(lambda.begin(), x, sz*sizeof(value_type), device_type::MemcpyHostToDevice);
    F->ParallelLCPMatvec(lambda);
    lambda.getDevice().Memcpy(y, lambda.begin(), sz*sizeof(value_type), device_type::MemcpyDeviceToHost);

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ParallelLCPPrecond(const PSolver_t *ksp, const value_type *x, value_type *y)
{
    PROFILESTART();
    const InterfacialVelocity *F(NULL);
    ksp->PrecondContext((const void**) &F);

    size_t sz(F->num_cvs_);
    if(sz > 0)
    {
        COUTDEBUG("Applying diagonal preconditioner");
        Arr_t::getDevice().Memcpy(y, x, sz*sizeof(value_type), device_type::MemcpyHostToHost);
    }

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
Solve(const PVec_t *rhs, PVec_t *u0, const value_type &dt, const SolverScheme &scheme) const
{
    PROFILESTART();
    INFO("Solving for position/velocity and tension using "<<scheme<<" scheme.");

    Error_t err = parallel_solver_->Solve(parallel_rhs_, parallel_u_);
    typename PVec_t::size_type iter;
    CHK(parallel_solver_->IterationNumber(iter));

    INFO("Parallel solver returned after "<<iter<<" iteration(s).");
    parallel_solver_->ViewReport();

    PROFILEEND("",0);
    return err;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::Update(PVec_t *u0)
{
    PROFILESTART();
    COUTDEBUG("Updating position and tension.");
    size_t vsz(stokesBlockSize()), tsz(tensionBlockSize());

    typename PVec_t::iterator i(NULL);
    typename PVec_t::size_type rsz;

    CHK(u0->GetArray(i, rsz));
    ASSERT(rsz==vsz+tsz,"Bad sizes");

    if (params_.pseudospectral){
        COUTDEBUG("Copy data from parallel solution array");
        pos_vel_.getDevice().Memcpy(pos_vel_.begin(), i    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
        tension_.getDevice().Memcpy(tension_.begin(), i+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
    } else { /* Galerkin */
        COUTDEBUG("Unpacking the solution from parallel vector");
        std::auto_ptr<Vec_t> voxSh = checkoutVec();
        std::auto_ptr<Sca_t> tSh   = checkoutSca();
        std::auto_ptr<Vec_t> wrk   = checkoutVec();

        voxSh->replicate(pos_vel_);
        tSh->replicate(tension_);
        wrk->replicate(pos_vel_);

        voxSh->getDevice().Memcpy(voxSh->begin(), i    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
        tSh  ->getDevice().Memcpy(tSh  ->begin(), i+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);

        COUTDEBUG("Mapping the solution to physical space");
        sht_.backward(*voxSh, *wrk, pos_vel_);
        sht_.backward(*tSh  , *wrk, tension_);

        recycle(voxSh);
        recycle(tSh);
        recycle(wrk);
    }

    CHK(u0->RestoreArray(i));

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
BgFlow(Vec_t &bg, const value_type &dt) const{
    //!@bug the time should be passed to the BgFlow handle.
    bg_flow_(S_.getPosition(), 0, bg);

    return ErrorEvent::Success;
}

// Compute velocity_far = velocity_bg + FMM(bending+tension) - DirectStokes(bending+tension)
template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
updateFarField() const
{
    PROFILESTART();
    ASSERT(pos_vel_.size() == S_.getPosition().size(), "inccorrect size");
    
    std::auto_ptr<Vec_t>        fi  = checkoutVec();
    std::auto_ptr<Vec_t>        ui  = checkoutVec();
    std::auto_ptr<Vec_t>        vel = checkoutVec();
    std::auto_ptr<Vec_t>        pos = checkoutVec();
    std::auto_ptr<Vec_t>        vel_bd = checkoutVec();
    fi->replicate(pos_vel_);
    ui->replicate(pos_vel_);
    vel->replicate(pos_vel_);
    pos->replicate(pos_vel_);
    vel_bd->replicate(pos_vel_);
    fe_.replicate(pos_vel_);

    // add bending force
    Intfcl_force_.bendingForce(S_, *fi);
    // add tensile force
    Intfcl_force_.tensileForce(S_, tension_, *vel);
    axpy(static_cast<value_type>(1.0), *fi, *vel, *fi);
    // add contact force
    S_.sht_.lowPassFilter(S_.fc_, *ui, *pos, *vel);
    //axpy(static_cast<value_type>(1.0), *fi, *vel, *fi);
    //axpy(static_cast<value_type>(1.0), *fi, S_.fc_, *fi);
    // add gravity force
    Intfcl_force_.gravityForce(S_, S_.getPosition(), *vel);
    axpy(static_cast<value_type>(1.0), *fi, *vel, *fi);
    // electro force
    //el_force->updateJacobiImplicit(fe_);
    //axpy(static_cast<value_type>(-1.0), fe_, *fi, *fi);
        
    // set single layer density
    stokes_.SetDensitySL(fi.get());

    // set doube layer density
    if(ves_props_.has_contrast){
        //COUT("has contrast");
        av(ves_props_.dl_coeff, pos_vel_, *ui);
        stokes_.SetDensityDL(ui.get());
    }
    else{
        stokes_.SetDensityDL(NULL);
    }
        
    // far field due to sl, dl
    stokes_.FarInteraction(*vel); // all - self
    //stokes_.GetRepulsionForce(S_.fc_);
    //stokes_(*vel);
    //stokes_.SelfInteraction(*fi);
    //axpy(static_cast<value_type>(-1.0), *fi, *vel, *vel);
               
    CHK(this->BgFlow(pos_vel_, this->dt_));
    COUT("max bg vel: "<<MaxAbs(pos_vel_));
    axpy(static_cast<value_type>(1.0), *vel, pos_vel_, pos_vel_);
    COUT("max bg vel: "<<MaxAbs(pos_vel_));

    /*
    // boundary
    ShufflePoints(S_.getPosition(), *pos);
    vel_bd->setPointOrder(PointMajor);

    MPI_Comm comm=MPI_COMM_WORLD;
    pvfmm::Profile::Tic("BD evalpot", &comm, true);
    fixed_bd->EvalPotential(pos->size()/COORD_DIM, pos->begin(), vel_bd->begin());
    pvfmm::Profile::Toc();

    ShufflePoints(*vel_bd, *vel);
    axpy(static_cast<value_type>(1.0), *vel, pos_vel_, pos_vel_);
    pos->setPointOrder(AxisMajor);      
    vel_bd->setPointOrder(AxisMajor);
    //for(int i=0; i<pos_vel_.size();i++)
        //std::cout<<"vel: "<<pos_vel_.begin()[i]<<"\n";
    
    //COUT("max bg vel: "<<MaxAbs(pos_vel_));
    */
    
    recycle(fi);
    recycle(ui);
    recycle(vel);
    recycle(pos);
    recycle(vel_bd);
    
    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
updateFarFieldBoundary() const
{
    PROFILESTART();
    ASSERT(pos_vel_.size() == S_.getPosition().size(), "inccorrect size");
    
    std::auto_ptr<Vec_t>        fi  = checkoutVec();
    std::auto_ptr<Vec_t>        ui  = checkoutVec();
    std::auto_ptr<Vec_t>        vel = checkoutVec();
    std::auto_ptr<Vec_t>        pos = checkoutVec();
    fi->replicate(pos_vel_);
    ui->replicate(pos_vel_);
    vel->replicate(pos_vel_);
    pos->replicate(pos_vel_);

    /*
    // add bending force
    Intfcl_force_.bendingForce(S_, *fi);
    // add tensile force
    Intfcl_force_.tensileForce(S_, tension_, *vel);
    axpy(static_cast<value_type>(1.0), *fi, *vel, *fi);
    // add contact force
    S_.sht_.lowPassFilter(S_.fc_, *ui, *pos, *vel);
    axpy(static_cast<value_type>(1.0), *fi, *vel, *fi);
    //axpy(static_cast<value_type>(1.0), *fi, S_.fc_, *fi);
    // add gravity force
    Intfcl_force_.gravityForce(S_, S_.getPosition(), *vel);
    axpy(static_cast<value_type>(1.0), *fi, *vel, *fi);
    */

    /*
    // set source
    stokes_.SetSrcCoord(S_.getPosition());
    
    // set target
    double* target_address;
    int num_target_points;
    target_address = fixed_bd->GetSamplePoints(num_target_points);
    pvfmm::Vector<double> coord(COORD_DIM*num_target_points, target_address, false);
    stokes_.SetTrgCoord(&coord);

    // set single layer density
    stokes_.SetDensitySL(fi.get());

    // set doube layer density
    if(ves_props_.has_contrast){
        //COUT("has contrast");
        av(ves_props_.dl_coeff, pos_vel_, *ui);
        stokes_.SetDensityDL(ui.get());
    }
    else{
        stokes_.SetDensityDL(NULL);
    }
        
    // potential from vesicles
    pvfmm::Vector<double> value(COORD_DIM*num_target_points);
    INFO("begin vesicle to boundary interaction");
    value=stokes_();
    INFO("end vesicle to boundary interaction");
    for(long i=0;i<value.Dim();i++) value[i] = -1.0*value[i];
    fixed_bd->SetBoundaryData(&value[0]);

    // bgflow
    //
    //
    */

    //test
    // set target point
    int num_pts_1d = 100;
    double bbmin[3], bbmax[3], len[3];
    bbmin[0] = -92-1.5; bbmin[1] = 10.9-1.5; bbmin[2] = -73-1.5;
    bbmax[0] = -92+1.5; bbmax[1] = 10.9+1.5; bbmax[2] = -73+1.5;

    len[0] = (bbmax[0]-bbmin[0])/(num_pts_1d-1);
    len[1] = (bbmax[1]-bbmin[1])/(num_pts_1d-1);
    len[2] = (bbmax[2]-bbmin[2])/(num_pts_1d-1);

    pvfmm::Vector<double> coord(3*num_pts_1d*num_pts_1d*num_pts_1d);
    pvfmm::Vector<double> value(3*num_pts_1d*num_pts_1d*num_pts_1d);

    pvfmm::Vector<double> result(num_pts_1d*num_pts_1d*num_pts_1d);
    for(int i=0; i<result.Dim(); i++) result[i] = 0;

    int count=0;
    for(int i=0; i<num_pts_1d; i++)
        for(int j=0; j<num_pts_1d; j++)
            for(int k=0; k<num_pts_1d; k++)
            {
                coord[count*3+0] = bbmin[0] + i*len[0];
                coord[count*3+1] = bbmin[1] + j*len[1];
                coord[count*3+2] = bbmin[2] + k*len[2];
                count++;
            }

    stokes_.SetSrcCoord(S_.getPosition());
    stokes_.SetTrgCoord(&coord);
    stokes_.SetDensitySL(NULL);
    
    vel->getDevice().Memset(vel->begin(), 0, sizeof(value_type)*vel->size());
    for(int i=0; i<vel->size(); i++)
        vel->begin()[i] = 1.0;
    stokes_.SetDensityDL(vel.get());
    
    value=stokes_();
    double max_1 = -100;
    int max_index = -1;
    for(int i=0; i<value.Dim(); i++){
        //max_1 = std::max(max_1, std::abs(value[i]) );
        if(abs(value[i])>0.001){
            result[i/3] = 1;
        }
    }
    INFO(params_.load_checkpoint);

    std::cout<<std::fixed<<std::setprecision(2);
    //for(int i=0; i<result.Dim(); i++)
        //std::cout<<result[i]<<"\n";
    std::string filename = "data/"+params_.load_checkpoint;
    ofstream myfile;
    myfile.open(filename);
    for(int i=0; i<result.Dim(); i++)
        myfile<<result[i]<<"\n";
    myfile.close();
    /*
    int start_index = std::max(0, max_index-3);
    int end_index = std::min((int)(value.Dim()-1), max_index+3);
    string tmp_out;
    for(int i_index = start_index; i_index<=end_index; i_index++)
        tmp_out = tmp_out + to_string(coord[i_index]) + ", ";
    COUT("max coords: "<<tmp_out);
    */
    //end test
    
    //fixed_bd->RestoreSamplePoints(&target_address);

    recycle(fi);
    recycle(ui);
    recycle(vel);
    recycle(pos);
    
    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
CallInteraction(const Vec_t &src, const Vec_t &den, Vec_t &pot) const
{
    PROFILESTART();
    std::auto_ptr<Vec_t>        X = checkoutVec();
    std::auto_ptr<Vec_t>        D = checkoutVec();
    std::auto_ptr<Vec_t>        P = checkoutVec();

    X->replicate(src);
    D->replicate(den);
    P->replicate(pot);

    // shuffle
    ShufflePoints(src, *X);
    ShufflePoints(den, *D);
    P->setPointOrder(PointMajor);

    // far interactions
    Error_t status;
    CHK( status = interaction_(*X, *D, *P));

    // shuffle back
    ShufflePoints(*P, pot);

    X->setPointOrder(AxisMajor);        /* ignoring current content */
    D->setPointOrder(AxisMajor);        /* ignoring current content */
    P->setPointOrder(AxisMajor);        /* ignoring current content */

    recycle(X);
    recycle(D);
    recycle(P);
    PROFILEEND("",0);

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
EvaluateFarInteraction(const Vec_t &src, const Vec_t &fi, Vec_t &vel) const
{
    if ( params_.interaction_upsample ){
        return EvalFarInter_ImpUpsample(src, fi, vel);
    } else {
        return EvalFarInter_Imp(src, fi, vel);
    }
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
EvalFarInter_Imp(const Vec_t &src, const Vec_t &fi, Vec_t &vel) const
{
    std::auto_ptr<Vec_t> den    = checkoutVec();
    std::auto_ptr<Vec_t> slf    = checkoutVec();

    den->replicate(src);
    slf->replicate(vel);

    // multiply area elment into density
    xv(S_.getAreaElement(), fi, *den);

    // compute self (to subtract)
    slf->getDevice().DirectStokes(src.begin(), den->begin(), quad_weights_.begin(),
        slf->getStride(), slf->getStride(), slf->getNumSubs(), src.begin() /* target */,
        0, slf->getStride() /* number of trgs per surface */,
        slf->begin());

    // incorporating the quadrature weights into the density (use pos as temp holder)
    ax<Sca_t>(quad_weights_, *den, *den);

    CHK(CallInteraction(src, *den, vel));

    // subtract self
    axpy(static_cast<value_type>(-1.0), *slf, vel, vel);

    recycle(den);
    recycle(slf);

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
EvalFarInter_ImpUpsample(const Vec_t &src, const Vec_t &fi, Vec_t &vel) const
{
    std::auto_ptr<Vec_t> pos = checkoutVec();
    std::auto_ptr<Vec_t> den = checkoutVec();
    std::auto_ptr<Vec_t> pot = checkoutVec();
    std::auto_ptr<Vec_t> slf = checkoutVec();
    std::auto_ptr<Vec_t> shc = checkoutVec();
    std::auto_ptr<Vec_t> wrk = checkoutVec();

    // prepare for upsampling
    int usf(sht_upsample_.getShOrder());
    pos->resize(src.getNumSubs(), usf);
    den->resize(src.getNumSubs(), usf);
    slf->resize(src.getNumSubs(), usf);
    shc->resize(src.getNumSubs(), usf);
    wrk->resize(src.getNumSubs(), usf);

    // multiply area elment into density (using pot as temp)
    pot->replicate(fi);
    xv(S_.getAreaElement(), fi, *pot);

    // upsample position and density
    Resample( src, sht_, sht_upsample_, *shc, *wrk, *pos);
    Resample(*pot, sht_, sht_upsample_, *shc, *wrk, *den);
    pot->resize(src.getNumSubs(), usf);

    // compute self (to subtract)
    slf->getDevice().DirectStokes(pos->begin(), den->begin(), quad_weights_up_.begin(),
        slf->getStride(), slf->getStride(), slf->getNumSubs(), pos->begin() /* target */,
        0, slf->getStride() /* number of trgs per surface */,
        slf->begin());

    // incorporating the quadrature weights into the density
    ax<Sca_t>(quad_weights_up_, *den, *den);

    CHK(CallInteraction(*pos, *den, *pot));

    // subtract self
    axpy(static_cast<value_type>(-1.0), *slf, *pot, *slf);

    // downsample
    Resample(*slf, sht_upsample_, sht_, *shc, *wrk, *pot);
    sht_.lowPassFilter(*pot, *wrk, *shc, vel);

    recycle(pos);
    recycle(den);
    recycle(pot);
    recycle(slf);
    recycle(shc);
    recycle(wrk);

    return ErrorEvent::Success;
}

// Linear solve to compute tension such that surface divergence:
// surf_div( velocity + stokes(tension) ) = 0
template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::getTension(
    const Vec_t &vel_in, Sca_t &tension) const
{
    PROFILESTART();
    std::auto_ptr<Sca_t> rhs = checkoutSca();
    std::auto_ptr<Sca_t> wrk = checkoutSca();

    S_.div(vel_in, *rhs);

    //! this just negates rhs (not a bug; bad naming for overleaded axpy)
    axpy(static_cast<value_type>(-1), *rhs, *rhs);

    int iter(params_.time_iter_max);
    int rsrt(params_.time_iter_max);
    value_type tol(params_.time_tol),relres(params_.time_tol);
    enum BiCGSReturn solver_ret;
    Error_t ret_val(ErrorEvent::Success);

    COUTDEBUG("Solving for tension");
    solver_ret = linear_solver_(*this, tension, *rhs, rsrt, iter, relres);

    if ( solver_ret  != BiCGSSuccess )
        ret_val = ErrorEvent::DivergenceError;

    COUTDEBUG("Tension solve: Total iter = "<< iter<<", relres = "<<relres);
    COUTDEBUG("Checking true relres");
    ASSERT(((*this)(tension, *wrk),
            axpy(static_cast<value_type>(-1), *wrk, *rhs, *wrk),
            relres = sqrt(AlgebraicDot(*wrk, *wrk))/sqrt(AlgebraicDot(*rhs,*rhs)),
            relres<tol
            ),
           "relres ("<<relres<<")<tol("<<tol<<")"
           );

    recycle(wrk);
    recycle(rhs);
    PROFILEEND("",0);

    return ret_val;
}

// Computes near (self) velocity due to force.
// Computes singular integration on the vesicle surface.
template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::stokes(
    const Vec_t &force, Vec_t &velocity) const
{
    PROFILESTART();

    /*
    int imax(S_.getPosition().getGridDim().first);
    int jmax(S_.getPosition().getGridDim().second);
    int np = S_.getPosition().getStride();
    int nv = S_.getPosition().getNumSubs();

    std::auto_ptr<Sca_t> t1 = checkoutSca();
    std::auto_ptr<Sca_t> t2 = checkoutSca();
    std::auto_ptr<Vec_t> v1 = checkoutVec();
    std::auto_ptr<Vec_t> v2 = checkoutVec();

    ax(w_sph_inv_, S_.getAreaElement(), *t1);

    int numinputs = 3;
    const Sca_t* inputs[] = {&S_.getPosition(), &force, t1.get()};
    Sca_t* outputs[] = {v1.get(), v2.get(), t2.get()};

    for(int ii=0;ii < imax; ++ii)
        for(int jj=0;jj < jmax; ++jj)
        {
            //move_pole(ii, jj, outputs);
            assert(false); exit(1); //@bug: move_pole deprecated

            ax(w_sph_, *t2, *t2);
            xv(*t2, *v2, *v2);

            S_.getPosition().getDevice().DirectStokes(v1->begin(), v2->begin(),
                sing_quad_weights_.begin(), np, np, nv, S_.getPosition().begin(),
                ii * jmax + jj, ii * jmax + jj + 1, velocity.begin());
        }

    recycle(t1);
    recycle(t2);
    recycle(v1);
    recycle(v2);
    */

    velocity.replicate(force);
    stokes_.SetDensitySL(&force);
    stokes_.SetDensityDL(NULL);
    stokes_.SelfInteraction(velocity);

    PROFILEEND("SelfInteraction_",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::stokesSLPerVesicle(
    const Vec_t &force, Vec_t &velocity, const int vesicle_i) const
{
    //PROFILESTART();
    velocity.replicate(force);

    long Ncoef = params_.sh_order*(params_.sh_order+2);

    pvfmm::Vector<value_type> force_sl(force.size(), (value_type*)force.begin(), false); 
    pvfmm::Vector<value_type> vel_sl(velocity.size(), (value_type*)velocity.begin(), false); 
    
    pvfmm::Vector<value_type> force_coef, vel_coef;

    vel_coef.ReInit(0);
    vel_coef.ReInit(COORD_DIM*Ncoef);

    SphericalHarmonics<value_type>::Grid2SHCPerVesicle(force_sl, params_.sh_order, params_.sh_order, force_coef);

    pvfmm::Matrix<value_type> Mv(1,COORD_DIM*Ncoef, &vel_coef  [0], false);
    pvfmm::Matrix<value_type> Mf(1,COORD_DIM*Ncoef, &force_coef[0], false);
    pvfmm::Matrix<value_type> M(COORD_DIM*Ncoef, COORD_DIM*Ncoef, (value_type*)stokes_.GetSLMatrixi(vesicle_i), false);
    pvfmm::Matrix<value_type>::GEMM(Mv,Mf,M);

    SphericalHarmonics<value_type>::SHC2GridPerVesicle(vel_coef, params_.sh_order, params_.sh_order, vel_sl);

    //PROFILEEND("SL_SELF_PER_VESICLE",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::stokesDLPerVesicle(
    const Vec_t &force, Vec_t &velocity, const int vesicle_i) const
{
    //PROFILESTART();
    velocity.replicate(force);

    long Ncoef = params_.sh_order*(params_.sh_order+2);

    pvfmm::Vector<value_type> force_dl(force.size(), (value_type*)force.begin(), false); 
    pvfmm::Vector<value_type> vel_dl(velocity.size(), (value_type*)velocity.begin(), false); 
    
    pvfmm::Vector<value_type> force_coef, vel_coef;

    vel_coef.ReInit(0);
    vel_coef.ReInit(COORD_DIM*Ncoef);

    SphericalHarmonics<value_type>::Grid2SHCPerVesicle(force_dl, params_.sh_order, params_.sh_order, force_coef);

    pvfmm::Matrix<value_type> Mv(1,COORD_DIM*Ncoef, &vel_coef  [0], false);
    pvfmm::Matrix<value_type> Mf(1,COORD_DIM*Ncoef, &force_coef[0], false);
    pvfmm::Matrix<value_type> M(COORD_DIM*Ncoef, COORD_DIM*Ncoef, (value_type*)stokes_.GetDLMatrixi(vesicle_i), false);
    pvfmm::Matrix<value_type>::GEMM(Mv,Mf,M);

    SphericalHarmonics<value_type>::SHC2GridPerVesicle(vel_coef, params_.sh_order, params_.sh_order, vel_dl);

    //PROFILEEND("DL_SELF_PER_VESICLE",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::stokes_double_layer(
    const Vec_t &force, Vec_t &velocity) const
{
    PROFILESTART();

    int imax(S_.getPosition().getGridDim().first);
    int jmax(S_.getPosition().getGridDim().second);
    int np = S_.getPosition().getStride();
    int nv = S_.getPosition().getNumSubs();

    std::auto_ptr<Sca_t> t1 = checkoutSca();
    std::auto_ptr<Sca_t> t2 = checkoutSca();
    std::auto_ptr<Vec_t> v1 = checkoutVec();
    std::auto_ptr<Vec_t> v2 = checkoutVec();
    std::auto_ptr<Vec_t> v3 = checkoutVec();

    ax(w_sph_inv_, S_.getAreaElement(), *t1);

    int numinputs = 4;
    const Sca_t* inputs[] = {&S_.getPosition(), &S_.getNormal(),   &force, t1.get()};
    Sca_t*      outputs[] = { v1.get()        ,  v3.get()      , v2.get(), t2.get()};

    for(int ii=0;ii < imax; ++ii)
        for(int jj=0;jj < jmax; ++jj)
        {
            //move_pole(ii, jj, outputs);
            assert(false); exit(1); //@bug: move_pole deprecated

            ax(w_sph_, *t2, *t2);
            xv(*t2, *v2, *v2);

            S_.getPosition().getDevice().DirectStokesDoubleLayer(v1->begin(), v3->begin(), v2->begin(),
                sing_quad_weights_.begin(), np, nv, S_.getPosition().begin(),
                ii * jmax + jj, ii * jmax + jj + 1, velocity.begin());

        }

    recycle(t1);
    recycle(t2);
    recycle(v1);
    recycle(v2);

    PROFILEEND("DblLayerSelfInteraction_",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
operator()(const Vec_t &x_new, Vec_t &time_mat_vec) const
{
    PROFILESTART();
    std::auto_ptr<Vec_t> fb = checkoutVec();

    COUTDEBUG("Time matvec");
    Intfcl_force_.linearBendingForce(S_, x_new, *fb);
    CHK(stokes(*fb, time_mat_vec));
    axpy(-dt_, time_mat_vec, x_new, time_mat_vec);
    recycle(fb);
    PROFILEEND("",0);

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
operator()(const Sca_t &tension, Sca_t &div_stokes_fs) const
{
    std::auto_ptr<Vec_t> fs = checkoutVec();
    std::auto_ptr<Vec_t> u = checkoutVec();

    COUTDEBUG("Tension matvec");
    Intfcl_force_.tensileForce(S_, tension, *fs);
    CHK(stokes(*fs, *u));
    S_.div(*u, div_stokes_fs);

    recycle(fs);
    recycle(u);

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
operator()(const Vec_t &x_new, const Sca_t &tension, Vec_t &time_mat_vec, Sca_t &div_stokes_fs) const
{
    std::auto_ptr<Vec_t> f = checkoutVec();
    std::auto_ptr<Vec_t> dlp_tmp = checkoutVec();
    f->replicate(x_new);
    dlp_tmp->replicate(x_new);
    time_mat_vec.replicate(x_new);
    
    axpy(dt_, x_new, *dlp_tmp);
    //double ts_if = omp_get_wtime();
    Intfcl_force_.implicitTractionJump(S_, *dlp_tmp, tension, *f);
    //double te_if = omp_get_wtime();
    //INFO("intfcl force time: "<<te_if-ts_if);
    
    //double ts_set = omp_get_wtime();
    stokes_.SetDensitySL(f.get());
    if(ves_props_.has_contrast)
    {
        av(ves_props_.dl_coeff, x_new, *dlp_tmp);
        stokes_.SetDensityDL(dlp_tmp.get());
    }
    else
    {
        stokes_.SetDensityDL(NULL);
    }
    //double te_set = omp_get_wtime();
    //INFO("stokes set time: "<<te_set-ts_set);
    
    //double ts_self = omp_get_wtime();
    stokes_.SelfInteraction(time_mat_vec);
    //double te_self = omp_get_wtime();
    //INFO("stokes self: "<<te_self-ts_self);
    
    //double ts_div = omp_get_wtime();
    S_.div(time_mat_vec, div_stokes_fs);
    axpy((value_type) -1.0, div_stokes_fs, div_stokes_fs);
    //double te_div = omp_get_wtime();
    //INFO("surf div: "<<te_div-ts_div);
    
    if(ves_props_.has_contrast){
        av(ves_props_.vel_coeff, x_new, *f);
        axpy(static_cast<value_type>(-1.0), time_mat_vec, *f, time_mat_vec);
    }
    else{
        axpy(static_cast<value_type>(-1.0), time_mat_vec, x_new, time_mat_vec);
    }
    
    recycle(f);
    recycle(dlp_tmp);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
operator()(const Vec_t &x_new, const Sca_t &tension, const Arr_t &lambda,
        Vec_t &time_mat_vec, Sca_t &tension_mat_vec, Arr_t &lambda_mat_vec) const
{
    std::auto_ptr<Vec_t> f_col = checkoutVec();
    std::auto_ptr<Vec_t> vel_f_col = checkoutVec();
    std::auto_ptr<Sca_t> div_vel_f_col = checkoutSca();
    
    f_col->replicate(x_new);
    vel_f_col->replicate(x_new);
    div_vel_f_col->replicate(tension);

    (*this)(x_new, tension, time_mat_vec, tension_mat_vec);

    CVJacobianTrans(lambda, *f_col);
    stokes(*f_col, *vel_f_col);
    
    axpy(static_cast<value_type>(-1.0), *vel_f_col, time_mat_vec, time_mat_vec);
    
    S_.div(*vel_f_col, *div_vel_f_col);
    axpy(static_cast<value_type>(1.0), *div_vel_f_col, tension_mat_vec, tension_mat_vec);

    axpy(dt_, x_new, *f_col);
    CVJacobian(*f_col, lambda_mat_vec);
    LCPSelect(lambda, lambda_mat_vec);
    
    recycle(f_col);
    recycle(vel_f_col);
    recycle(div_vel_f_col);

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
operator()(const Arr_t &lambda, Arr_t &lambda_mat_vec) const
{
    cblas_dgemv(CblasRowMajor, CblasNoTrans, num_cvs_, num_cvs_, 
            1.0, lcp_matrix_.begin(), num_cvs_, lambda.begin(), 1, 0.0, lambda_mat_vec.begin(), 1);
    LCPSelect(lambda, lambda_mat_vec);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
operator()(const Vec_t &x_new, const Sca_t &tension, Vec_t &time_mat_vec, Sca_t &div_stokes_fs, const int vesicle_i) const
{
    int omp_id = omp_get_thread_num();

    Vec_t f(1, x_new.getShOrder());
    Vec_t dlp_tmp(1, x_new.getShOrder());
    f.replicate(x_new);
    dlp_tmp.replicate(x_new);
 
    axpy(dt_, x_new, dlp_tmp);
    Intfcl_force_.implicitTractionJumpPerVesicle(*S_is_[omp_id], dlp_tmp, tension, f, vesicle_i);
    stokesSLPerVesicle(f, time_mat_vec, vesicle_i);

    if (ves_props_.has_contrast){
        Arr_t dl_coeff_tmp(1);
        (*dl_coeff_tmp.begin()) = (ves_props_.dl_coeff.begin()[vesicle_i]);
        av(dl_coeff_tmp, x_new, f);
        
        stokesDLPerVesicle(f, dlp_tmp, vesicle_i);
        axpy(static_cast<value_type>(1.0), dlp_tmp, time_mat_vec, time_mat_vec);
    }
    
    S_is_[omp_id]->div(time_mat_vec, div_stokes_fs);
    axpy(static_cast<value_type>(-1.0), div_stokes_fs, div_stokes_fs);
    
    if (ves_props_.has_contrast){
        Arr_t vel_coeff_tmp(1);
        (*vel_coeff_tmp.begin()) = (ves_props_.vel_coeff.begin()[vesicle_i]);
        av(vel_coeff_tmp, x_new, f);

        axpy(static_cast<value_type>(-1.0), time_mat_vec, f, time_mat_vec);
    }
    else{
        axpy(static_cast<value_type>(-1.0), time_mat_vec, x_new, time_mat_vec);
    }
    
    return ErrorEvent::Success;
}

// This function evaluates J*dx, where J is the contact volumes' Jacobian,
// dx is the displacement stored in x_new.
// J is sparse matrix of size ncv*N, where ncv is the number of contact
// volumes, N is the total points on vesicles.
// The ith row of J is the contact volume gradient of ith contact volume.
// Instead of forming J, we use vGrad_index which stores the components
// index(belongs to which contact volume) to evaluate J*dx.
// The result dx_matvec=J*dx is of size ncv*1, the ith components of 
// dx_matvec is the sum of all the components of xy(x_new,vGrad) which have
// index i(stored in vGrad_index).
// Since we are accessing all the components of the Vec_t, we copy the vector
// from device to host.
template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
CVJacobian(const Vec_t &x_new, Arr_t &lambda_mat_vec) const
{
    std::auto_ptr<Vec_t> x_vgrad = checkoutVec();
    std::auto_ptr<Vec_t> x_new_up = checkoutVec();
    std::auto_ptr<Vec_t> wrk1 = checkoutVec();
    std::auto_ptr<Vec_t> wrk2 = checkoutVec();
    x_vgrad->replicate(vgrad_);
    x_new_up->replicate(vgrad_);
    wrk1->replicate(vgrad_);
    wrk2->replicate(vgrad_);

    Resample(x_new, sht_, sht_upsample_, *wrk1, *wrk2, *x_new_up);

    xy(*x_new_up, vgrad_, *x_vgrad);

    //axpy(dt_, *x_vgrad, *x_vgrad);
    
    std::vector<value_type> x_vgrad_host;
    x_vgrad_host.resize(x_vgrad->size(), 0.0);
    // copy x_vgrad to host
    x_vgrad->getDevice().Memcpy(&x_vgrad_host.front(), x_vgrad->begin(),
        x_vgrad->size() * sizeof(value_type),
        x_vgrad->getDevice().MemcpyDeviceToHost);

    std::vector<value_type> lambda_mat_vec_host(lambda_mat_vec.size(), 0.0);

    size_t ncount = vgrad_.size();
    //#pragma omp parallel for
    for (size_t icount = 0; icount < ncount; icount++)
    {
        if(vgrad_ind_[icount] > 0)
        {
            lambda_mat_vec_host[ vgrad_ind_[icount] - 1 ] += x_vgrad_host[icount];
            //#pragma omp atomic
            //{
            //}
        }
    }

    // copy lambda_mat_vec to device
    lambda_mat_vec.getDevice().Memcpy(lambda_mat_vec.begin(), &lambda_mat_vec_host.front(),
            lambda_mat_vec.size() * sizeof(value_type),
            lambda_mat_vec.getDevice().MemcpyHostToDevice);

    recycle(x_vgrad);
    recycle(x_new_up);
    recycle(wrk1);
    recycle(wrk2);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
CVJacobianTrans(const Arr_t &lambda, Vec_t &f_col) const
{
    std::vector<value_type> f_col_host;
    f_col_host.resize(vgrad_.size(), 0.0);

    std::vector<value_type> vgrad_host;
    vgrad_host.resize(vgrad_.size(), 0.0);
    // copy vgrad_ to host
    vgrad_.getDevice().Memcpy(&vgrad_host.front(), vgrad_.begin(),
        vgrad_.size() * sizeof(value_type),
        vgrad_.getDevice().MemcpyDeviceToHost);

    std::vector<value_type> lambda_host;
    lambda_host.resize(lambda.size(), 0.0);
    // copy lambda to host
    lambda.getDevice().Memcpy(&lambda_host.front(), lambda.begin(),
            lambda.size() * sizeof(value_type),
            lambda.getDevice().MemcpyDeviceToHost);

    size_t ncount = vgrad_.size();
    #pragma omp parallel for
    for (size_t icount = 0; icount < ncount; icount++)
    {
        if(vgrad_ind_[icount] > 0)
            f_col_host[icount] = lambda_host[ vgrad_ind_[icount] - 1 ] * vgrad_host[icount];
    }
    
    /*
    f_col.getDevice().Memcpy(f_col.begin(), &f_col_host.front(),
        f_col.size() * sizeof(value_type),
        f_col.getDevice().MemcpyHostToDevice);
    */
        
    //filtered f_col_up and downsample
    std::auto_ptr<Vec_t> xwrk = checkoutVec();
    std::auto_ptr<Vec_t> f_col_up = checkoutVec();
    std::auto_ptr<Vec_t> wrk1 = checkoutVec();
    std::auto_ptr<Vec_t> wrk2 = checkoutVec();
    xwrk->replicate(vgrad_);
    f_col_up->replicate(vgrad_);
    wrk1->replicate(vgrad_);
    wrk2->replicate(vgrad_);
    
    xwrk->getDevice().Memcpy(xwrk->begin(), &f_col_host.front(),
        xwrk->size() * sizeof(value_type),
        xwrk->getDevice().MemcpyHostToDevice);

    sht_filter_high(*xwrk, *f_col_up, &sht_upsample_, params_.rep_exponent);
    Resample(*f_col_up, sht_upsample_, sht_, *wrk1, *wrk2, f_col);
    
    recycle(xwrk);
    recycle(f_col_up);
    recycle(wrk1);
    recycle(wrk2);
    //end of filtered

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
LCPSelect(const Arr_t &lambda, Arr_t &lambda_mat_vec) const
{
    std::vector<value_type> lambda_host;
    std::vector<value_type> lambda_mat_vec_host;

    lambda_host.resize(lambda.size(), 0.0);
    lambda_mat_vec_host.resize(lambda.size(), 0.0);

    lambda.getDevice().Memcpy(&lambda_host.front(), lambda.begin(),
            lambda.size() * sizeof(value_type),
            lambda.getDevice().MemcpyDeviceToHost);
    lambda_mat_vec.getDevice().Memcpy(&lambda_mat_vec_host.front(), lambda_mat_vec.begin(),
            lambda_mat_vec.size() * sizeof(value_type),
            lambda_mat_vec.getDevice().MemcpyDeviceToHost);

    size_t ncount = lambda.size();
    //#pragma omp parallel for
    for (size_t icount = 0; icount < ncount; icount++)
    {
        if(PA_[icount] == 0)
            lambda_mat_vec_host[icount] = lambda_host[icount];
    
    }

    lambda_mat_vec.getDevice().Memcpy(lambda_mat_vec.begin(), &lambda_mat_vec_host.front(), 
            lambda_mat_vec.size() * sizeof(value_type),
            lambda_mat_vec.getDevice().MemcpyHostToDevice);

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
SolveLCP(Vec_t &u_lcp, Sca_t &ten_lcp, Arr_t &lambda_lcp, Arr_t &cvs) const
{
    /*
     * LCP_flag = 1, preprocessing
     * LCP_flag = 2, iterating
     * LCP_flag = 3, relative error termination
     * LCP_flag = 4, absolute error termination
     * LCP_flag = 5, stagnation
     * LCP_flag = 6, local minima
     * LCP_flag = 7, nondescent
     * LCP_flag = 8, maxlimit iters
     */

    int LCP_flag = 1;
    
    int LCP_n = num_cvs_;
    int LCP_max_iter = 100;

    // LCP parameters
    value_type LCP_eps = 1e-16;
    value_type LCP_h = 1e-7;
    value_type LCP_alpha = 0.5;
    value_type LCP_beta = 0.001;
    value_type LCP_gamma = 1e-28;
    value_type LCP_rho = LCP_eps;

    // setup
    std::vector<value_type> LCP_convergence(LCP_max_iter, 0.0);
    value_type LCP_err = 1e+16;
    int LCP_iter = 1;

    // checkout vecs, scas and arrs for calculation
    std::auto_ptr<Vec_t> time_mat_vec = checkoutVec();
    std::auto_ptr<Vec_t> du = checkoutVec();
    time_mat_vec->replicate(u_lcp);
    du->replicate(u_lcp);

    std::auto_ptr<Sca_t> tension_mat_vec = checkoutSca();
    std::auto_ptr<Sca_t> dtension = checkoutSca();
    tension_mat_vec->replicate(ten_lcp);
    dtension->replicate(ten_lcp);
    
    Arr_t lambda_mat_vec(LCP_n);
    Arr_t dlambda(LCP_n);
    Arr_t LCP_y(LCP_n);
    Arr_t LCP_phi(LCP_n);
    
    // init vecs, scas and arrs
    axpy(static_cast<value_type>(0.0), u_lcp, u_lcp);
    axpy(static_cast<value_type>(0.0), *time_mat_vec, *time_mat_vec);
    axpy(static_cast<value_type>(0.0), *du, *du);

    axpy(static_cast<value_type>(0.0), ten_lcp, ten_lcp);
    axpy(static_cast<value_type>(0.0), *tension_mat_vec, *tension_mat_vec);
    axpy(static_cast<value_type>(0.0), *dtension, *dtension);

    Arr_t::getDevice().axpy(static_cast<value_type>(0.0), lambda_lcp.begin(), static_cast<value_type*>(NULL), 
                lambda_lcp.size(), lambda_lcp.begin());
    Arr_t::getDevice().axpy(static_cast<value_type>(0.0), lambda_mat_vec.begin(), static_cast<value_type*>(NULL), 
                lambda_mat_vec.size(), lambda_mat_vec.begin());
    Arr_t::getDevice().axpy(static_cast<value_type>(0.0), dlambda.begin(), static_cast<value_type*>(NULL), 
                dlambda.size(), dlambda.begin());
    Arr_t::getDevice().axpy(static_cast<value_type>(0.0), LCP_y.begin(), static_cast<value_type*>(NULL), 
                LCP_y.size(), LCP_y.begin());
    Arr_t::getDevice().axpy(static_cast<value_type>(0.0), LCP_phi.begin(), static_cast<value_type*>(NULL), 
                LCP_phi.size(), LCP_phi.begin());
    
    // allocate memory for solving Newton's system
    size_t vsz(stokesBlockSize()), tsz(tensionBlockSize()), lsz(num_cvs_);
    ASSERT(S_.getPosition().size()==vsz,"Bad sizes");
    size_t N_size = vsz+tsz+lsz;
    value_type x_host[N_size], rhs_host[N_size];
    
    int iter(params_.time_iter_max);
    value_type relres(params_.time_tol);
    
    value_type LCP_old_err;
    LCP_flag = 2;
    while(LCP_iter < LCP_max_iter)
    {
        // if size of PA is large, should we not resize every iteration
        PA_.clear();
        PA_.resize(num_cvs_, 1);
        (*this)(u_lcp, ten_lcp, lambda_lcp, *time_mat_vec, *tension_mat_vec, lambda_mat_vec);
        
        Arr_t::getDevice().axpy(static_cast<value_type>(1.0), lambda_mat_vec.begin(), cvs.begin(), 
                num_cvs_, LCP_y.begin());
        
        minmap(LCP_y, lambda_lcp, LCP_phi);
        
        LCP_old_err = LCP_err;
        LCP_err = 0.5 * Arr_t::getDevice().AlgebraicDot(LCP_phi.begin(), LCP_phi.begin(), LCP_phi.size());

        INFO("LCP Newtown iter: "<<LCP_iter<<". -- err: "<<LCP_err<<" -- relative err: "
                <<fabs(LCP_err - LCP_old_err)/fabs(LCP_old_err) );

        INFO("lambda: "<<lambda_lcp);

        // relative stopping criteria
        if(fabs(LCP_err - LCP_old_err)/fabs(LCP_old_err) < 1e-6)
        {
            LCP_flag = 3;
            break;
        }
        
        // absolute stopping criteria
        if(LCP_err < 1e-21)
        {
            LCP_flag =4;
            break;
        }

        // solve the Newton system to get descent direction
        // copy device type to value_type array to call GMRES
        axpy(static_cast<value_type>(0.0), *du, *du);
        axpy(static_cast<value_type>(0.0), *dtension, *dtension);
        Arr_t::getDevice().axpy(static_cast<value_type>(0.0), dlambda.begin(), static_cast<value_type*>(NULL), 
                dlambda.size(), dlambda.begin());
        Arr_t::getDevice().axpy(static_cast<value_type>(-1.0), LCP_phi.begin(), static_cast<value_type*>(NULL), 
                LCP_phi.size(), LCP_phi.begin());

        // copy to unknown solution
        du->getDevice().Memcpy(x_host, du->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        dtension->getDevice().Memcpy(x_host+vsz, dtension->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        dlambda.getDevice().Memcpy(x_host+vsz+tsz, dlambda.begin(), lsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    
        // copy to rhs
        du->getDevice().Memcpy(rhs_host, du->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        dtension->getDevice().Memcpy(rhs_host+vsz, dtension->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        LCP_phi.getDevice().Memcpy(rhs_host+vsz+tsz, LCP_phi.begin(), lsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

        // solve the linear system using gmres
        // set relative tolerance
        value_type LCP_lin_tol = 1.0e-03;
        //if( MaxAbs(LCP_phi) < 1.0e-05 )
        //    LCP_lin_tol = 5.0e-01;
        
        INFO("Begin of SolveLCP Newton system.");
        int solver_ret = linear_solver_gmres_(JacobiImplicitLCPApply, JacobiImplicitLCPPrecond, x_host, rhs_host, 
            LCP_lin_tol, 8.0e-11, N_size, iter, 200);
        INFO("End of SolveLCP Newton system.");

        // copy host to device
        du->getDevice().Memcpy(du->begin(), x_host    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
        dtension->getDevice().Memcpy(dtension->begin(), x_host+vsz, tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
        dlambda.getDevice().Memcpy(dlambda.begin(), x_host+vsz+tsz, lsz * sizeof(value_type), device_type::MemcpyHostToDevice);

        // TODO: the gradient of meric function \nabla\theta is LCP_phi^T*LCP_matrix, 
        // which we can't get with matrix free version LCP solver, either form LCP matrix
        // explicitly or do some approximate \nabla\theta calculation.
        // So LCP_flag 6,7 is not tested, and we use Newton's method withou line search for now.
        // (Line search requires \nabla\theta)

        value_type dlambda_norm = Arr_t::getDevice().AlgebraicDot(dlambda.begin(), dlambda.begin(), dlambda.size());
        if(dlambda_norm < LCP_eps)
        {
            LCP_flag = 5;
            break;
            // could use dlambda = -nabla_theta
        }
        
        // TODO: test for whether dropped into a local minima flag 6
        // TODO: test for sufficient descent direction flag 7

        // update solution with direction calculated
        // TODO: do line search with \nabla\theta for acceptable LCP_tau
        value_type LCP_tau = 1.0;

        axpy(LCP_tau, *du, u_lcp, u_lcp);
        axpy(LCP_tau, *dtension, ten_lcp, ten_lcp);
        Arr_t::getDevice().axpy(LCP_tau, dlambda.begin(), lambda_lcp.begin(), 
                dlambda.size(), lambda_lcp.begin());
        
        //bool lambdaupdated = false;
        //check_lambda();
        // if lambdaupdated == true, solve for du and dtension of new lambda
        // end of update solution
    }

    recycle(time_mat_vec);
    recycle(du);
    recycle(tension_mat_vec);
    recycle(dtension);

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
SolveLCPSmall(Arr_t &lambda_lcp, Arr_t &cvs) const
{
    /*
     * LCP_flag = 1, preprocessing
     * LCP_flag = 2, iterating
     * LCP_flag = 3, relative error termination
     * LCP_flag = 4, absolute error termination
     * LCP_flag = 5, stagnation
     * LCP_flag = 6, local minima
     * LCP_flag = 7, nondescent
     * LCP_flag = 8, maxlimit iters
     */

    int LCP_flag = 1;
    
    int LCP_n = num_cvs_;
    int LCP_max_iter = 100;

    // LCP parameters
    value_type LCP_eps = 1e-16;
    value_type LCP_h = 1e-7;
    value_type LCP_alpha = 0.5;
    value_type LCP_beta = 0.001;
    value_type LCP_gamma = 1e-28;
    value_type LCP_rho = LCP_eps;

    // setup
    std::vector<value_type> LCP_convergence(LCP_max_iter, 0.0);
    value_type LCP_err = 1e+16;
    int LCP_iter = 1;

    // arrs for calculation
    Arr_t lambda_mat_vec(LCP_n);
    Arr_t dlambda(LCP_n);
    Arr_t LCP_y(LCP_n);
    Arr_t LCP_phi(LCP_n);
    
    // init vecs, scas and arrs
    Arr_t::getDevice().axpy(static_cast<value_type>(0.0), lambda_lcp.begin(), static_cast<value_type*>(NULL), 
                lambda_lcp.size(), lambda_lcp.begin());
    Arr_t::getDevice().axpy(static_cast<value_type>(0.0), lambda_mat_vec.begin(), static_cast<value_type*>(NULL), 
                lambda_mat_vec.size(), lambda_mat_vec.begin());
    Arr_t::getDevice().axpy(static_cast<value_type>(0.0), dlambda.begin(), static_cast<value_type*>(NULL), 
                dlambda.size(), dlambda.begin());
    Arr_t::getDevice().axpy(static_cast<value_type>(0.0), LCP_y.begin(), static_cast<value_type*>(NULL), 
                LCP_y.size(), LCP_y.begin());
    Arr_t::getDevice().axpy(static_cast<value_type>(0.0), LCP_phi.begin(), static_cast<value_type*>(NULL), 
                LCP_phi.size(), LCP_phi.begin());
    
    // allocate memory for solving Newton's system
    size_t lsz(num_cvs_);
    size_t N_size = lsz;
    value_type x_host[N_size], rhs_host[N_size];
    
    int iter(params_.time_iter_max);
    value_type relres(params_.time_tol);
    
    value_type LCP_old_err;
    LCP_flag = 2;
    while(LCP_iter < LCP_max_iter)
    {
        // if size of PA is large, should we not resize every iteration
        PA_.clear();
        PA_.resize(num_cvs_, 1);
        (*this)(lambda_lcp, lambda_mat_vec);
        
        Arr_t::getDevice().axpy(static_cast<value_type>(1.0), lambda_mat_vec.begin(), cvs.begin(), 
                num_cvs_, LCP_y.begin());
        
        minmap(LCP_y, lambda_lcp, LCP_phi);
        
        LCP_old_err = LCP_err;
        LCP_err = 0.5 * Arr_t::getDevice().AlgebraicDot(LCP_phi.begin(), LCP_phi.begin(), LCP_phi.size());

        INFO("LCP small Newtown iter: "<<LCP_iter<<". -- err: "<<LCP_err<<" -- relative err: "
                <<fabs(LCP_err - LCP_old_err)/fabs(LCP_old_err) );

        INFO("lambda small: "<<lambda_lcp);

        // relative stopping criteria
        if(fabs(LCP_err - LCP_old_err)/fabs(LCP_old_err) < 1e-6)
        {
            LCP_flag = 3;
            break;
        }
        
        // absolute stopping criteria
        if(LCP_err < 1e-21)
        {
            LCP_flag =4;
            break;
        }

        // solve the Newton system to get descent direction
        // copy device type to value_type array to call GMRES
        Arr_t::getDevice().axpy(static_cast<value_type>(0.0), dlambda.begin(), static_cast<value_type*>(NULL), 
                dlambda.size(), dlambda.begin());
        Arr_t::getDevice().axpy(static_cast<value_type>(-1.0), LCP_phi.begin(), static_cast<value_type*>(NULL), 
                LCP_phi.size(), LCP_phi.begin());

        // copy to unknown solution
        dlambda.getDevice().Memcpy(x_host, dlambda.begin(), lsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
    
        // copy to rhs
        LCP_phi.getDevice().Memcpy(rhs_host, LCP_phi.begin(), lsz * sizeof(value_type), device_type::MemcpyDeviceToHost);

        // solve the linear system using gmres
        INFO("Begin of SolveLCPSmall Newton system.");
        INFO("rhs: "<<LCP_phi);
        int solver_ret = linear_solver_gmres_(LCPApply, LCPPrecond, x_host, rhs_host, 
            relres, 0, N_size, iter, 200);
        INFO("End of SolveLCPSmall Newton system.");

        // copy host to device
        dlambda.getDevice().Memcpy(dlambda.begin(), x_host, lsz * sizeof(value_type), device_type::MemcpyHostToDevice);

        // TODO: the gradient of meric function \nabla\theta is LCP_phi^T*LCP_matrix, 
        // which we can't get with matrix free version LCP solver, either form LCP matrix
        // explicitly or do some approximate \nabla\theta calculation.
        // So LCP_flag 6,7 is not tested, and we use Newton's method withou line search for now.
        // (Line search requires \nabla\theta)

        value_type dlambda_norm = Arr_t::getDevice().AlgebraicDot(dlambda.begin(), dlambda.begin(), dlambda.size());
        if(dlambda_norm < LCP_eps)
        {
            LCP_flag = 5;
            break;
            // could use dlambda = -nabla_theta
        }
        
        // TODO: test for whether dropped into a local minima flag 6
        // TODO: test for sufficient descent direction flag 7

        // update solution with direction calculated
        // TODO: do line search with \nabla\theta for acceptable LCP_tau
        value_type LCP_tau = 1.0;

        Arr_t::getDevice().axpy(LCP_tau, dlambda.begin(), lambda_lcp.begin(), 
                dlambda.size(), lambda_lcp.begin());
        
        //bool lambdaupdated = false;
        //check_lambda();
        // if lambdaupdated == true, solve for du and dtension of new lambda
        // end of update solution
        LCP_iter++;
    }
    
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
minmap(const Arr_t &xin1, const Arr_t &xin2, Arr_t &xout) const
{
    const value_type* x1i = xin1.begin();
    const value_type* x2i = xin2.begin();
    value_type* xoi = xout.begin();
    
    size_t length = xin1.size();

    #pragma omp parallel for
    for (size_t idx = 0; idx < length; idx++)
    {
        xoi[idx] = (x1i[idx] < x2i[idx]) ? x1i[idx] : x2i[idx];
        PA_[idx] = (x1i[idx] < x2i[idx]) ? 1 : 0;
    }

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
RemoveContactSimple(Vec_t &u1, const Vec_t &x_old) const
{
    std::auto_ptr<Vec_t> xwrk = checkoutVec();
    xwrk->replicate(S_.getPosition());

    std::auto_ptr<Vec_t> xwrk_up = checkoutVec();
    std::auto_ptr<Vec_t> wrk1 = checkoutVec();
    std::auto_ptr<Vec_t> wrk2 = checkoutVec();
    xwrk_up->replicate(vgrad_);
    wrk1->replicate(vgrad_);
    wrk2->replicate(vgrad_);
    
    std::auto_ptr<Sca_t> twrk = checkoutSca();
    twrk->replicate(tension_);

    // pos_s stores the start configuration
    // pos_e stores the end configuration
    // vGrad stores the gradient of contact volumes
    int np_up = 2*params_.upsample_freq*(params_.upsample_freq+1);
    static std::vector<value_type> pos_s(np_up*COORD_DIM*nv_, 0.0);
    static std::vector<value_type> pos_e(np_up*COORD_DIM*nv_, 0.0);
    static std::vector<value_type> vGrad(np_up*COORD_DIM*nv_, 0.0);
    
    static pvfmm::Vector<value_type> pos_s_pole, pos_e_pole;
    
    // the candidate position and copy to end configuration
    axpy(static_cast<value_type> (1.0), u1, x_old, *xwrk);
    
    GetColPosAll(x_old, pos_s, pos_s_pole);
    GetColPosAll(*xwrk, pos_e, pos_e_pole);

    if(params_.periodic_length > 0)
    {
        TransferVesicle(pos_s, pos_e, pos_s_pole, pos_e_pole);
    }

    int resolveCount = 0;
    std::vector<value_type> IV;
    std::vector<value_type> IS_BDV;
    
    CI_.getVolumeAndGradient(IV, IS_BDV, num_cvs_, vGrad, vgrad_ind_, pos_s, pos_e, &pos_s_pole[0], &pos_e_pole[0],
            params_.min_sep_dist, params_.periodic_length);
    while(num_cvs_ > 0)
    {
        // worker for array size of num_cvs_
        Arr_t awrk(num_cvs_);
        
        // cvs stores the contact volumes
        Arr_t cvs(num_cvs_);
        // copy contact volumes to cvs
        cvs.getDevice().Memcpy(cvs.begin(), &IV.front(), 
                num_cvs_ * sizeof(value_type),
                cvs.getDevice().MemcpyHostToDevice);
        
        // copy contact volume gradient to vgrad_
        vgrad_.getDevice().Memcpy(vgrad_.begin(), &vGrad.front(), 
                vgrad_.size() * sizeof(value_type),
                vgrad_.getDevice().MemcpyHostToDevice);

        // get magnitude of projection direction
        // filter projection direction
        sht_filter_high(vgrad_, *xwrk_up, &sht_upsample_, params_.rep_exponent);
        // downsample xwrk_up to xwrk
        Resample(*xwrk_up, sht_upsample_, sht_, *wrk1, *wrk2, *xwrk);

        CVJacobian(*xwrk, awrk);
        //CVJacobian(vgrad_, awrk);
        awrk.getDevice().xyInv(cvs.begin(), awrk.begin(), cvs.size(), awrk.begin());
        // projection direction
        CVJacobianTrans(awrk, *xwrk);
        
        axpy(static_cast<value_type>(-1.0), *xwrk, u1, u1);
        
        // get candidate position
        axpy(static_cast<value_type> (1.0), u1, x_old, *xwrk);

        resolveCount++;
        INFO("Projecting to avoid collision iter#: "<<resolveCount);
        INFO(cvs);

        GetColPosAll(*xwrk, pos_e, pos_e_pole);
        
        if(params_.periodic_length > 0)
        {
            GetColPosAll(x_old, pos_s, pos_s_pole);
            TransferVesicle(pos_s, pos_e, pos_s_pole, pos_e_pole);
        }

        CI_.getVolumeAndGradient(IV, IS_BDV, num_cvs_, vGrad, vgrad_ind_, pos_s, pos_e, &pos_s_pole[0], &pos_e_pole[0],
                params_.min_sep_dist, params_.periodic_length);
    }

    recycle(xwrk);
    recycle(xwrk_up);
    recycle(wrk1);
    recycle(wrk2);
    recycle(twrk);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ParallelRemoveContactSimple(Vec_t &u1, const Vec_t &x_old) const
{
    MPI_Comm comm = MPI_COMM_WORLD;
    pvfmm::Profile::Tic("RmContactSimple", &comm, true);

    std::auto_ptr<Vec_t> xwrk = checkoutVec();
    std::auto_ptr<Vec_t> xwrk_up = checkoutVec();
    std::auto_ptr<Vec_t> u1_up = checkoutVec();
    std::auto_ptr<Vec_t> x_old_up = checkoutVec();
    xwrk->replicate(S_.getPosition());
    xwrk_up->replicate(vgrad_);
    u1_up->replicate(vgrad_);
    x_old_up->replicate(vgrad_);
            
    axpy(static_cast<value_type> (1.0), u1, x_old, *xwrk);
    //if(params_.min_sep_dist>0)
    //{
        //ParallelGetVolumeAndGradient(x_old, *xwrk);
        ParallelGetVolumeAndGradientWithBoundary(x_old, *xwrk, 
                NULL, 0, 0);
                //fixed_bd->tri_vertices, fixed_bd->num_vertices_per_patch_1d, fixed_bd->num_patches);

        /*
        int np_up = 2*params_.sh_order*(params_.sh_order+1);
        if(params_.col_upsample)
            np_up = 2*params_.upsample_freq*(params_.upsample_freq+1);
        static std::vector<value_type> pos_s(np_up*COORD_DIM*nv_, 0.0);
        static std::vector<value_type> pos_e(np_up*COORD_DIM*nv_, 0.0);
        static std::vector<value_type> vGrad(np_up*COORD_DIM*nv_, 0.0);
        static pvfmm::Vector<value_type> pos_s_pole, pos_e_pole;

        GetColPosAll(x_old, pos_s, pos_s_pole);
        GetColPosAll(*xwrk, pos_e, pos_e_pole);

        sum_num_cvs_ = 0;
        num_cvs_ = 0;
        IV_.clear();
        IS_BDV_.clear();
        vgrad_.getDevice().Memset(vgrad_.begin(), 0, sizeof(value_type)*vgrad_.size());
        std::memset(&vgrad_ind_[0], 0, vgrad_ind_.size()*sizeof(vgrad_ind_[0]));

        if(params_.periodic_length > 0)
        {
            TransferVesicle(pos_s, pos_e, pos_s_pole, pos_e_pole);
        }
        CI_bd_.getVolumeAndGradient(IV_, IS_BDV_, num_cvs_, vGrad, vgrad_ind_, pos_s, pos_e, &pos_s_pole[0], &pos_e_pole[0],
                params_.min_sep_dist, fixed_bd->tri_vertices, nv_, params_.periodic_length);
        sum_num_cvs_ = num_cvs_;
        vgrad_.getDevice().Memcpy(vgrad_.begin(), &vGrad.front(), 
                vgrad_.size() * sizeof(value_type),
                vgrad_.getDevice().MemcpyHostToDevice);
        */
    //}
    /*
    if(sum_num_cvs_>0 && params_.min_sep_dist>0 && params_.col_upsample)
    {
        //u1_up
        pvfmm::Vector<value_type> u1_pvfmm(u1.size(), (value_type*)u1.begin(), false);
        pvfmm::Vector<value_type> u1_up_pvfmm(u1_up->size(), (value_type*)u1_up->begin(), false);
        static pvfmm::Vector<value_type> u1_coeff;
        SphericalHarmonics<value_type>::Grid2SHC(u1_pvfmm, params_.sh_order, params_.sh_order, u1_coeff);
        SphericalHarmonics<value_type>::SHC2Grid(u1_coeff, params_.sh_order, params_.upsample_freq, u1_up_pvfmm);


        // x_old_up
        pvfmm::Vector<value_type> x_old_pvfmm(x_old.size(), (value_type*)x_old.begin(), false);
        pvfmm::Vector<value_type> x_old_up_pvfmm(x_old_up->size(), (value_type*)x_old_up->begin(), false);
        SphericalHarmonics<value_type>::Grid2SHC(x_old_pvfmm, params_.sh_order, params_.sh_order, u1_coeff);
        SphericalHarmonics<value_type>::SHC2Grid(u1_coeff, params_.sh_order, params_.upsample_freq, x_old_up_pvfmm);
    }
    */
    int resolveCount = 0;
    while(sum_num_cvs_>0 && params_.min_sep_dist>0)
    {
        // worker for array size of num_cvs_
        Arr_t awrk(num_cvs_);
        
        // cvs stores the contact volumes
        Arr_t cvs(num_cvs_);
        // copy contact volumes to cvs
        cvs.getDevice().Memcpy(cvs.begin(), &IV_.front(), num_cvs_ * sizeof(value_type), cvs.getDevice().MemcpyHostToDevice);

        /*
        // get magnitude
        ParallelCVJacobian(vgrad_, awrk, true);
        awrk.getDevice().xyInv(cvs.begin(), awrk.begin(), cvs.size(), awrk.begin());

        // projection direction with magnitude
        ParallelCVJacobianTrans(awrk, *xwrk_up, true);
        
        if(params_.col_upsample)
        {
            axpy(static_cast<value_type>(-1.0), *xwrk_up, *u1_up, *u1_up);
            axpy(static_cast<value_type> (1.0), *u1_up, *x_old_up, *xwrk_up);

            pvfmm::Vector<value_type> xwrk_up_pvfmm(xwrk_up->size(), (value_type*)xwrk_up->begin(), false);
            pvfmm::Vector<value_type> xwrk_pvfmm(xwrk->size(), (value_type*)xwrk->begin(), false);
            static pvfmm::Vector<value_type> xwrk_up_coeff;
            SphericalHarmonics<value_type>::Grid2SHC(xwrk_up_pvfmm, params_.upsample_freq, params_.upsample_freq, xwrk_up_coeff);
            SphericalHarmonics<value_type>::SHC2Grid(xwrk_up_coeff, params_.upsample_freq, params_.sh_order, xwrk_pvfmm);
        }
        else
        {
            axpy(static_cast<value_type>(-1.0), *xwrk_up, u1, u1);
            axpy(static_cast<value_type> (1.0), u1, x_old, *xwrk);
        }
        */

        // get magnitude of projection direction
        if(params_.col_upsample)
        {
            // filter projection direction
            sht_filter_high(vgrad_, *xwrk_up, &sht_upsample_, params_.rep_exponent);
            // downsample xwrk_up to xwrk
            Resample(*xwrk_up, sht_upsample_, sht_, *u1_up, *x_old_up, *xwrk);
        }
        else
            sht_filter_high(vgrad_, *xwrk, &sht_, params_.rep_exponent);

        // get magnitude
        ParallelCVJacobian(*xwrk, awrk);

        /*
        std::cout<<"cvs: \n";
        for(int ii=0;ii<cvs.size();ii++)
            std::cout<<cvs.begin()[ii]<<std::endl;

        std::cout<<"awrk: \n";
        for(int ii=0;ii<awrk.size();ii++)
            std::cout<<awrk.begin()[ii]<<std::endl;
        */

        awrk.getDevice().xyInv(cvs.begin(), awrk.begin(), cvs.size(), awrk.begin());

        for(size_t idx=0; idx<awrk.size(); idx++)
        {
            if(awrk.begin()[idx] >= 0)
                awrk.begin()[idx] = -1e-10;
        }
        /*
        std::cout<<"mag: \n";
        for(int ii=0;ii<awrk.size();ii++)
            std::cout<<awrk.begin()[ii]<<std::endl;
        */
        
        // projection direction with magnitude
        ParallelCVJacobianTrans(awrk, *xwrk);

        // to be deleted
        /*
        ParallelCVJacobian(vgrad_, awrk, true);
        std::cout<<"awrk: \n";
        for(int ii=0;ii<awrk.size();ii++)
            std::cout<<awrk.begin()[ii]<<std::endl;
        */
        // end of to be deleted
        
        axpy(static_cast<value_type> (-1.0), *xwrk, u1, u1);
        axpy(static_cast<value_type> (1.0), u1, x_old, *xwrk);
        
        resolveCount++;
        INFO("Projecting to avoid collision iter#: "<<resolveCount);
        //COUT(cvs);
        //COUT(awrk);
        //COUT("Norm of vgrad: " << AlgebraicDot(vgrad_, vgrad_));
        
        //ParallelGetVolumeAndGradient(x_old, *xwrk);i
        ParallelGetVolumeAndGradientWithBoundary(x_old, *xwrk, 
                NULL, 0, 0);
                //fixed_bd->tri_vertices, fixed_bd->num_vertices_per_patch_1d, fixed_bd->num_patches);

        /*
        GetColPosAll(*xwrk, pos_e, pos_e_pole);

        sum_num_cvs_ = 0;
        num_cvs_ = 0;
        IV_.clear();
        IS_BDV_.clear();
        vgrad_.getDevice().Memset(vgrad_.begin(), 0, sizeof(value_type)*vgrad_.size());
        std::memset(&vgrad_ind_[0], 0, vgrad_ind_.size()*sizeof(vgrad_ind_[0]));

        if(params_.periodic_length > 0)
        {
            TransferVesicle(pos_s, pos_e, pos_s_pole, pos_e_pole);
        }
        CI_bd_.getVolumeAndGradient(IV_, IS_BDV_, num_cvs_, vGrad, vgrad_ind_, pos_s, pos_e, &pos_s_pole[0], &pos_e_pole[0], 
                params_.min_sep_dist, fixed_bd->tri_vertices, nv_, params_.periodic_length);
        sum_num_cvs_ = num_cvs_;
        vgrad_.getDevice().Memcpy(vgrad_.begin(), &vGrad.front(), 
                vgrad_.size() * sizeof(value_type),
                vgrad_.getDevice().MemcpyHostToDevice);
        */
    }

    u1.getDevice().Memcpy(u1.begin(), xwrk->begin(), u1.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);

    recycle(xwrk);
    recycle(xwrk_up);
    recycle(u1_up);
    recycle(x_old_up);
    
    pvfmm::Profile::Toc();
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
TransferVesicle(std::vector<value_type> &pos_s, std::vector<value_type> &pos_e, 
        pvfmm::Vector<value_type> &pole_s_pole, pvfmm::Vector<value_type> &pole_e_pole) const
{
    // Set point_coord, point_value, poly_connect
    size_t p1 = params_.sh_order;
    if(params_.col_upsample)
        p1 = params_.upsample_freq;
    size_t N_ves = pos_e.size()/(2*p1*(p1+1)*COORD_DIM); // Number of vesicles
    
    #pragma omp parallel for
    for(size_t k=0;k<N_ves;k++){ // Set point_coord
        value_type C[COORD_DIM]={0,0,0};
        for(long l=0;l<COORD_DIM;l++) C[l]=0;
        
        for(size_t i=0;i<p1+1;i++){
            for(size_t j=0;j<2*p1;j++){
                for(size_t l=0;l<COORD_DIM;l++){
                    C[l]+=pos_e[j+2*p1*(i+(p1+1)*(l+k*COORD_DIM))];
                }
            }
        }
      
        for(long l=0;l<COORD_DIM;l++) C[l]/=2*p1*(p1+1);
        for(long l=0;l<COORD_DIM;l++) C[l]=(floor(C[l]/params_.periodic_length))*params_.periodic_length;
      
        for(size_t i=0;i<p1+1;i++){
            for(size_t j=0;j<2*p1;j++){
                for(size_t l=0;l<COORD_DIM;l++){
                    pos_s[j+2*p1*(i+(p1+1)*(l+k*COORD_DIM))] -= C[l];
                    pos_e[j+2*p1*(i+(p1+1)*(l+k*COORD_DIM))] -= C[l];
                }
            }
        }

        for(size_t l=0; l<COORD_DIM;l++){
            pole_s_pole[k*COORD_DIM*2 + 2*l + 0] -= C[l];
            pole_s_pole[k*COORD_DIM*2 + 2*l + 1] -= C[l];
            pole_e_pole[k*COORD_DIM*2 + 2*l + 0] -= C[l];
            pole_e_pole[k*COORD_DIM*2 + 2*l + 1] -= C[l];
        }
    }
    
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
FormLCPMatrix(Arr_t &lcp_matrix) const
{
    // get surface resolution we are working on
    SurfContainer* Surf;
    Surf = &S_;

    // checkout workers
    std::auto_ptr<Vec_t> f_i = checkoutVec();
    std::auto_ptr<Vec_t> vel_i = checkoutVec();
    std::auto_ptr<Sca_t> ten_i = checkoutSca();
    
    f_i->replicate(Surf->getPosition());
    vel_i->replicate(Surf->getPosition());
    ten_i->replicate(Surf->getPosition());
    
    Arr_t lambda_lcp(num_cvs_);

    value_type *lambda_iter;
    // end of checkout workers

    // clear lcp_matrix
    Arr_t::getDevice().Memset(lcp_matrix.begin(), 0, sizeof(value_type)*lcp_matrix.size());
    
    // fill lcp_matrix
    for(int i=0;i<num_cvs_;i++)
    {
        // reset lambda_lcp
        Arr_t::getDevice().Memset(lambda_lcp.begin(), 0, sizeof(value_type)*lambda_lcp.size());
        
        // set i_th component of lambda_lcp as 1 to select i_th row of contact volume Jacobian
        lambda_iter = lambda_lcp.begin();
        lambda_iter += i;
        (*lambda_iter) = 1;

        // select i_th row of contact volume jacobian
        CVJacobianTrans(lambda_lcp, *f_i);
        // prepare rhs for Stokes linear system
        stokes(*f_i, *vel_i);
        Surf->div(*vel_i, *ten_i);
        //axpy(static_cast<value_type>(-1.0),*ten_i,*ten_i);
            
        // using GMRES Solver solve for velocity due to the i_th row of contact volume Jacobian
        size_t vsz(stokesBlockSize()), tsz(tensionBlockSize());
        ASSERT(S_.getPosition().size()==vsz,"Bad sizes");
        size_t N_size = vsz+tsz;

        // copy device type to value_type array to call GMRES
        value_type x_host[N_size], rhs_host[N_size];
        
        // copy to rhs
        vel_i->getDevice().Memcpy(rhs_host    , vel_i->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        ten_i->getDevice().Memcpy(rhs_host+vsz, ten_i->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        
        // init solution to zeros
        Arr_t::getDevice().axpy(static_cast<value_type>(0.0), rhs_host, static_cast<value_type*>(NULL), 
            N_size, x_host);
    
        // solve the linear system using gmres
        int solver_ret = linear_solver_gmres_(JacobiImplicitApply, JacobiImplicitPrecond, x_host, rhs_host, 
                params_.time_tol, 0, N_size, params_.time_iter_max, 200);

        // copy host to device
        vel_i->getDevice().Memcpy(vel_i->begin(), x_host    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
        axpy(dt_, *vel_i, *vel_i);
        // end of using GMRES Solver
       
        CVJacobian(*vel_i, lambda_lcp);
        for(int j=0;j<num_cvs_;j++)
        {
            // Accumulate LCPMatrix entry j,i, matrix is stored in row order
            lambda_iter = lambda_lcp.begin();
            lambda_iter += j;
            value_type val_tmp = (*lambda_iter);

            lambda_iter = lcp_matrix.begin();
            lambda_iter += (j*num_cvs_ + i);
            (*lambda_iter) += val_tmp;
        }
    }

    // release workers
    recycle(f_i);
    recycle(vel_i);
    recycle(ten_i);
    
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
FormLCPMatrixSparse(Arr_t &lcp_matrix) const
{
    typedef boost::unordered_map< int, Vec_t* > CVMAP;
    CVMAP cvmap;
    
    // get surface resolution we are working on
    SurfContainer* Surf;
    Surf = S_i_;

    // checkout workers
    std::auto_ptr<Vec_t> f_i = checkoutVec();
    std::auto_ptr<Vec_t> f_i_up = checkoutVec();
    std::auto_ptr<Vec_t> vel_i = checkoutVec();
    std::auto_ptr<Vec_t> vel_i_up = checkoutVec();
    std::auto_ptr<Vec_t> wrk1 = checkoutVec();
    std::auto_ptr<Vec_t> wrk2 = checkoutVec();
    std::auto_ptr<Sca_t> ten_i = checkoutSca();
    f_i->replicate(Surf->getPosition());
    f_i_up->resize(1, params_.upsample_freq);
    vel_i->replicate(Surf->getPosition());
    vel_i_up->resize(1, params_.upsample_freq);
    wrk1->resize(1, params_.upsample_freq);
    wrk2->resize(1, params_.upsample_freq);
    ten_i->replicate(Surf->getPosition());
    
    // clear lcp_matrix
    Arr_t::getDevice().Memset(lcp_matrix.begin(), 0, sizeof(value_type)*lcp_matrix.size());

    // number of vesicles
    int nves = nv_;
    // sh_order collision detection working on
    int sh_order = params_.upsample_freq;
    // iterator for vgrad
    value_type* iter_vgrad;
        
    // to copy current vesicle's vgrad_ to i_vgrad
    Vec_t i_vgrad(1, sh_order);
    // to copy current vesicle's vgrad_ind_ to i_vgrad_ind 
    std::vector<int> i_vgrad_ind(i_vgrad.size(), 0);
    
    size_t ncount = i_vgrad_ind.size();
    int vgrad_index = 0;
    // iterators for cvmap
    typename CVMAP::iterator got;
    typename CVMAP::const_iterator iter_i, iter_j;
    contact_vesicle_list_.clear();
    // TODO: OMP this for loop?
    for(int i_vesicle = 0; i_vesicle < nves; ++i_vesicle)
    {
        bool has_contact = false;

        // set position of current vesicle surface
        f_i->getDevice().Memcpy(f_i->begin(), &(S_.getPosition().begin()[i_vesicle*f_i->size()]),
                f_i->size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
        Surf->setPosition(*f_i);
        
        // copy the i_th vesicle's part in vgrad_
        iter_vgrad = vgrad_.begin();
        i_vgrad.getDevice().Memcpy( i_vgrad.begin(), &(iter_vgrad[i_vesicle*i_vgrad.size()]), 
                i_vgrad.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice );

        // copy the i_th vesicle's part in vgrad_ind_
        std::memcpy(&(i_vgrad_ind[0]), &(vgrad_ind_[i_vesicle*i_vgrad.size()]), i_vgrad.size()*sizeof(int));
        
        // form cvmap
        for(iter_i = cvmap.cbegin(); iter_i != cvmap.cend(); iter_i++)
            delete iter_i->second;
        cvmap.clear();
        for(size_t icount = 0; icount < ncount; icount++)
        {
            vgrad_index = i_vgrad_ind[icount];
            if(vgrad_index > 0)
            {   
                has_contact = true;
                got = cvmap.find(vgrad_index);
                if( got != cvmap.end() )
                {
                    got->second->begin()[icount] = i_vgrad.begin()[icount];
                }
                else
                {
                   cvmap.insert( std::make_pair< int, Vec_t* >( vgrad_index, new Vec_t(1, sh_order) ) );
                   axpy(static_cast<value_type>(0.0), i_vgrad, *cvmap[vgrad_index]);
                   cvmap[vgrad_index]->begin()[icount] = i_vgrad.begin()[icount];
                }
            }
        }

        if(has_contact)
            contact_vesicle_list_.push_back(i_vesicle);

        // iterate cvmap
        for(iter_i = cvmap.cbegin(); iter_i != cvmap.cend(); iter_i++)
        {
            // filter high freq
            sht_filter_high(*iter_i->second, *f_i_up, &sht_upsample_, params_.rep_exponent);
            // downsample f_i_up to f_i
            Resample(*f_i_up, sht_upsample_, sht_, *wrk1, *wrk2, *f_i);
            
            // prepare rhs for current vesicle
            stokesSLPerVesicle(*f_i, *vel_i, i_vesicle);
            Surf->div(*vel_i, *ten_i);
            //axpy(static_cast<value_type>(-1.0), *ten_i, *ten_i);

            // using GMRES Solver solve for velocity due to the i_th row of contact volume Jacobian
            size_t vsz(f_i->size()), tsz(ten_i->size());
            size_t N_size = vsz+tsz;

            // copy device type to value_type array to call GMRES
            value_type x_host[N_size], rhs_host[N_size];
        
            // copy to rhs
            vel_i->getDevice().Memcpy(rhs_host    , vel_i->begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
            ten_i->getDevice().Memcpy(rhs_host+vsz, ten_i->begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        
            // init solution to zeros
            Arr_t::getDevice().axpy(static_cast<value_type>(0.0), rhs_host, static_cast<value_type*>(NULL), 
                    N_size, x_host);
    
            // solve the linear system using gmres
            int solver_ret = linear_solver_gmres_(JacobiImplicitApplyPerVesicle, JacobiImplicitPrecondPerVesicle, 
                    x_host, rhs_host, params_.time_tol, 0, N_size, params_.time_iter_max, 200);

            // copy host to device
            vel_i->getDevice().Memcpy(vel_i->begin(), x_host    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
            axpy(dt_, *vel_i, *vel_i);
            // end of using GMRES Solver
            // upsample vel_i to vel_i_up
            Resample(*vel_i, sht_, sht_upsample_, *wrk1, *wrk2, *vel_i_up);
            
            for(iter_j = cvmap.cbegin(); iter_j != cvmap.cend(); iter_j++)
            {
                // Accumulate LCPMatrix entry j,i, matrix is stored in row order
                value_type val_tmp = AlgebraicDot(*vel_i_up, *iter_j->second);
                lcp_matrix.begin()[(iter_j->first-1)*num_cvs_ + (iter_i->first-1)] += val_tmp;
            }
        }
    }
    
    for(iter_i = cvmap.cbegin(); iter_i != cvmap.cend(); iter_i++)
        delete iter_i->second;
    cvmap.clear();
    
    // release workers
    recycle(f_i);
    recycle(f_i_up);
    recycle(vel_i);
    recycle(vel_i_up);
    recycle(wrk1);
    recycle(wrk2);
    recycle(ten_i);
    
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
GetDx(Vec_t &col_dx, Sca_t &col_tension, const Vec_t &col_f) const
{
    //COUT("begin getdx");
    // clear vec
    col_dx.getDevice().Memset(col_dx.begin(), 0, col_dx.size()*sizeof(value_type));
    col_tension.getDevice().Memset(col_tension.begin(), 0, col_tension.size()*sizeof(value_type));
    
    //mkl_set_num_threads(1);
    //COUT("mkl max threads: "<<mkl_get_max_threads());
    int omp_p = omp_get_max_threads();
    //COUT("omp max threads: "<<omp_p);
    int N_ves = contact_vesicle_list_.size();
    #pragma omp parallel for
    for(int tid=0; tid<omp_p; tid++)
    {
        int tida = ((tid+0)*N_ves)/omp_p;
        int tidb = ((tid+1)*N_ves)/omp_p;
        int omp_id = omp_get_thread_num();
//#pragma omp critical
        //COUT("omp thread id: "<<omp_id<<" out of: "<<omp_get_num_threads());

        // checkout workers
        // get surface resolution we are working on
        SurfContainer* Surf;
        Surf = S_is_[omp_id];
        // checkout vecs, scas
        Vec_t f_i(1, Surf->getPosition().getShOrder());
        Vec_t vel_i(1, Surf->getPosition().getShOrder());
        Sca_t ten_i(1, Surf->getPosition().getShOrder());

        size_t vsz(f_i.size()), tsz(ten_i.size());
        size_t N_size = vsz+tsz;

        for(int i=tida; i<tidb; i++)
        {
            int i_vesicle = contact_vesicle_list_[i];
            
            // set position of current vesicle surface
            f_i.getDevice().Memcpy(f_i.begin(), &(S_.getPosition().begin()[i_vesicle*vsz]),
                    vsz*sizeof(value_type), device_type::MemcpyDeviceToDevice);
            Surf->setPosition(f_i);
            
            // prepare rhs for current vesicle
            f_i.getDevice().Memcpy(f_i.begin(), &(col_f.begin()[i_vesicle*vsz]), 
                    vsz*sizeof(value_type), device_type::MemcpyDeviceToDevice);

            if(MaxAbs(f_i) == 0)
                continue;

            stokesSLPerVesicle(f_i, vel_i, i_vesicle);
            Surf->div(vel_i, ten_i);
            //axpy(static_cast<value_type>(-1.0), ten_i, ten_i);
        
            // copy device type to value_type array to call GMRES
            value_type x_host[N_size], rhs_host[N_size];
        
            // copy to rhs
            vel_i.getDevice().Memcpy(rhs_host    , vel_i.begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
            ten_i.getDevice().Memcpy(rhs_host+vsz, ten_i.begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        
            // init solution to zeros
            Arr_t::getDevice().axpy(static_cast<value_type>(0.0), rhs_host, static_cast<value_type*>(NULL), 
                    N_size, x_host);
            
            // solve the linear system using gmres
            //#pragma omp critical
            //{
                int solver_ret = linear_solver_gmress_[omp_id](JacobiImplicitApplyPerVesicle, JacobiImplicitPrecondPerVesicle, 
                        x_host, rhs_host, params_.time_tol*0.1, 0, N_size, params_.time_iter_max, 200, i_vesicle);
            //}

            // copy host to device
            #pragma omp critical(dx)
            {
                col_dx.getDevice().Memcpy( &(col_dx.begin()[i_vesicle*vsz]), x_host, 
                        vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
                col_tension.getDevice().Memcpy( &(col_tension.begin()[i_vesicle*tsz]), x_host+vsz,
                        tsz * sizeof(value_type), device_type::MemcpyHostToDevice);
            }
        }
    }
    MKL_Free_Buffers();
    //mkl_set_num_threads(omp_p);
    //COUT("mkl max threads: "<<mkl_get_max_threads());
    //COUT("end getdx");
    
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
GetColPos(const Vec_t &xin, std::vector<value_type> &pos_vec, pvfmm::Vector<value_type> &pos_pole) const
{
    // get poles
    const pvfmm::Vector<value_type> pos_pvfmm(xin.size(), (value_type*)xin.begin(), false); 
    pvfmm::Vector<value_type> pos_coef;
    SphericalHarmonics<value_type>::Grid2SHCPerVesicle(pos_pvfmm, params_.sh_order, params_.sh_order, pos_coef);
    SphericalHarmonics<value_type>::SHC2Pole(pos_coef, params_.sh_order, pos_pole);

    if(params_.upsample_freq > params_.sh_order && params_.col_upsample)
    {
        pvfmm::Vector<value_type> pos_up(pos_vec.size(), (value_type*)&pos_vec[0], false);
        // upsample
        SphericalHarmonics<value_type>::SHC2GridPerVesicle(pos_coef, params_.sh_order, params_.upsample_freq, pos_up);
    }
    else
    {
        xin.getDevice().Memcpy(&pos_vec[0], xin.begin(),
                xin.size() * sizeof(value_type),
                xin.getDevice().MemcpyDeviceToHost);
    }
   
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
GetColPosAll(const Vec_t &xin, std::vector<value_type> &pos_vec, pvfmm::Vector<value_type> &pos_pole) const
{
    // get poles
    const pvfmm::Vector<value_type> pos_pvfmm(xin.size(), (value_type*)xin.begin(), false); 
    pvfmm::Vector<value_type> pos_coef;
    SphericalHarmonics<value_type>::Grid2SHC(pos_pvfmm, params_.sh_order, params_.sh_order, pos_coef);
    SphericalHarmonics<value_type>::SHC2Pole(pos_coef, params_.sh_order, pos_pole);

    if(params_.upsample_freq > params_.sh_order && params_.col_upsample)
    {
        pvfmm::Vector<value_type> pos_up(pos_vec.size(), (value_type*)&pos_vec[0], false);
        // upsample
        SphericalHarmonics<value_type>::SHC2Grid(pos_coef, params_.sh_order, params_.upsample_freq, pos_up);
    }
    else
    {
        xin.getDevice().Memcpy(&pos_vec[0], xin.begin(),
                xin.size() * sizeof(value_type),
                xin.getDevice().MemcpyDeviceToHost);
    }
   
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
UpdateVgradInd(int *ind1, int *ind2, int base, size_t length) const
{
    #pragma omp parallel for
    for(size_t i=0; i<length; i++)
    {
        if(ind2[i]>0)
            ind1[i] = ind2[i] + base;
    }
   
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ParallelGetVolumeAndGradient(const Vec_t &X_s, const Vec_t &X_e) const
{
    int myrank, np;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &np);

    // clear variables
    pvfmm::Profile::Tic("ClearVariables", &comm, true);
    sum_num_cvs_ = 0;
    num_cvs_ = 0;
    IV_.clear();
    vgrad_.getDevice().Memset(vgrad_.begin(), 0, sizeof(value_type)*vgrad_.size());
    std::memset(&vgrad_ind_[0], 0, vgrad_ind_.size()*sizeof(vgrad_ind_[0]));
    for(typename std::map<int, Vec_t*>::iterator i=ghost_vgrad_.begin(); i!=ghost_vgrad_.end(); i++)
        delete i->second;
    ghost_vgrad_.clear();
    for(std::map<int, std::vector<int>*>::iterator i=ghost_vgrad_ind_.begin(); i!=ghost_vgrad_ind_.end(); i++)
        delete i->second;
    ghost_vgrad_ind_.clear();
    pvfmm::Profile::Toc();

    // call VesBoundingBox get index pairs, we can sub-divide [X_s, X_e] to do culling
    // vesicle id [0, N-1], N is total number of vesicles globally
    int col_upsample_freq = params_.sh_order;
    if(params_.col_upsample)
    {
        col_upsample_freq = params_.upsample_freq;
    }
    std::vector< std::pair<size_t, size_t> > BBIPairs;
    pvfmm::Profile::Tic("SetBoundingBox", &comm, true);
    if(params_.col_upsample)
        VBBI_->SetVesBoundingBox(X_s, X_e, params_.min_sep_dist, params_.sh_order, col_upsample_freq);
    else
        VBBI_->SetVesBoundingBox(X_s, X_e, params_.min_sep_dist, params_.sh_order, params_.sh_order);
    pvfmm::Profile::Toc();
    VBBI_->GetContactBoundingBoxPair(BBIPairs);
    
    INFO("After getcontactbbpairs.\n");
/*
    for(size_t i=0; i<BBIPairs.size(); i++)
    {
        COUT("pair: "<<BBIPairs[i].first<<", "<<BBIPairs[i].second<<"\n");
    }
*/
    
    pvfmm::Profile::Tic("PrepareData1", &comm, true);
    // send ghost vesicles, receive ghost vesicles
    size_t nv = X_s.getNumSubs();     // number of vesicles per process
    size_t stride = X_s.getStride();    // number of points per vesicle
    size_t stride_up = 2*col_upsample_freq*(col_upsample_freq+1);

    // data to send vesicle id: vid
    pvfmm::Vector<int> sves_cnt(np);
    pvfmm::Vector<int> sves_dsp(np);
    pvfmm::Vector<int> rves_cnt(np);
    pvfmm::Vector<int> rves_dsp(np);
    std::vector<size_t> sves_id;
    std::vector<size_t> rves_id;

    // data to send vesicle coordinate
    pvfmm::Vector<int> sves_coord_cnt(np);
    pvfmm::Vector<int> sves_coord_dsp(np);
    pvfmm::Vector<int> rves_coord_cnt(np);
    pvfmm::Vector<int> rves_coord_dsp(np);

    // get send pairs, send vid vesicle to pid process
    std::vector< std::pair<size_t, size_t> > pid_vid;
    for(size_t i=0; i<BBIPairs.size(); i++)
    {
        if( BBIPairs[i].second < nv*myrank || BBIPairs[i].second >= nv*(myrank+1) )
        {
            // only send smaller vesicle id to process owns larger vesicle id
            // avoid duplicate contact
            if(BBIPairs[i].first < BBIPairs[i].second)
                pid_vid.push_back(std::make_pair(floor(BBIPairs[i].second/nv), BBIPairs[i].first));
        }
    }
    // sort by pid, and remove duplicate
    std::sort(pid_vid.begin(), pid_vid.end());
    pid_vid.erase(std::unique(pid_vid.begin(),pid_vid.end()), pid_vid.end());
    
    // set send count for vid and vesicle coordinate, set ves_id data to send
    sves_cnt.SetZero(); rves_cnt.SetZero();
    sves_coord_cnt.SetZero(); rves_coord_cnt.SetZero();
    sves_id.clear(); rves_id.clear();
    for(size_t i=0; i<pid_vid.size(); i++)
    {
        sves_cnt[pid_vid[i].first]++;
        sves_coord_cnt[pid_vid[i].first] += COORD_DIM*stride;
        sves_id.push_back(pid_vid[i].second);
    }

    // all to all communication for receive count of vesicle id
    MPI_Alltoall(&sves_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &rves_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    // all to all communication for receive count of vesicle coordinate
    MPI_Alltoall(&sves_coord_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &rves_coord_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    // set displacement of sending and receiving for vesicle id
    sves_dsp[0] = 0; pvfmm::omp_par::scan(&sves_cnt[0], &sves_dsp[0], sves_cnt.Dim());
    rves_dsp[0] = 0; pvfmm::omp_par::scan(&rves_cnt[0], &rves_dsp[0], rves_cnt.Dim());

    // set displacement of sending and receiving for vesicle coordinate
    sves_coord_dsp[0] = 0; pvfmm::omp_par::scan(&sves_coord_cnt[0], &sves_coord_dsp[0], sves_coord_cnt.Dim());
    rves_coord_dsp[0] = 0; pvfmm::omp_par::scan(&rves_coord_cnt[0], &rves_coord_dsp[0], rves_coord_cnt.Dim());

    // total send size of vesicle id
    size_t send_size_ves = sves_cnt[np-1] + sves_dsp[np-1];
    // total receive size of vesicle id
    size_t recv_size_ves = rves_cnt[np-1] + rves_dsp[np-1];
    
    ASSERT(sves_id.size() == send_size_ves, "Bad send ves size");
    // init receive id data
    rves_id.resize(recv_size_ves, 0);
    pvfmm::Profile::Toc();
    
    // send and receive ghost vesicles' global id, local_id = global_id - myrank*nv, local_id = [0, nv-1]
    // TODO replace with Mpi_Alltoallv_sparse
    pvfmm::Profile::Tic("SendRecvGhostVesID", &comm, true);
    MPI_Alltoallv(&sves_id[0], &sves_cnt[0], &sves_dsp[0], pvfmm::par::Mpi_datatype<size_t>::value(),
                  &rves_id[0], &rves_cnt[0], &rves_dsp[0], pvfmm::par::Mpi_datatype<size_t>::value(), comm);
    pvfmm::Profile::Toc();

    pvfmm::Profile::Tic("PrepareData2", &comm, true);
    // init variables for sending, receiving vesicle coordinate
    pvfmm::Vector<value_type> send_ves_coord_s, send_ves_coord_e;
    pvfmm::Vector<value_type> recv_ves_coord_s, recv_ves_coord_e;
    send_ves_coord_s.ReInit(send_size_ves*COORD_DIM*stride);
    send_ves_coord_e.ReInit(send_size_ves*COORD_DIM*stride);
    recv_ves_coord_s.ReInit(recv_size_ves*COORD_DIM*stride);
    recv_ves_coord_e.ReInit(recv_size_ves*COORD_DIM*stride);

    // prepare send coord
    for(size_t i=0; i<sves_id.size(); i++)
    {
        size_t local_i = sves_id[i] - myrank*nv;
        ASSERT(local_i >= 0, "Bad local ves id size");
        ASSERT(local_i < nv, "Bad local ves id size");

        for(size_t k=0; k<COORD_DIM; k++)
        {
            for(size_t j=0; j<stride; j++)
            {
                send_ves_coord_s[i*stride*COORD_DIM + k*stride + j] = 
                    X_s.begin()[local_i*stride*COORD_DIM + k*stride + j];
                send_ves_coord_e[i*stride*COORD_DIM + k*stride + j] = 
                    X_e.begin()[local_i*stride*COORD_DIM + k*stride + j];
            }
        }
    }
    pvfmm::Profile::Toc();
    
    // send and receive ghost vesicles' coord
    // TODO replace with Mpi_Alltoallv_sparse
    pvfmm::Profile::Tic("SendRecvGhostVesPos", &comm, true);
    MPI_Alltoallv(&send_ves_coord_s[0], &sves_coord_cnt[0], &sves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(),
                  &recv_ves_coord_s[0], &rves_coord_cnt[0], &rves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(), 
                  comm);    
    MPI_Alltoallv(&send_ves_coord_e[0], &sves_coord_cnt[0], &sves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(),
                  &recv_ves_coord_e[0], &rves_coord_cnt[0], &rves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(), 
                  comm);
    pvfmm::Profile::Toc();
    // recv_ves_coord contains the ghost vesicles' coord, rves_id contains the global vesicle id

    // map for global vesicle id and coordinate
    typedef std::map< int, Vec_t* > GVMAP;
    typename GVMAP::iterator got;
    typename GVMAP::const_iterator iter_i;

    pvfmm::Profile::Tic("AddGhostPos", &comm, true);
    // insert ghost vesilces
    GVMAP ghost_ves_s, ghost_ves_e;
    for(size_t i = 0; i<rves_id.size(); i++)
    {
        got = ghost_ves_s.find(rves_id[i]);
        ASSERT(got == ghost_ves_s.end(), "received vid is not unique?????");
        if(got == ghost_ves_s.end())
        {
            ghost_ves_s.insert( std::make_pair< int, Vec_t* >( rves_id[i], new Vec_t(1, params_.sh_order) ) );
            ghost_ves_e.insert( std::make_pair< int, Vec_t* >( rves_id[i], new Vec_t(1, params_.sh_order) ) );
            //copy recv_ves_coord to ghost_ves[rves_id[i]]
            Arr_t::getDevice().Memcpy( ghost_ves_s[rves_id[i]]->begin(), &recv_ves_coord_s[i*COORD_DIM*stride], 
                    ghost_ves_s[rves_id[i]]->size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
            Arr_t::getDevice().Memcpy( ghost_ves_e[rves_id[i]]->begin(), &recv_ves_coord_e[i*COORD_DIM*stride], 
                    ghost_ves_e[rves_id[i]]->size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);

            // prepare for ghost_vgrad_ and ghost_vgrad_ind_
            ghost_vgrad_.insert( std::make_pair< int, Vec_t* >( rves_id[i], new Vec_t(1, col_upsample_freq) ) );
            Arr_t::getDevice().Memset(ghost_vgrad_[rves_id[i]]->begin(), 0, ghost_vgrad_[rves_id[i]]->size()*sizeof(value_type));
            ghost_vgrad_ind_.insert( std::make_pair< int, std::vector<int>* >( rves_id[i], new std::vector<int>(stride_up*COORD_DIM, 0) ) );
        }
    }
    pvfmm::Profile::Toc();

    // loop all BBIpairs to call CI_pair_ do contact detect
    size_t omp_p = omp_get_max_threads();
    int N_pairs = BBIPairs.size();
    pvfmm::Profile::Tic("BBIPairsContact", &comm, true);
    //COUT("total number of pairs to check: "<<N_pairs<<".");
    int tvs = 2*(2*col_upsample_freq*(col_upsample_freq+1)+2);
    int tfs = 2*(col_upsample_freq*col_upsample_freq*4+4*col_upsample_freq);
    int tes = tfs*3/2;
    pvfmm::Profile::Add_FLOP( ((long long)N_pairs)*(tvs+tfs+tes)*500);
    #pragma omp parallel for
    for(size_t tid=0; tid<omp_p; tid++)
    {
        size_t tida = ((tid+0)*N_pairs)/omp_p;
        size_t tidb = ((tid+1)*N_pairs)/omp_p;
        size_t omp_id = omp_get_thread_num();
        
        // work space variables
        std::vector<value_type> x_s(stride_up*COORD_DIM*2, 0.0);
        std::vector<value_type> x_e(stride_up*COORD_DIM*2, 0.0);
        pvfmm::Vector<value_type> x_s_pole, x_e_pole;
        Vec_t x_pair(2, params_.sh_order);
        Vec_t x_pair_s(2, params_.sh_order);
        Vec_t x_pair_e(2, params_.sh_order);
        std::vector<value_type> vgrad_pair(stride_up*COORD_DIM*2, 0.0);
        std::vector<int> vgrad_pair_ind(stride_up*COORD_DIM*2, 0.0);
        Vec_t vgrad1(1, col_upsample_freq), vgrad2(1, col_upsample_freq);
        ASSERT(vgrad1.size() == stride_up*COORD_DIM, "wrong vgrad1 size");
        typename GVMAP::const_iterator got_i;
        ContactInterface *CI_pair = CI_pairs_[omp_id];
        
        for(size_t i=tida; i<tidb; i++)
        {
            int vid1 = BBIPairs[i].first;
            int vid2 = BBIPairs[i].second;
            // do contact detection first id > second id to remove duplicate contacts
            // this is local vesicle vs ghost vesicle
            if( (vid2 < nv*myrank || vid2 >= nv*(myrank+1)) && (vid1 > vid2) )
            {
                //COUT("checking pair ghost: "<<vid1<<", "<<vid2);
                size_t local_i = vid1 - myrank*nv;
                ASSERT(local_i >= 0, "Bad local ves id size");
                ASSERT(local_i < nv, "Bad local ves id size");
                // copy local vesicle start position
                x_pair_s.getDevice().Memcpy(&(x_pair_s.begin()[0]), &X_s.begin()[local_i*x_pair_s.size()/2],
                        x_pair_s.size()*sizeof(value_type)/2, device_type::MemcpyDeviceToDevice);
                // copy ghost vesicle start position
                got_i = ghost_ves_s.find(vid2);
                ASSERT(got_i != ghost_ves_s.cend(), "missing ghost vesicle?????");
                x_pair_s.getDevice().Memcpy(&(x_pair_s.begin()[x_pair_s.size()/2]), got_i->second->begin(), 
                        x_pair_s.size()*sizeof(value_type)/2, device_type::MemcpyDeviceToDevice);
                // upsample and get poles
                //GetColPos(x_pair, x_s, x_s_pole);

                // copy local vesicle end position
                x_pair_e.getDevice().Memcpy(&(x_pair_e.begin()[0]), &X_e.begin()[local_i*x_pair_e.size()/2],
                        x_pair_e.size()*sizeof(value_type)/2, device_type::MemcpyDeviceToDevice);
                // copy ghost vesicle end position
                got_i = ghost_ves_e.find(vid2);
                ASSERT(got_i != ghost_ves_e.cend(), "missing ghost vesicle?????");
                x_pair_e.getDevice().Memcpy(&(x_pair_e.begin()[x_pair_e.size()/2]), got_i->second->begin(), 
                        x_pair_e.size()*sizeof(value_type)/2, device_type::MemcpyDeviceToDevice);
                // upsample and get poles
                GetColPos(x_pair_e, x_e, x_e_pole);

                //TODO: do we need this?
                GetStartPosPair(x_pair_s, x_pair_e, x_pair);
                GetColPos(x_pair, x_s, x_s_pole);

                // transfer vesicle to periodic box
                if(params_.periodic_length > 0)
                {
                    TransferVesicle(x_s, x_e, x_s_pole, x_e_pole);
                }

                // get contact of vesicle pair
                int num_cvs = 0;
                std::vector<value_type> IV;
                std::vector<value_type> IS_BDV;
                std::memset(&vgrad_pair[0], 0, vgrad_pair.size()*sizeof(vgrad_pair[0]));
                std::memset(&vgrad_pair_ind[0], 0, vgrad_pair_ind.size()*sizeof(vgrad_pair_ind[0]));
                double t_check_start = omp_get_wtime();
                //#pragma omp critical(detect)
                {
                    CI_pair->getVolumeAndGradient(IV, IS_BDV, num_cvs, vgrad_pair, vgrad_pair_ind, x_s, x_e, &x_s_pole[0], &x_e_pole[0],
                            params_.min_sep_dist, params_.periodic_length);
                }
                double t_check_end = omp_get_wtime();
                //COUT("time for checking pair: "<<vid1<<", "<<vid2<<" is: "<<(t_check_end-t_check_start));

                // updates
                if(num_cvs > 0)
                {
                    #pragma omp critical(iv)
                    {
                        // first vesicle local vesicle
                        vgrad1.getDevice().Memcpy(vgrad1.begin(), &vgrad_pair[0], 
                                vgrad1.size()*sizeof(value_type), device_type::MemcpyHostToDevice);
                        vgrad2.getDevice().Memcpy(vgrad2.begin(), &vgrad_.begin()[local_i*vgrad2.size()],
                                vgrad2.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
                        axpy(static_cast<value_type>(1.0), vgrad1, vgrad2, vgrad1);

                        vgrad1.getDevice().Memcpy(&vgrad_.begin()[local_i*vgrad1.size()], vgrad1.begin(),
                                vgrad1.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
                        UpdateVgradInd(&vgrad_ind_[local_i*vgrad1.size()], &vgrad_pair_ind[0], num_cvs_, vgrad1.size());

                        // second vesicle ghost vesicle
                        vgrad1.getDevice().Memcpy(vgrad1.begin(), &vgrad_pair[vgrad1.size()], 
                                vgrad1.size()*sizeof(value_type), device_type::MemcpyHostToDevice);
                        vgrad2.getDevice().Memcpy(vgrad2.begin(), ghost_vgrad_[vid2]->begin(),
                                vgrad2.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
                        axpy(static_cast<value_type>(1.0), vgrad1, vgrad2, vgrad1);

                        vgrad1.getDevice().Memcpy(ghost_vgrad_[vid2]->begin(), vgrad1.begin(),
                                vgrad1.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
                        UpdateVgradInd(&ghost_vgrad_ind_[vid2]->front(), &vgrad_pair_ind[vgrad1.size()], num_cvs_, vgrad1.size());
                        // update number of contact volumes and contact volume vector
                        num_cvs_ += num_cvs;
                        IV_.insert(IV_.end(), IV.begin(), IV.end());
                    }
                }
                //COUT("finish checking pair: "<<vid1<<", "<<vid2);
            }
            else if(vid1 > vid2)
            {
                //COUT("checking pair: "<<vid1<<", "<<vid2);
                size_t local_i = vid1 - myrank*nv;
                ASSERT(local_i >= 0, "Bad local ves id size");
                ASSERT(local_i < nv, "Bad local ves id size");
                // copy local vesicle 1 start position
                x_pair_s.getDevice().Memcpy(&x_pair_s.begin()[0], &X_s.begin()[local_i*x_pair_s.size()/2],
                        x_pair_s.size()*sizeof(value_type)/2, device_type::MemcpyDeviceToDevice);
                local_i = vid2 - myrank*nv;
                ASSERT(local_i >= 0, "Bad local ves id size");
                ASSERT(local_i < nv, "Bad local ves id size");
                // copy local vesicle 2 start position
                x_pair_s.getDevice().Memcpy(&x_pair_s.begin()[x_pair_s.size()/2], &X_s.begin()[local_i*x_pair_s.size()/2], 
                        x_pair_s.size()*sizeof(value_type)/2, device_type::MemcpyDeviceToDevice);
                // upsample and get poles
                //GetColPos(x_pair, x_s, x_s_pole);

                local_i = vid1 - myrank*nv;
                // copy local vesicle 1 end position
                x_pair_e.getDevice().Memcpy(&x_pair_e.begin()[0],               &X_e.begin()[local_i*x_pair_e.size()/2],
                        x_pair_e.size()*sizeof(value_type)/2, device_type::MemcpyDeviceToDevice);
                local_i = vid2 - myrank*nv;
                // copy local vesicle 2 end position
                x_pair_e.getDevice().Memcpy(&x_pair_e.begin()[x_pair_e.size()/2], &X_e.begin()[local_i*x_pair_e.size()/2], 
                        x_pair_e.size()*sizeof(value_type)/2, device_type::MemcpyDeviceToDevice);
                // upsample and get poles
                GetColPos(x_pair_e, x_e, x_e_pole);

                //TODO: do we need this?
                GetStartPosPair(x_pair_s, x_pair_e, x_pair);
                GetColPos(x_pair, x_s, x_s_pole);

                // transfer vesicles to periodic box
                if(params_.periodic_length > 0)
                {
                    TransferVesicle(x_s, x_e, x_s_pole, x_e_pole);
                }

                // get contact of vesicle pair
                int num_cvs = 0;
                std::vector<value_type> IV;
                std::vector<value_type> IS_BDV;
                std::memset(&vgrad_pair[0], 0, vgrad_pair.size()*sizeof(vgrad_pair[0]));
                std::memset(&vgrad_pair_ind[0], 0, vgrad_pair_ind.size()*sizeof(vgrad_pair_ind[0]));
                double t_check_start = omp_get_wtime();
                //#pragma omp critical(detect)
                {
                    CI_pair->getVolumeAndGradient(IV, IS_BDV, num_cvs, vgrad_pair, vgrad_pair_ind, x_s, x_e, &x_s_pole[0], &x_e_pole[0],
                            params_.min_sep_dist, params_.periodic_length);
                }
                double t_check_end = omp_get_wtime();
                //COUT("time for checking pair: "<<vid1<<", "<<vid2<<" is: "<<(t_check_end-t_check_start));

                // updates
                if(num_cvs > 0)
                {
                    #pragma omp critical(iv)
                    {
                        // first vesicle local vesicle
                        local_i = vid1 - myrank*nv;
                        vgrad1.getDevice().Memcpy(vgrad1.begin(), &vgrad_pair[0], 
                                vgrad1.size()*sizeof(value_type), device_type::MemcpyHostToDevice);
                        vgrad2.getDevice().Memcpy(vgrad2.begin(), &vgrad_.begin()[local_i*vgrad2.size()],
                                vgrad2.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
                        axpy(static_cast<value_type>(1.0), vgrad1, vgrad2, vgrad1);

                        vgrad1.getDevice().Memcpy(&vgrad_.begin()[local_i*vgrad1.size()], vgrad1.begin(),
                                vgrad1.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
                        UpdateVgradInd(&vgrad_ind_[local_i*vgrad1.size()], &vgrad_pair_ind[0], num_cvs_, vgrad1.size());

                        // second vesicle local vesicle
                        local_i = vid2 - myrank*nv;
                        vgrad1.getDevice().Memcpy(vgrad1.begin(), &vgrad_pair[vgrad1.size()], 
                                vgrad1.size()*sizeof(value_type), device_type::MemcpyHostToDevice);
                        vgrad2.getDevice().Memcpy(vgrad2.begin(), &vgrad_.begin()[local_i*vgrad2.size()],
                                vgrad2.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
                        axpy(static_cast<value_type>(1.0), vgrad1, vgrad2, vgrad1);

                        vgrad1.getDevice().Memcpy(&vgrad_.begin()[local_i*vgrad1.size()], vgrad1.begin(),
                                vgrad1.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
                        UpdateVgradInd(&vgrad_ind_[local_i*vgrad1.size()], &vgrad_pair_ind[vgrad1.size()], num_cvs_, vgrad1.size());
                        num_cvs_ += num_cvs;
                        IV_.insert(IV_.end(), IV.begin(), IV.end());
                    }
                }
                //COUT("finish checking pair: "<<vid1<<", "<<vid2);
            }
        }
    }
    pvfmm::Profile::Toc();
    
    pvfmm::Profile::Tic("UpdateVGID", &comm, true);
    // update vgrad_ind to global vgrad_ind
    int cv_id_disp;
    int num_cvs_myrank = num_cvs_;
    // scan displacement of cvid to update cvid to global cvid
    MPI_Scan(&num_cvs_myrank, &cv_id_disp, 1, MPI_INT, MPI_SUM, comm);
    cv_id_disp = cv_id_disp - num_cvs_myrank;
    // scan global number of cvs
    MPI_Allreduce(&num_cvs_myrank, &sum_num_cvs_, 1, MPI_INT, MPI_SUM, comm);
    if(num_cvs_myrank > 0)
    {
        // update vgrad_ind_
        UpdateVgradInd(&vgrad_ind_[0], &vgrad_ind_[0], cv_id_disp, vgrad_ind_.size());
        // update ghost_vgrad_ind_
        for(std::map<int, std::vector<int>*>::iterator i=ghost_vgrad_ind_.begin(); i!=ghost_vgrad_ind_.end(); i++)
        {
            UpdateVgradInd(&i->second->front(), &i->second->front(), cv_id_disp, i->second->size());
        }
    }
    pvfmm::Profile::Toc();

    pvfmm::Profile::Tic("PrepareData3", &comm, true);
    // MPI merge global vgrad_ and vgrad_ind_,
    sves_cnt.SetZero(); rves_cnt.SetZero();
    sves_coord_cnt.SetZero(); rves_coord_cnt.SetZero();
    sves_id.clear(); rves_id.clear();
    // TODO: don't send zero components ghost vgrad and ghost vgrad_ind!
    for(iter_i=ghost_vgrad_.begin(); iter_i!=ghost_vgrad_.end(); iter_i++)
    {
        sves_cnt[floor(iter_i->first/nv)]++; 
        sves_coord_cnt[floor(iter_i->first/nv)] += COORD_DIM*stride_up;
        sves_id.push_back(iter_i->first);
    }
    MPI_Alltoall(&sves_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &rves_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    MPI_Alltoall(&sves_coord_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &rves_coord_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    sves_dsp[0] = 0; pvfmm::omp_par::scan(&sves_cnt[0], &sves_dsp[0], sves_cnt.Dim());
    rves_dsp[0] = 0; pvfmm::omp_par::scan(&rves_cnt[0], &rves_dsp[0], rves_cnt.Dim());

    sves_coord_dsp[0] = 0; pvfmm::omp_par::scan(&sves_coord_cnt[0], &sves_coord_dsp[0], sves_coord_cnt.Dim());
    rves_coord_dsp[0] = 0; pvfmm::omp_par::scan(&rves_coord_cnt[0], &rves_coord_dsp[0], rves_coord_cnt.Dim());

    send_size_ves = sves_cnt[np-1] + sves_dsp[np-1];
    recv_size_ves = rves_cnt[np-1] + rves_dsp[np-1];
    
    ASSERT(sves_id.size() == send_size_ves, "Bad send ves size");
    rves_id.resize(recv_size_ves, 0);
    pvfmm::Profile::Toc();
    
    // send and receive ghost vesicles' global id, local_id = global_id - myrank*nv
    // TODO replace with Mpi_Alltoallv_sparse
    pvfmm::Profile::Tic("SendRecvGhostVesIDv", &comm, true);
    MPI_Alltoallv(&sves_id[0], &sves_cnt[0], &sves_dsp[0], pvfmm::par::Mpi_datatype<size_t>::value(),
                  &rves_id[0], &rves_cnt[0], &rves_dsp[0], pvfmm::par::Mpi_datatype<size_t>::value(), comm);
    pvfmm::Profile::Toc();

    pvfmm::Profile::Tic("PrepareData4", &comm, true);
    send_ves_coord_s.ReInit(send_size_ves*COORD_DIM*stride_up);
    recv_ves_coord_s.ReInit(recv_size_ves*COORD_DIM*stride_up);

    pvfmm::Vector<int> send_ves_ind, recv_ves_ind;
    send_ves_ind.ReInit(send_size_ves*COORD_DIM*stride_up);
    recv_ves_ind.ReInit(recv_size_ves*COORD_DIM*stride_up);
    // prepare send coord
    size_t i = 0;
    for(iter_i=ghost_vgrad_.begin(); iter_i!=ghost_vgrad_.end(); iter_i++)
    {
        for(size_t k=0; k<COORD_DIM; k++)
        {
            for(size_t j=0; j<stride_up; j++)
            {
                send_ves_coord_s[i*stride_up*COORD_DIM + k*stride_up + j] = 
                    ghost_vgrad_[iter_i->first]->begin()[k*stride_up + j];
                send_ves_ind[i*stride_up*COORD_DIM + k*stride_up + j] = 
                    (*ghost_vgrad_ind_[iter_i->first])[k*stride_up + j];
            }
        }
        i++;
    }
    pvfmm::Profile::Toc();
    
    // send and receive ghost vesicles' coord
    // TODO replace with Mpi_Alltoallv_sparse
    pvfmm::Profile::Tic("SendRecvGhostVesPosv", &comm, true);
    MPI_Alltoallv(&send_ves_coord_s[0], &sves_coord_cnt[0], &sves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(),
                  &recv_ves_coord_s[0], &rves_coord_cnt[0], &rves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(), 
                  comm);    
    MPI_Alltoallv(&send_ves_ind[0], &sves_coord_cnt[0], &sves_coord_dsp[0], pvfmm::par::Mpi_datatype<int>::value(),
                  &recv_ves_ind[0], &rves_coord_cnt[0], &rves_coord_dsp[0], pvfmm::par::Mpi_datatype<int>::value(), 
                  comm);
    pvfmm::Profile::Toc();
    
    pvfmm::Profile::Tic("AddGhostVgrad", &comm, true);
    // recv_ves_coord contains the vgrad coord, rves_id contains the global vesicle id
    // add vgrad to local vgrad, and the same for vgra_ind
    Vec_t vgrad1(1, col_upsample_freq), vgrad2(1, col_upsample_freq);
    for(i=0; i<rves_id.size(); i++)
    {
        size_t local_i = rves_id[i] - myrank*nv;
        ASSERT(local_i >= 0, "Bad local ves id size");
        ASSERT(local_i < nv, "Bad local ves id size");

        vgrad1.getDevice().Memcpy(vgrad1.begin(), &recv_ves_coord_s[i*vgrad1.size()], 
                vgrad1.size()*sizeof(value_type), device_type::MemcpyHostToDevice);
        vgrad2.getDevice().Memcpy(vgrad2.begin(), &vgrad_.begin()[local_i*vgrad2.size()],
                vgrad2.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
        axpy(static_cast<value_type>(1.0), vgrad1, vgrad2, vgrad1);

        vgrad1.getDevice().Memcpy(&vgrad_.begin()[local_i*vgrad1.size()], vgrad1.begin(),
                vgrad1.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
        UpdateVgradInd(&vgrad_ind_[local_i*vgrad1.size()], &recv_ves_ind[i*vgrad1.size()], 0, vgrad1.size());
    }

    // clean memory
    for(iter_i=ghost_ves_s.begin(); iter_i!=ghost_ves_s.end();iter_i++)
        delete iter_i->second;
    for(iter_i=ghost_ves_e.begin(); iter_i!=ghost_ves_e.end();iter_i++)
        delete iter_i->second;
    ghost_ves_s.clear();
    ghost_ves_e.clear();

    // TODO clear ghost_vgrad_ and ghost_vgrad_ind_
    /*
    for(typename std::map<int, Vec_t*>::iterator i=ghost_vgrad_.begin(); i!=ghost_vgrad_.end(); i++)
        delete i->second;
    ghost_vgrad_.clear();
    for(std::map<int, std::vector<int>*>::iterator i=ghost_vgrad_ind_.begin(); i!=ghost_vgrad_ind_.end(); i++)
        delete i->second;
    ghost_vgrad_ind_.clear();
    */
    pvfmm::Profile::Toc();

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ParallelGetVolumeAndGradientWithBoundary(const Vec_t &X_s, const Vec_t &X_e, 
        const value_type *pos_bd, const int num_points_per_patch_1d, const int num_patches) const
{
    // TODO: test bounding box intersection call that generates correct BOXI-BOX pairs.
    // get BOXI-BOX pairs from bounding box intersection call, BOXI and BOX can be boundary or vesicle.
    // first index BOXI is always incident boundary or incident vesicle, 
    // BOX can be ghost or incident boundary, vesicle.
    // pairs are ordered, both i-j and j-i shoule be generated from bounding box intersection call
    //
    // "I" means incident vesicle or boundary, "G" means ghost vesicle or boundary(boundary<=>patch)
    //
    // split BOXI-BOX to B-VI, B-VG, V-BI, V-BG, V-VI, V-VG pairs
    //
    // first phase, do BI+BG with VI contact, form boundary contact list and vesicle contact list
    // B-VI, add B to boundary list for contact, add VI to vesicle list for contact
    // B-VG, send B to VG process
    // V-BI, add V to vesicle list and BI to boundary list for contact, may duplicate with B-VI
    // V-BG, receive BG and add BG to boundary list for contact
    // V-VI, add V and VI to vesicle list for contact
    // no communication is needed after contact detection since all vesicles in contact list are incident vesicle.
    //
    // second phase, do VI-VG contact, loop over all V-VG pairs if V>VG do contact
    // V-VG, send V to VG's process if V<VG; on ther other side, receive VG if V>VG and add VG to vesicle contact list
    // communication is needed after contact detection, send contact information to VG's process
    //
    //

    int myrank, np;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &np);

    // clear contact volume variables
    pvfmm::Profile::Tic("ClearVariables", &comm, true);
    sum_num_cvs_ = 0;
    num_cvs_ = 0;
    IV_.clear();
    IS_BDV_.clear();
    vgrad_.getDevice().Memset(vgrad_.begin(), 0, sizeof(value_type)*vgrad_.size());
    std::memset(&vgrad_ind_[0], 0, vgrad_ind_.size()*sizeof(vgrad_ind_[0]));
    for(typename std::map<int, Vec_t*>::iterator i=ghost_vgrad_.begin(); i!=ghost_vgrad_.end(); i++)
        delete i->second;
    ghost_vgrad_.clear();
    for(std::map<int, std::vector<int>*>::iterator i=ghost_vgrad_ind_.begin(); i!=ghost_vgrad_ind_.end(); i++)
        delete i->second;
    ghost_vgrad_ind_.clear();
    pvfmm::Profile::Toc();
    // end of clear contact volume variables

    // call BDVesBoundingBox get index pairs. 
    // index order on each process first vesicle then boundary
    int col_upsample_freq = params_.sh_order;
    if(params_.col_upsample)
    {
        col_upsample_freq = params_.upsample_freq;
    }
    std::vector< std::pair<size_t, size_t> > BBIPairs;
    pvfmm::Profile::Tic("SetBoundingBox", &comm, true);
    BDVBBI_->SetBDVesBoundingBox(X_s, X_e, params_.min_sep_dist, X_s.getShOrder(), col_upsample_freq,
            pos_bd, num_points_per_patch_1d, num_patches);
    INFO("After setboundingbox.");
    pvfmm::Profile::Toc();
    pvfmm::Profile::Tic("GetBoundingBox", &comm, true);
    BDVBBI_->GetContactBoundingBoxPair(BBIPairs);
    pvfmm::Profile::Toc();
    INFO("After getcontactbbpairs.");
    // end of BDVesBoundingBox pairs

    // split pairs into six lists of pairs
    // now assums nv is same across all processes, also num_patches is same across all processes
    size_t nv = X_s.getNumSubs();   // number of vesicle on this process
    size_t npa = (size_t)num_patches;   // number of patches on this process
    size_t nall = nv+npa;   // number of vesicles+patches on this process
    std::vector< std::pair<size_t, size_t> > B_VI, B_VG, V_BI, V_BG, V_VI, V_VG;
    for(size_t i_pair=0; i_pair<BBIPairs.size(); i_pair++)
    {
        size_t ind1 = BBIPairs[i_pair].first;   //always incident boundary or vesicle
        size_t ind2 = BBIPairs[i_pair].second;  //can be incident or ghost boundary, ghost vesicle
        //COUT("pairs: "<<ind1<<", "<<ind2<<", nall: "<<nall);
        bool first_bd = false; 
        bool second_bd = false;
        bool second_ghost = false;

        size_t ind1_mod = ind1%nall;
        size_t ind2_mod = ind2%nall;
        if(ind1_mod>=nv)
            first_bd=true;
        if(ind2_mod>=nv)
            second_bd=true;

        assert( ind1>=nall*myrank && ind1<nall*(myrank+1) );

        if( ind2<nall*myrank || ind2>=nall*(myrank+1) )
            second_ghost = true;

        if(first_bd)
        {
            if(second_bd)
                continue;

            if(second_ghost)
                B_VG.push_back( std::make_pair(ind1, ind2) );
            else
                B_VI.push_back( std::make_pair(ind1, ind2) );
        }
        else
        {
            if(second_bd)
            {
                if(second_ghost)
                    V_BG.push_back( std::make_pair(ind1, ind2) );
                else
                    V_BI.push_back( std::make_pair(ind1, ind2) );
            }
            else
            {
                if(second_ghost)
                    V_VG.push_back( std::make_pair(ind1, ind2) );
                else
                    V_VI.push_back( std::make_pair(ind1, ind2) );
            }
        }
    }
    /*
    COUT("B_VI");
    for(int i=0; i<B_VI.size(); i++)
    {
        COUT("B_VI: "<<B_VI[i].first<<", "<<B_VI[i].second);
    }
    COUT("V_VI");
    for(int i=0; i<V_VI.size(); i++)
    {
        COUT("V_VI: "<<V_VI[i].first<<", "<<V_VI[i].second);
    }
    COUT("V_BI");
    for(int i=0; i<V_BI.size(); i++)
    {
        COUT("V_BI: "<<V_BI[i].first<<", "<<V_BI[i].second);
    }
    COUT(B_VG.size()<<", "<<V_VG.size()<<", "<<V_BG.size());
    */
    // end of split pairs
    
    // send and receive ghost boundary patches
    pvfmm::Profile::Tic("BDdataComm", &comm, true);
    // now assums nv is same across all processes, also num_patches is same across all processes

    // vectors for send/receive boundary patch ids
    pvfmm::Vector<int> sbd_cnt(np);
    pvfmm::Vector<int> sbd_dsp(np);
    pvfmm::Vector<int> rbd_cnt(np);
    pvfmm::Vector<int> rbd_dsp(np);
    std::vector<size_t> sbd_id;
    std::vector<size_t> rbd_id;

    // vectors to send/receive boundary coordinate
    pvfmm::Vector<int> sbd_coord_cnt(np);
    pvfmm::Vector<int> sbd_coord_dsp(np);
    pvfmm::Vector<int> rbd_coord_cnt(np);
    pvfmm::Vector<int> rbd_coord_dsp(np);

    // get send pairs, send bid patch to pid process
    std::vector< std::pair<size_t, size_t> > pid_bid;
    for(size_t i=0; i<B_VG.size(); i++)
    {
        pid_bid.push_back( std::make_pair( floor(B_VG[i].second/nall), B_VG[i].first ) );
    }
    // sort by pid, and remove duplicate
    std::sort(pid_bid.begin(), pid_bid.end());
    pid_bid.erase(std::unique(pid_bid.begin(),pid_bid.end()), pid_bid.end());

    // set send count for bid and boundary coordinate, set bd_id data to send
    sbd_cnt.SetZero(); rbd_cnt.SetZero();
    sbd_coord_cnt.SetZero(); rbd_coord_cnt.SetZero();
    sbd_id.clear(); rbd_id.clear();
    size_t bd_stride = num_points_per_patch_1d*num_points_per_patch_1d*COORD_DIM;
    for(size_t i=0; i<pid_bid.size(); i++)
    {
        size_t pid = pid_bid[i].first;
        sbd_cnt[pid]++;
        sbd_coord_cnt[pid] += bd_stride;
        sbd_id.push_back(pid_bid[i].second);
    }

    // all to all communication for receive count of bd id
    MPI_Alltoall(&sbd_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &rbd_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    // all to all communication for receive count of bd coordinate
    MPI_Alltoall(&sbd_coord_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &rbd_coord_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    // set displacement of sending and receiving count of bd id
    sbd_dsp[0] = 0; pvfmm::omp_par::scan(&sbd_cnt[0], &sbd_dsp[0], sbd_cnt.Dim());
    rbd_dsp[0] = 0; pvfmm::omp_par::scan(&rbd_cnt[0], &rbd_dsp[0], rbd_cnt.Dim());

    // set displacement of sending and receiving count of bd coordinate
    sbd_coord_dsp[0] = 0; pvfmm::omp_par::scan(&sbd_coord_cnt[0], &sbd_coord_dsp[0], sbd_coord_cnt.Dim());
    rbd_coord_dsp[0] = 0; pvfmm::omp_par::scan(&rbd_coord_cnt[0], &rbd_coord_dsp[0], rbd_coord_cnt.Dim());

    // total send size of bd id
    size_t send_size_bd = sbd_cnt[np-1] + sbd_dsp[np-1];
    // total receive size of bd id
    size_t recv_size_bd = rbd_cnt[np-1] + rbd_dsp[np-1];

    ASSERT(sbd_id.size() == send_size_bd, "Bad send bd size");
    // init receive bd id data
    rbd_id.resize(recv_size_bd, 0);

    // send and receive bd's global id, local_id = global_id - myrank*nall - nv, local_id = [0, num_patches-1]
    MPI_Alltoallv(&sbd_id[0], &sbd_cnt[0], &sbd_dsp[0], pvfmm::par::Mpi_datatype<size_t>::value(),
                  &rbd_id[0], &rbd_cnt[0], &rbd_dsp[0], pvfmm::par::Mpi_datatype<size_t>::value(), comm);

    // init variables for sending/receiving boundary coordinate
    pvfmm::Vector<value_type> send_bd_coord, recv_bd_coord;
    send_bd_coord.ReInit(send_size_bd*bd_stride);
    recv_bd_coord.ReInit(recv_size_bd*bd_stride);

    // prepare send coord
    for(size_t i=0; i<sbd_id.size(); i++)
    {
        size_t local_i = sbd_id[i] - myrank*nall - nv;
        ASSERT(local_i >= 0, "Bad local bd id");
        ASSERT(local_i<num_patches, "Bad local bd id");
        std::memcpy(&send_bd_coord[i*bd_stride], &pos_bd[local_i*bd_stride], sizeof(value_type)*bd_stride);
    }

    // send and received ghost bd coord
    MPI_Alltoallv(&send_bd_coord[0], &sbd_coord_cnt[0], &sbd_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(),
                  &recv_bd_coord[0], &rbd_coord_cnt[0], &rbd_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(),
                  comm);
    // end of send and receive ghost boundary patches
    pvfmm::Profile::Toc();
    // Do contact (VI)-(VI+Ball)


    // send and receive ghost vesicles
    pvfmm::Profile::Tic("VESdataComm", &comm, true);
    // now assums nv is same across all processes, also num_patches is same across all processes
    //
    //
    //
    // data to send vesicle id: vid
    pvfmm::Vector<int> sves_cnt(np);
    pvfmm::Vector<int> sves_dsp(np);
    pvfmm::Vector<int> rves_cnt(np);
    pvfmm::Vector<int> rves_dsp(np);
    std::vector<size_t> sves_id;
    std::vector<size_t> rves_id;

    // data to send vesicle coordinate
    pvfmm::Vector<int> sves_coord_cnt(np);
    pvfmm::Vector<int> sves_coord_dsp(np);
    pvfmm::Vector<int> rves_coord_cnt(np);
    pvfmm::Vector<int> rves_coord_dsp(np);

    // get send pairs, send vid vesicle to pid process
    std::vector< std::pair<size_t, size_t> > pid_vid;
    for(size_t i=0; i<V_VG.size(); i++)
    {
        int v_r_id = V_VG[i].first;
        int v_g_id = V_VG[i].second;
        int send_flag = abs(v_r_id - v_g_id) % 2;
        //if( V_VG[i].first < V_VG[i].second)
        if( (send_flag==0 && V_VG[i].first < V_VG[i].second) || (send_flag==1 && V_VG[i].first > V_VG[i].second) )
        {
            // only send smaller vesicle id to process owns larger vesicle id
            // avoid duplicate contact
            pid_vid.push_back(std::make_pair(floor(V_VG[i].second/nall), V_VG[i].first));
        }
    }
    // sort by pid, and remove duplicate
    std::sort(pid_vid.begin(), pid_vid.end());
    pid_vid.erase(std::unique(pid_vid.begin(),pid_vid.end()), pid_vid.end());
    
    // set send count for vid and vesicle coordinate, set ves_id data to send
    sves_cnt.SetZero(); rves_cnt.SetZero();
    sves_coord_cnt.SetZero(); rves_coord_cnt.SetZero();
    sves_id.clear(); rves_id.clear();
    size_t stride = X_s.getStride();
    size_t ves_stride = stride*COORD_DIM;
    for(size_t i=0; i<pid_vid.size(); i++)
    {
        sves_cnt[pid_vid[i].first]++;
        sves_coord_cnt[pid_vid[i].first] += ves_stride;
        sves_id.push_back(pid_vid[i].second);
    }

    // all to all communication for receive count of vesicle id
    MPI_Alltoall(&sves_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &rves_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    // all to all communication for receive count of vesicle coordinate
    MPI_Alltoall(&sves_coord_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &rves_coord_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    // set displacement of sending and receiving for vesicle id
    sves_dsp[0] = 0; pvfmm::omp_par::scan(&sves_cnt[0], &sves_dsp[0], sves_cnt.Dim());
    rves_dsp[0] = 0; pvfmm::omp_par::scan(&rves_cnt[0], &rves_dsp[0], rves_cnt.Dim());

    // set displacement of sending and receiving for vesicle coordinate
    sves_coord_dsp[0] = 0; pvfmm::omp_par::scan(&sves_coord_cnt[0], &sves_coord_dsp[0], sves_coord_cnt.Dim());
    rves_coord_dsp[0] = 0; pvfmm::omp_par::scan(&rves_coord_cnt[0], &rves_coord_dsp[0], rves_coord_cnt.Dim());

    // total send size of vesicle id
    size_t send_size_ves = sves_cnt[np-1] + sves_dsp[np-1];
    // total receive size of vesicle id
    size_t recv_size_ves = rves_cnt[np-1] + rves_dsp[np-1];
    
    ASSERT(sves_id.size() == send_size_ves, "Bad send ves size");
    // init receive id data
    rves_id.resize(recv_size_ves, 0);
    
    // send and receive ghost vesicles' global id, local_id = global_id - myrank*nall, local_id = [0, nv-1]
    // TODO replace with Mpi_Alltoallv_sparse
    MPI_Alltoallv(&sves_id[0], &sves_cnt[0], &sves_dsp[0], pvfmm::par::Mpi_datatype<size_t>::value(),
                  &rves_id[0], &rves_cnt[0], &rves_dsp[0], pvfmm::par::Mpi_datatype<size_t>::value(), comm);

    // init variables for sending, receiving vesicle coordinate
    pvfmm::Vector<value_type> send_ves_coord_s, send_ves_coord_e;
    pvfmm::Vector<value_type> recv_ves_coord_s, recv_ves_coord_e;
    send_ves_coord_s.ReInit(send_size_ves*ves_stride);
    send_ves_coord_e.ReInit(send_size_ves*ves_stride);
    recv_ves_coord_s.ReInit(recv_size_ves*ves_stride);
    recv_ves_coord_e.ReInit(recv_size_ves*ves_stride);

    // prepare send coord
    for(size_t i=0; i<sves_id.size(); i++)
    {
        size_t local_i = sves_id[i] - myrank*nall;
        ASSERT(local_i >= 0, "Bad local ves id size");
        ASSERT(local_i < nv, "Bad local ves id size");
        
        std::memcpy(&send_ves_coord_s[i*ves_stride], &X_s.begin()[local_i*ves_stride], sizeof(value_type)*ves_stride);
        std::memcpy(&send_ves_coord_e[i*ves_stride], &X_e.begin()[local_i*ves_stride], sizeof(value_type)*ves_stride);

        /*
        for(size_t k=0; k<COORD_DIM; k++)
        {
            for(size_t j=0; j<stride; j++)
            {
                send_ves_coord_s[i*stride*COORD_DIM + k*stride + j] = 
                    X_s.begin()[local_i*stride*COORD_DIM + k*stride + j];
                send_ves_coord_e[i*stride*COORD_DIM + k*stride + j] = 
                    X_e.begin()[local_i*stride*COORD_DIM + k*stride + j];
            }
        }
        */
    }
    
    // send and receive ghost vesicles' coord
    // TODO replace with Mpi_Alltoallv_sparse
    MPI_Alltoallv(&send_ves_coord_s[0], &sves_coord_cnt[0], &sves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(),
                  &recv_ves_coord_s[0], &rves_coord_cnt[0], &rves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(), 
                  comm);    
    MPI_Alltoallv(&send_ves_coord_e[0], &sves_coord_cnt[0], &sves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(),
                  &recv_ves_coord_e[0], &rves_coord_cnt[0], &rves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(), 
                  comm);
    //
    //
    // end of send and receive ghost vesicles
    pvfmm::Profile::Toc();
    // Do contact VI-VG
    
    pvfmm::Profile::Tic("ContactCall", &comm, true);
    //local contact detection
    // resident vesicle and boundary patch list
    std::vector<size_t> v_resident;
    std::vector<size_t> bd_resident;
    for(size_t i=0; i<V_VI.size(); i++)
    {
        if(V_VI[i].first < V_VI[i].second)
            continue;

        int local_v_i1 = V_VI[i].first - myrank*nall;
        int local_v_i2 = V_VI[i].second - myrank*nall;

        assert(local_v_i1>=0);
        assert(local_v_i1<nv);
        assert(local_v_i2>=0);
        assert(local_v_i2<nv);

        v_resident.push_back(local_v_i1);
        v_resident.push_back(local_v_i2);
    }
    for(size_t i=0; i<V_VG.size(); i++)
    {
        int v_r_id = V_VG[i].first;
        int v_g_id = V_VG[i].second;
        int send_flag = abs(v_r_id - v_g_id) % 2;

        //if(V_VG[i].first < V_VG[i].second)
        if( (send_flag==0 && V_VG[i].first < V_VG[i].second) || (send_flag==1 && V_VG[i].first > V_VG[i].second) )
            continue;

        int local_v_i = V_VG[i].first - myrank*nall;
        assert(local_v_i>=0);
        assert(local_v_i<nv);
        v_resident.push_back(local_v_i);
    }
    for(size_t i=0; i<V_BG.size(); i++)
    {
        int local_v_i = V_BG[i].first - myrank*nall;
        assert(local_v_i>=0);
        assert(local_v_i<nv);
        v_resident.push_back(local_v_i);
    }
    for(size_t i=0; i<V_BI.size(); i++)
    {
        int local_v_i = V_BI[i].first - myrank*nall;
        int local_bd_i = V_BI[i].second - myrank*nall - nv;
        assert(local_v_i>=0);
        assert(local_v_i<nv);
        assert(local_bd_i>=0);
        assert(local_bd_i<num_patches);
        v_resident.push_back(local_v_i);
        bd_resident.push_back(local_bd_i);
    }
    std::sort(v_resident.begin(), v_resident.end());
    std::sort(bd_resident.begin(), bd_resident.end());
    v_resident.erase(std::unique(v_resident.begin(), v_resident.end()), v_resident.end());
    bd_resident.erase(std::unique(bd_resident.begin(), bd_resident.end()), bd_resident.end());
    
    // copy data
    static int c_nv_prev = 0;
    static int c_nbd_prev = 0;
    int c_nv = v_resident.size() + recv_size_ves;
    int c_nbd = bd_resident.size() + recv_size_bd;
    Vec_t v_pos_s(c_nv, X_s.getShOrder());
    Vec_t v_pos_e(c_nv, X_e.getShOrder());
    size_t ves_stride_up = 2*col_upsample_freq*(col_upsample_freq+1)*COORD_DIM;
    std::vector<value_type> v_pos_up_s(c_nv*ves_stride_up);
    std::vector<value_type> v_pos_up_e(c_nv*ves_stride_up);
    pvfmm::Vector<value_type> v_pos_up_s_pole, v_pos_up_e_pole;
    pvfmm::Vector<value_type> bd_coord;
    bd_coord.ReInit(c_nbd*bd_stride);
    std::vector<value_type> vgrad(v_pos_up_s.size(), 0);
    std::vector<int> vgrad_ind(v_pos_up_s.size(), 0);

    if(v_resident.size()!=0)
    {
        //COUT("v_resident: "<<v_resident.size());
        // vesicles
        #pragma omp parallel for
        for(size_t i=0; i<v_resident.size(); i++)
        {
            size_t local_v_i = v_resident[i];
            v_pos_s.getDevice().Memcpy(&v_pos_s.begin()[i*ves_stride], &X_s.begin()[local_v_i*ves_stride], 
                    ves_stride*sizeof(value_type), device_type::MemcpyHostToDevice);
            v_pos_e.getDevice().Memcpy(&v_pos_e.begin()[i*ves_stride], &X_e.begin()[local_v_i*ves_stride],
                    ves_stride*sizeof(value_type), device_type::MemcpyHostToDevice);
        }
        v_pos_s.getDevice().Memcpy(&v_pos_s.begin()[v_resident.size()*ves_stride], &recv_ves_coord_s[0],
                    recv_size_ves*ves_stride*sizeof(value_type), device_type::MemcpyHostToDevice);
        v_pos_e.getDevice().Memcpy(&v_pos_e.begin()[v_resident.size()*ves_stride], &recv_ves_coord_e[0],
                    recv_size_ves*ves_stride*sizeof(value_type), device_type::MemcpyHostToDevice);
        GetColPosAll(v_pos_s, v_pos_up_s, v_pos_up_s_pole);
        GetColPosAll(v_pos_e, v_pos_up_e, v_pos_up_e_pole);
        // boundary pathces
        #pragma omp parallel for
        for(size_t i=0; i<bd_resident.size(); i++)
        {
            size_t local_bd_i = bd_resident[i];
            //COUT("local_bd_i: "<<local_bd_i);
            memcpy(&bd_coord[0]+i*bd_stride, &pos_bd[local_bd_i*bd_stride], bd_stride*sizeof(value_type));
        }
        //COUT("ghost_bd_size: "<<recv_size_bd);
        memcpy(&bd_coord[0]+bd_resident.size()*bd_stride, &recv_bd_coord[0], recv_size_bd*bd_stride*sizeof(value_type));

        // call contact detection
        if(c_nv != c_nv_prev || c_nbd != c_nbd_prev)
        {
            if(CI_ves_bd_ != nullptr)
            {
                //COUT("deleting CI_ves_bd: "<<CI_ves_bd_);
                delete CI_ves_bd_;
            }
            CI_ves_bd_ = new ContactInterfaceBoundary();
            //COUT(v_pos_up_s.size()<<", "<<v_pos_up_s_pole.Dim()<<", "<<bd_coord.Dim());
            CI_ves_bd_->generateMesh(v_pos_up_s, &v_pos_up_s_pole[0], col_upsample_freq, c_nv, 
                    &bd_coord[0], num_points_per_patch_1d, c_nbd);
            CI_ves_bd_->init(SURF_SIMPLE);
        }
        //COUT("c_nv: "<<c_nv<<", c_nbd: "<<c_nbd<<", nv resident: "<<v_resident.size());
        CI_ves_bd_->getVolumeAndGradient(IV_, IS_BDV_, num_cvs_, vgrad, vgrad_ind, 
                v_pos_up_s, v_pos_up_e, &v_pos_up_s_pole[0], &v_pos_up_e_pole[0],
                params_.min_sep_dist, &bd_coord[0], v_resident.size(), params_.periodic_length);
        //COUT("after contact3d call");
        c_nv_prev = c_nv; c_nbd_prev = c_nbd;
        // add flops
        int tvs = 2*(2*col_upsample_freq*(col_upsample_freq+1)+2);
        int tfs = 2*(col_upsample_freq*col_upsample_freq*4+4*col_upsample_freq);
        int tes = tfs*3/2;
        int bvs = num_points_per_patch_1d*num_points_per_patch_1d;
        int bfs = 2*bvs;
        int bes = bfs*3/2;
        //COUT("c_nv: "<<c_nv<<", c_nbd: "<<c_nbd);
        //COUT(((long long)c_nv) * ((long long)(tvs+tfs+tes))*500);
        //COUT(((long long)c_nbd) * ((long long)(bvs+bfs+bes))*500);
        pvfmm::Profile::Add_FLOP( ((long long)c_nv) * ((long long)(tvs+tfs+tes))*500);
        pvfmm::Profile::Add_FLOP( ((long long)c_nbd) * ((long long)(bvs+bfs+bes))*500);
    }
    else
    {
        //COUT("no v_resident: "<<c_nv);
        assert(c_nv == 0);
    }
    pvfmm::Profile::Toc();
    
    // update vgrad_ind to global vgrad_ind
    pvfmm::Profile::Tic("UpdateVGID", &comm, true);
    //COUT("updatevgid");
    int cv_id_disp;
    int num_cvs_myrank = num_cvs_;
    // scan displacement of cvid to update cvid to global cvid
    MPI_Scan(&num_cvs_myrank, &cv_id_disp, 1, MPI_INT, MPI_SUM, comm);
    cv_id_disp = cv_id_disp - num_cvs_myrank;
    // scan global number of cvs
    MPI_Allreduce(&num_cvs_myrank, &sum_num_cvs_, 1, MPI_INT, MPI_SUM, comm);
    if(num_cvs_myrank > 0)
    {
        // update vgrad_ind_
        UpdateVgradInd(&vgrad_ind[0], &vgrad_ind[0], cv_id_disp, vgrad_ind.size());
    }
    pvfmm::Profile::Toc();
    //COUT("done updatevgid");

    //copy vgrad and vgrad_ind to vgrad_ and vgrad_ind_
    for(size_t i=0; i<v_resident.size(); i++)
    {
        size_t local_v_i = v_resident[i];
        vgrad_.getDevice().Memcpy(&vgrad_.begin()[local_v_i*ves_stride_up], &vgrad[i*ves_stride_up], 
                ves_stride_up*sizeof(value_type), device_type::MemcpyHostToDevice);
    
        memcpy(&vgrad_ind_[local_v_i*ves_stride_up], &vgrad_ind[i*ves_stride_up], 
                ves_stride_up*sizeof(int));
    }
    // copy vgrad_ind to ghost_vgrad_ind_ for parallelcvjacobiantrans
    for(size_t i = 0; i<rves_id.size(); i++)
    {
        int rves_pid = floor(rves_id[i]/nall);
        int ves_id_tmp = rves_id[i] - rves_pid*nall + rves_pid*nv;
        if(ghost_vgrad_ind_.find(ves_id_tmp) == ghost_vgrad_ind_.end())
        {
            std::vector<int>* tmp_pointer = new std::vector<int>(ves_stride_up);
            memcpy(&tmp_pointer->front(), &vgrad_ind[v_resident.size()*ves_stride_up+i*ves_stride_up], 
                    ves_stride_up*sizeof(int));
            ghost_vgrad_ind_.insert( std::pair< int, std::vector<int>* >(ves_id_tmp, tmp_pointer) );
        }
    }


    // Send back ghost vesicle vgrad and vgrad_ind
    //COUT("send back ghost");
    send_ves_coord_s.ReInit(send_size_ves*ves_stride_up);
    pvfmm::Vector<int> send_ves_ind;
    send_ves_ind.ReInit(send_size_ves*ves_stride_up);
    for(size_t i=0; i<np; i++)
    {
        rves_coord_cnt[i] = rves_coord_cnt[i]/ves_stride*ves_stride_up;
        rves_coord_dsp[i] = rves_coord_dsp[i]/ves_stride*ves_stride_up;
        sves_coord_cnt[i] = sves_coord_cnt[i]/ves_stride*ves_stride_up;
        sves_coord_dsp[i] = sves_coord_dsp[i]/ves_stride*ves_stride_up;
    }
    size_t ghost_start_index = v_resident.size()*ves_stride_up;
    MPI_Alltoallv(&vgrad[ghost_start_index], &rves_coord_cnt[0], &rves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(), 
                 &send_ves_coord_s[0],       &sves_coord_cnt[0], &sves_coord_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(), 
                 comm);
    MPI_Alltoallv(&vgrad_ind[ghost_start_index], &rves_coord_cnt[0], &rves_coord_dsp[0], pvfmm::par::Mpi_datatype<int>::value(), 
                 &send_ves_ind[0],               &sves_coord_cnt[0], &sves_coord_dsp[0], pvfmm::par::Mpi_datatype<int>::value(), 
                 comm);
    //COUT("done send back ghost");

    // update vgrad and vgrad_ind 
    //COUT("final update");
    Vec_t vgrad1(1, col_upsample_freq), vgrad2(1, col_upsample_freq);
    for(size_t i=0; i<sves_id.size(); i++)
    {
        size_t local_v_i = sves_id[i] - myrank*nall;
        assert(local_v_i >= 0);
        assert(local_v_i < nv);

        vgrad1.getDevice().Memcpy(vgrad1.begin(), &send_ves_coord_s[i*ves_stride_up],
                ves_stride_up*sizeof(value_type), device_type::MemcpyHostToDevice);
        vgrad2.getDevice().Memcpy(vgrad2.begin(), &vgrad_.begin()[local_v_i*ves_stride_up],
                ves_stride_up*sizeof(value_type), device_type::MemcpyDeviceToDevice);
        axpy(static_cast<value_type>(1.0), vgrad1, vgrad2, vgrad1);

        vgrad1.getDevice().Memcpy(&vgrad_.begin()[local_v_i*ves_stride_up], vgrad1.begin(), 
                ves_stride_up*sizeof(value_type), device_type::MemcpyDeviceToDevice);
        UpdateVgradInd(&vgrad_ind_[local_v_i*ves_stride_up], &send_ves_ind[i*ves_stride_up], 0, ves_stride_up);
    }
    //COUT("done final update");
    
    /*
    int np_up = 2*params_.sh_order*(params_.sh_order+1);
    if(params_.col_upsample)
    {
        std::cout<<"col upsample true\n";
        np_up = 2*params_.upsample_freq*(params_.upsample_freq+1);
    }
    static std::vector<value_type> pos_s(np_up*COORD_DIM*nv_, 0.0);
    static std::vector<value_type> pos_e(np_up*COORD_DIM*nv_, 0.0);
    static std::vector<value_type> vGrad(np_up*COORD_DIM*nv_, 0.0);
    static pvfmm::Vector<value_type> pos_s_pole, pos_e_pole;
        
    GetColPosAll(X_s, pos_s, pos_s_pole);
    GetColPosAll(X_e, pos_e, pos_e_pole);

    sum_num_cvs_ = 0;
    num_cvs_ = 0;
    IV_.clear();
    IS_BDV_.clear();
    vgrad_.getDevice().Memset(vgrad_.begin(), 0, sizeof(value_type)*vgrad_.size());
    std::memset(&vgrad_ind_[0], 0, vgrad_ind_.size()*sizeof(vgrad_ind_[0]));

    if(params_.periodic_length > 0)
    {
        TransferVesicle(pos_s, pos_e, pos_s_pole, pos_e_pole);
    }
    CI_bd_.getVolumeAndGradient(IV_, IS_BDV_, num_cvs_, vGrad, vgrad_ind_, pos_s, pos_e, &pos_s_pole[0], &pos_e_pole[0], 
            params_.min_sep_dist, fixed_bd->tri_vertices, params_.periodic_length);
    sum_num_cvs_ = num_cvs_;
    vgrad_.getDevice().Memcpy(vgrad_.begin(), &vGrad.front(), 
            vgrad_.size() * sizeof(value_type),
            vgrad_.getDevice().MemcpyHostToDevice);
    */
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ParallelFormLCPMatrixSparse(std::map<std::pair<size_t, size_t>, value_type> &lcp_matrix) const
{
    //mkl_set_num_threads(1);
    //COUT("mkl max threads: "<<mkl_get_max_threads());
    // max number of threads
    int omp_p = omp_get_max_threads();
    //COUT("omp max threads: "<<omp_p);

    // form lcp_matrix as in FormLCPMatrixSparse
    typedef boost::unordered_map< int, Vec_t* > CVMAP;
    CVMAP cvmaps[omp_p];
    
    // col upsample freq
    int col_upsample_freq = params_.sh_order;
    if(params_.col_upsample)
        col_upsample_freq = params_.upsample_freq;

    // checkout workers
    // TODO: make it static?
    std::vector<Vec_t*> f_is;
    std::vector<Vec_t*> f_i_ups;
    std::vector<Vec_t*> vel_is;
    std::vector<Vec_t*> vel_i_ups;
    std::vector<Vec_t*> wrk1s;
    std::vector<Vec_t*> wrk2s;
    std::vector<Vec_t*> i_vgrads;
    std::vector<Sca_t*> ten_is;
    std::vector<int> i_vgrad_inds[omp_p];
    for(int i=0; i<omp_p; i++)
    {
        f_is.push_back(new Vec_t(1, params_.sh_order));
        f_i_ups.push_back(new Vec_t(1, col_upsample_freq));
        vel_is.push_back(new Vec_t(1, params_.sh_order));
        vel_i_ups.push_back(new Vec_t(1, col_upsample_freq));
        wrk1s.push_back(new Vec_t(1, col_upsample_freq));
        wrk2s.push_back(new Vec_t(1, col_upsample_freq));
        i_vgrads.push_back(new Vec_t(1, col_upsample_freq));
        ten_is.push_back(new Sca_t(1, params_.sh_order));
        //i_vgrad_inds[i] = std::vector<int>(i_vgrads[i]->size(), 0);
        i_vgrad_inds[i].resize(i_vgrads[i]->size(), 0);
    }

    // clear lcp_matrix
    lcp_matrix.clear();
    // number of vesicles
    int nves = nv_;
    // sh_order collision detection working on
    int sh_order = col_upsample_freq;
    size_t ncount = i_vgrad_inds[0].size();
    contact_vesicle_list_.clear();
    MPI_Comm comm = MPI_COMM_WORLD;
    pvfmm::Profile::Tic("FormLCPMatrixLocal", &comm, true);
    //COUT("before form lcp local");
    #pragma omp parallel for schedule(dynamic)
    for(int i_vesicle = 0; i_vesicle < nves; i_vesicle++)
    {
        // working thread id
        int omp_id = omp_get_thread_num();
//#pragma omp critical
        //COUT("omp thread id: "<<omp_id<<" out of: "<<omp_get_num_threads());

        // workers reference
        CVMAP &cvmap = cvmaps[omp_id];
        Vec_t &f_i = *f_is[omp_id];
        Vec_t &f_i_up = *f_i_ups[omp_id];
        Vec_t &vel_i = *vel_is[omp_id];
        Vec_t &vel_i_up = *vel_i_ups[omp_id];
        Vec_t &wrk1 = *wrk1s[omp_id];
        Vec_t &wrk2 = *wrk2s[omp_id];
        Vec_t &i_vgrad = *i_vgrads[omp_id];
        Sca_t &ten_i = *ten_is[omp_id];
        std::vector<int> &i_vgrad_ind = i_vgrad_inds[omp_id];
        // get surface resolution we are working on
        SurfContainer* Surf;
        Surf = S_is_[omp_id];

        // lcp_matrix iterator
        typename std::map<std::pair<size_t, size_t>, value_type>::iterator got_lcp_matrix_ij;
        
        // iterator for vgrad
        value_type* iter_vgrad;

        // vgrad index
        int vgrad_index = 0;
        
        // iterators for cvmap
        typename CVMAP::iterator got;
        typename CVMAP::const_iterator iter_i, iter_j;

        bool has_contact = false;

        // set position of current vesicle surface
        f_i.getDevice().Memcpy(f_i.begin(), &(S_.getPosition().begin()[i_vesicle*f_i.size()]),
                f_i.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
        Surf->setPosition(f_i);
        
        // copy the i_th vesicle's part in vgrad_
        iter_vgrad = vgrad_.begin();
        i_vgrad.getDevice().Memcpy( i_vgrad.begin(), &(iter_vgrad[i_vesicle*i_vgrad.size()]), 
                i_vgrad.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice );

        // copy the i_th vesicle's part in vgrad_ind_
        std::memcpy(&(i_vgrad_ind[0]), &(vgrad_ind_[i_vesicle*i_vgrad.size()]), ncount*sizeof(int));
        
        // form cvmap
        for(got = cvmap.begin(); got != cvmap.end(); got++)
            delete got->second;
        cvmap.clear();
        for(size_t icount = 0; icount < ncount; icount++)
        {
            vgrad_index = i_vgrad_ind[icount];
            if(vgrad_index > 0)
            {   
                has_contact = true;
                got = cvmap.find(vgrad_index);
                if( got != cvmap.end() )
                {
                    got->second->begin()[icount] = i_vgrad.begin()[icount];
                }
                else
                {
                   cvmap.insert( std::make_pair( (int)vgrad_index, new Vec_t(1, sh_order) ) );
                   axpy(static_cast<value_type>(0.0), i_vgrad, *cvmap[vgrad_index]);
                   cvmap[vgrad_index]->begin()[icount] = i_vgrad.begin()[icount];
                }
            }
        }

        #pragma omp critical(vlist)
        {
            if(has_contact)
                contact_vesicle_list_.push_back(i_vesicle);
        }

        // iterate cvmap
        for(iter_i = cvmap.cbegin(); iter_i != cvmap.cend(); iter_i++)
        {
            if(params_.col_upsample)
            {
                // filter high freq
                //sht_filter_high(*iter_i->second, f_i_up, &sht_upsample_, params_.rep_exponent);
                sht_filter_high(*iter_i->second, f_i_up, Surf->sht_resample_, params_.rep_exponent);
                // downsample f_i_up to f_i
                //Resample(f_i_up, sht_upsample_, sht_, wrk1, wrk2, f_i);
                Resample(f_i_up, *Surf->sht_resample_, Surf->sht_, wrk1, wrk2, f_i);
            }
            else
                //sht_filter_high(*iter_i->second, f_i, &sht_, params_.rep_exponent);
                sht_filter_high(*iter_i->second, f_i, &Surf->sht_, params_.rep_exponent);
            // prepare rhs for current vesicle
            stokesSLPerVesicle(f_i, vel_i, i_vesicle);
            Surf->div(vel_i, ten_i);
            //axpy(static_cast<value_type>(-1.0), ten_i, ten_i);

            // using GMRES Solver solve for velocity due to the i_th row of contact volume Jacobian
            size_t vsz(f_i.size()), tsz(ten_i.size());
            size_t N_size = vsz+tsz;

            // copy device type to value_type array to call GMRES
            value_type x_host[N_size], rhs_host[N_size];
        
            // copy to rhs
            vel_i.getDevice().Memcpy(rhs_host    , vel_i.begin(), vsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
            ten_i.getDevice().Memcpy(rhs_host+vsz, ten_i.begin(), tsz * sizeof(value_type), device_type::MemcpyDeviceToHost);
        
            // init solution to zeros
            Arr_t::getDevice().axpy(static_cast<value_type>(0.0), rhs_host, static_cast<value_type*>(NULL), 
                    N_size, x_host);
    
            // solve the linear system using gmres
            //#pragma omp critical
            //{
                int solver_ret = linear_solver_gmress_[omp_id](JacobiImplicitApplyPerVesicle, JacobiImplicitPrecondPerVesicle, 
                        x_host, rhs_host, params_.time_tol*0.1, 0, N_size, params_.time_iter_max, 200, i_vesicle);
            //}

            // copy host to device
            vel_i.getDevice().Memcpy(vel_i.begin(), x_host    , vsz * sizeof(value_type), device_type::MemcpyHostToDevice);
            axpy(dt_, vel_i, vel_i);
            // end of using GMRES Solver
            
            // upsample vel_i to vel_i_up
            if(params_.col_upsample)
                //Resample(vel_i, sht_, sht_upsample_, wrk1, wrk2, vel_i_up);
                Resample(vel_i, Surf->sht_, *Surf->sht_resample_, wrk1, wrk2, vel_i_up);
            else
                axpy(static_cast<value_type>(1.0), vel_i, vel_i_up);
            
            for(iter_j = cvmap.cbegin(); iter_j != cvmap.cend(); iter_j++)
            {
                // Accumulate LCPMatrix entry j,i, matrix is stored in row order
                value_type val_tmp = AlgebraicDot(vel_i_up, *iter_j->second);
                #pragma omp critical(lcp)
                {
                    got_lcp_matrix_ij = lcp_matrix.find(std::pair<size_t, size_t>(iter_j->first, iter_i->first));
                    if(got_lcp_matrix_ij != lcp_matrix.end())
                        got_lcp_matrix_ij->second += val_tmp;
                    else
                        lcp_matrix.insert(std::make_pair(std::pair<size_t, size_t>(iter_j->first, iter_i->first), val_tmp));
                }

            }
        }
    }
    //COUT("after form lcp local");
    MKL_Free_Buffers();
    //mkl_set_num_threads(omp_p);
    //COUT("mkl max threads: "<<mkl_get_max_threads());
    //#pragma omp parallel for
    for(int i=0; i<omp_p; i++)
    {
        CVMAP &cvmap = cvmaps[i];
        // clear cvmap
        typename CVMAP::iterator got;
        for(got = cvmap.begin(); got != cvmap.end(); got++)
            delete got->second;
        cvmap.clear();
        
        delete f_is[i];
        delete f_i_ups[i];
        delete vel_is[i];
        delete vel_i_ups[i];
        delete wrk1s[i];
        delete wrk2s[i];
        delete i_vgrads[i];
        delete ten_is[i];
        i_vgrad_inds[i].clear();
    }
    pvfmm::Profile::Toc();
    
    pvfmm::Profile::Tic("FormLCPMatrixGlobal", &comm, true);
    typename std::map<std::pair<size_t, size_t>, value_type>::iterator got_lcp_matrix_ij;
    // MPI communication lcp_matrix
    // send and receive lcp_matrix(i, j) so that local lcp_matrix contains row block
    int myrank, np;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &np);
    
    // number of cvs of all processes
    pvfmm::Vector<size_t> world_num_cvs(np);
    size_t num_cvs_myrank = num_cvs_;
    MPI_Allgather(&num_cvs_myrank,   1, pvfmm::par::Mpi_datatype<size_t>::value(),
                  &world_num_cvs[0], 1, pvfmm::par::Mpi_datatype<size_t>::value(), comm);

    pvfmm::Vector<size_t> cvid_dsp(np);
    cvid_dsp[0]=0; pvfmm::omp_par::scan(&world_num_cvs[0], &cvid_dsp[0], world_num_cvs.Dim());
    // cv_id of current process is (cv_id_offset, cv_id_offset+num_cvs_]

    // data to send values in lcp_matrix
    pvfmm::Vector<int> s_value_cnt(np); s_value_cnt.SetZero();
    pvfmm::Vector<int> s_value_dsp(np); s_value_dsp.SetZero();
    pvfmm::Vector<int> r_value_cnt(np); r_value_cnt.SetZero();
    pvfmm::Vector<int> r_value_dsp(np); r_value_dsp.SetZero();
    std::vector<value_type> s_value; s_value.clear();
    std::vector<value_type> r_value; r_value.clear();
    
    // data to send entry index i,j in lcp_matrix
    pvfmm::Vector<int> s_ind_cnt(np); s_ind_cnt.SetZero();
    pvfmm::Vector<int> s_ind_dsp(np); s_ind_dsp.SetZero();
    pvfmm::Vector<int> r_ind_cnt(np); r_ind_cnt.SetZero();
    pvfmm::Vector<int> r_ind_dsp(np); r_ind_dsp.SetZero();
    std::vector<size_t> s_ind; s_ind.clear();
    std::vector<size_t> r_ind; r_ind.clear();

    // lcp_matrix should be sorted by got_lcp_matrix_ij->first.first
    for(got_lcp_matrix_ij = lcp_matrix.begin(); got_lcp_matrix_ij != lcp_matrix.end(); got_lcp_matrix_ij++)
    {
        size_t ind_i = got_lcp_matrix_ij->first.first;
        size_t ind_j = got_lcp_matrix_ij->first.second;
        value_type value = got_lcp_matrix_ij->second;
        int pid = std::lower_bound(&cvid_dsp[0], &cvid_dsp[0]+np, ind_i) - &cvid_dsp[0] - 1;
        ASSERT(pid >= 0, "wrong pid"); ASSERT(pid < np, "wrong pid");
        ASSERT(ind_i>cvid_dsp[pid], "wrong ind_i"); ASSERT(ind_i<=(cvid_dsp[pid]+world_num_cvs[pid]), "wrong ind_i");
        if(pid!=myrank)
        {
            s_value.push_back(value);
            s_ind.push_back(ind_i); s_ind.push_back(ind_j);
            
            s_value_cnt[pid]++;
            s_ind_cnt[pid] += 2;
        }
    }
    // send and receive
    MPI_Alltoall(&s_value_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &r_value_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);
    MPI_Alltoall(&s_ind_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &r_ind_cnt[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    s_value_dsp[0]=0; pvfmm::omp_par::scan(&s_value_cnt[0], &s_value_dsp[0], s_value_cnt.Dim());
    r_value_dsp[0]=0; pvfmm::omp_par::scan(&r_value_cnt[0], &r_value_dsp[0], r_value_cnt.Dim());
    s_ind_dsp[0]=0; pvfmm::omp_par::scan(&s_ind_cnt[0], &s_ind_dsp[0], s_ind_cnt.Dim());
    r_ind_dsp[0]=0; pvfmm::omp_par::scan(&r_ind_cnt[0], &r_ind_dsp[0], r_ind_cnt.Dim());

    size_t recv_size = r_value_cnt[np-1] + r_value_dsp[np-1];
    r_value.resize(recv_size, 0);
    r_ind.resize(2*recv_size, 0);

    // TODO replace with Mpi_Alltoallv_sparse
    MPI_Alltoallv(&s_value[0], &s_value_cnt[0], &s_value_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(),
                  &r_value[0], &r_value_cnt[0], &r_value_dsp[0], pvfmm::par::Mpi_datatype<value_type>::value(), comm);
    MPI_Alltoallv(&s_ind[0], &s_ind_cnt[0], &s_ind_dsp[0], pvfmm::par::Mpi_datatype<size_t>::value(), 
                  &r_ind[0], &r_ind_cnt[0], &r_ind_dsp[0], pvfmm::par::Mpi_datatype<size_t>::value(), comm);
    
    // print lcp matrix portion before delete and add
    //COUT("size of lcp matrix before delete: "<<parallel_lcp_matrix_.size());
/*
    for(got_lcp_matrix_ij = parallel_lcp_matrix_.begin(); got_lcp_matrix_ij!=parallel_lcp_matrix_.end(); got_lcp_matrix_ij++)
    {
        COUT("entry("<<got_lcp_matrix_ij->first.first<<", "<<got_lcp_matrix_ij->first.second<<") = "<<
                got_lcp_matrix_ij->second);
    }
*/

    // add received data to local lcp_matrix
    for(size_t i=0; i<r_value.size();i++)
    {
        ASSERT(r_ind[2*i]>cvid_dsp[myrank], "received wrong lcp entry");
        ASSERT(r_ind[2*i]<=(cvid_dsp[myrank]+num_cvs_myrank), "received wrong lcp entry");

        got_lcp_matrix_ij = lcp_matrix.find(std::pair<size_t, size_t>(r_ind[2*i], r_ind[2*i+1]));
        if(got_lcp_matrix_ij != lcp_matrix.end())
            got_lcp_matrix_ij->second += r_value[i];
        else
            lcp_matrix.insert(std::make_pair(std::pair<size_t, size_t>(r_ind[2*i], r_ind[2*i+1]), r_value[i]));
    }
    
    // sort lcp_matrix and delete all enties sent out
    ASSERT(s_value.size()*2 == s_ind.size(), "send value index size don't match");
    for(size_t i=0; i<s_value.size();i++)
        lcp_matrix.erase(std::make_pair(s_ind[2*i],s_ind[2*i+1]));
    // end of MPI communication lcp_matrix
    pvfmm::Profile::Toc();
    
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ParallelSolveLCPSmall(Arr_t &lambda, const Arr_t &cvs) const
{
    /*
     * lcp_flag = 1, preprocessing
     * lcp_flag = 2, iterating
     * lcp_flag = 3, relative error termination
     * lcp_flag = 4, absolute error termination
     * lcp_flag = 5, stagnation
     * lcp_flag = 6, local minima
     * lcp_flag = 7, nondescent
     * lcp_flag = 8, maxlimit iters
     */
    int lcp_flag = 1;
    int lcp_n = num_cvs_;
    int lcp_max_iter = 100;
    
    // lcp parameters
    value_type lcp_eps = 1e-16;
    value_type lcp_h = 1e-7;
    value_type lcp_alpha = 0.5;
    value_type lcp_beta = 0.001;
    value_type lcp_gamma = 1e-28;
    value_type lcp_rho = lcp_eps;

    // setup
    std::vector<value_type> lcp_convergence(lcp_max_iter, 0.0);
    value_type lcp_err = 1e+16;
    int lcp_iter = 1;

    // arrs for calculation
    Arr_t lambda_mat_vec(lcp_n);
    Arr_t dlambda(lcp_n);
    Arr_t lcp_y(lcp_n);
    Arr_t lcp_phi(lcp_n);

    // init variables
    Arr_t::getDevice().Memset(lambda.begin(), 0, sizeof(value_type)*lambda.size());
    Arr_t::getDevice().Memset(lambda_mat_vec.begin(), 0, sizeof(value_type)*lambda_mat_vec.size());
    Arr_t::getDevice().Memset(dlambda.begin(), 0, sizeof(value_type)*dlambda.size());
    Arr_t::getDevice().Memset(lcp_y.begin(), 0, sizeof(value_type)*lcp_y.size());
    Arr_t::getDevice().Memset(lcp_phi.begin(), 0, sizeof(value_type)*lcp_phi.size());
    
    int iter(params_.time_iter_max);
    value_type relres(params_.time_tol);
    
    value_type lcp_old_err;
    lcp_flag = 2;
    PA_.clear();
    PA_.resize(num_cvs_, 1);

    // MPI communication
    // send receive lambda
    ghost_lambda_.clear();
    int myrank, np;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &np);
    pvfmm::Profile::Tic("SolveLCPLComm", &comm, true);
    
    // number of cvs of all processes
    // TODO: could remember world_num_cvs(np) and cvid_dsp(np) to save some MPI_allgather
    pvfmm::Vector<size_t> world_num_cvs(np);
    size_t num_cvs_myrank = num_cvs_;
    MPI_Allgather(&num_cvs_myrank,   1, pvfmm::par::Mpi_datatype<size_t>::value(),
                  &world_num_cvs[0], 1, pvfmm::par::Mpi_datatype<size_t>::value(), comm);

    // calculate cvid displacement for each process
    pvfmm::Vector<size_t> cvid_dsp(np);
    cvid_dsp[0]=0; pvfmm::omp_par::scan(&world_num_cvs[0], &cvid_dsp[0], world_num_cvs.Dim());
    cvid_dsp_ = cvid_dsp[myrank];
    ASSERT(sum_num_cvs_==(cvid_dsp[np-1]+world_num_cvs[np-1]), "wrong sum_num_cvs_");
    // cv_id of current process is (cvid_dsp, cvid_dsp+num_cvs_]
    
    // prepare all to all communication for lambdas
    s_ind_cnt_.SetZero(); s_ind_dsp_.SetZero(); r_ind_cnt_.SetZero(); r_ind_dsp_.SetZero();
    s_ind_.clear(); r_ind_.clear();
    
    typename std::map<std::pair<size_t, size_t>, value_type>::iterator got_lcp_matrix_ij;
    // get send pairs, send cvid to pid
    std::vector< std::pair<size_t, size_t> > pid_cvid;
    for(got_lcp_matrix_ij = parallel_lcp_matrix_.begin(); got_lcp_matrix_ij != parallel_lcp_matrix_.end(); got_lcp_matrix_ij++)
    {
        size_t ind_i = got_lcp_matrix_ij->first.first;
        size_t ind_j = got_lcp_matrix_ij->first.second;
        if(ind_j>cvid_dsp_ && ind_j<=(cvid_dsp_+num_cvs_) )
            continue;
        int pid = std::lower_bound(&cvid_dsp[0], &cvid_dsp[0]+np, ind_j) - &cvid_dsp[0] - 1;
        ASSERT(pid >= 0, "wrong pid"); ASSERT(pid < np, "wrong pid");
        ASSERT(ind_j>cvid_dsp[pid], "bad pid for ind_j"); ASSERT(ind_j<=(cvid_dsp[pid]+world_num_cvs[pid]), "bad pid for ind_j");
        ASSERT(pid!=myrank, "wrong pid!!!");
        pid_cvid.push_back(std::make_pair((size_t)pid, (size_t)ind_i));
    }
    // sort by pid, and remove duplicate
    std::sort(pid_cvid.begin(), pid_cvid.end());
    pid_cvid.erase(std::unique(pid_cvid.begin(),pid_cvid.end()), pid_cvid.end());
            
    // set send count for vid and vesicle coordinate, set ves_id data to send
    for(size_t i=0; i<pid_cvid.size(); i++)
    {
        s_ind_cnt_[pid_cvid[i].first]++;
        s_ind_.push_back(pid_cvid[i].second);
    }

    // send and receive
    MPI_Alltoall(&s_ind_cnt_[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &r_ind_cnt_[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    s_ind_dsp_[0]=0; pvfmm::omp_par::scan(&s_ind_cnt_[0], &s_ind_dsp_[0], s_ind_cnt_.Dim());
    r_ind_dsp_[0]=0; pvfmm::omp_par::scan(&r_ind_cnt_[0], &r_ind_dsp_[0], r_ind_cnt_.Dim());

    size_t recv_size = r_ind_cnt_[np-1] + r_ind_dsp_[np-1];
    r_ind_.resize(recv_size, 0); 
    s_value_.resize(s_ind_.size(), 0);
    r_value_.resize(recv_size, 0);

    // TODO replace with Mpi_Alltoallv_sparse
    MPI_Alltoallv(&s_ind_[0], &s_ind_cnt_[0], &s_ind_dsp_[0], pvfmm::par::Mpi_datatype<size_t>::value(), 
                  &r_ind_[0], &r_ind_cnt_[0], &r_ind_dsp_[0], pvfmm::par::Mpi_datatype<size_t>::value(), comm);
    // end of MPI communication
    pvfmm::Profile::Toc();
    
    pvfmm::Profile::Tic("SolveLCPNewtons", &comm, true);
    while(lcp_iter < lcp_max_iter)
    {
        pvfmm::Profile::Tic("SolveLCPOneNewton", &comm, true);
        // configure linear system solver
        CHK(ConfigureLCPSolver());
        typename PVec_t::iterator pvec_i(NULL);
        typename PVec_t::size_type p_sz;

        // init PA_
        PA_.clear();
        PA_.resize(num_cvs_, 1);

        Arr_t::getDevice().Memcpy(lambda_mat_vec.begin(), lambda.begin(), 
                num_cvs_*sizeof(value_type), device_type::MemcpyDeviceToDevice);

        ParallelLCPMatvec(lambda_mat_vec);
        
        Arr_t::getDevice().axpy(static_cast<value_type>(1.0), lambda_mat_vec.begin(), cvs.begin(), 
                num_cvs_, lcp_y.begin());
       
        minmap(lcp_y, lambda, lcp_phi);

        Arr_t::getDevice().axpy(static_cast<value_type>(0.0), dlambda.begin(), static_cast<value_type*>(NULL), 
                dlambda.size(), dlambda.begin());
        Arr_t::getDevice().axpy(static_cast<value_type>(-1.0), lcp_phi.begin(), static_cast<value_type*>(NULL), 
                lcp_phi.size(), lcp_phi.begin());
       
        // assemble initial
        CHK(parallel_u_->GetArray(pvec_i, p_sz));
        ASSERT(num_cvs_ == p_sz, "num_cvs_ not equal petsc size");
        Arr_t::getDevice().Memcpy(pvec_i, dlambda.begin(), p_sz*sizeof(value_type), device_type::MemcpyDeviceToHost);
        CHK(parallel_u_->RestoreArray(pvec_i));
        // assemble rhs
        CHK(parallel_rhs_->GetArray(pvec_i, p_sz));
        ASSERT(num_cvs_ == p_sz, "num_cvs_ not equal petsc size");
        Arr_t::getDevice().Memcpy(pvec_i, lcp_phi.begin(), p_sz*sizeof(value_type), device_type::MemcpyDeviceToHost);
        CHK(parallel_rhs_->RestoreArray(pvec_i));

        // error
        lcp_old_err = lcp_err;
        parallel_rhs_->Norm(lcp_err);
        lcp_err = 0.5*lcp_err*lcp_err;

        //INFO("parallel lcp small Newtown iter: "<<lcp_iter<<". -- err: "<<lcp_err<<" -- relative err: "
        //        <<fabs(lcp_err - lcp_old_err)/fabs(lcp_old_err) );
        //INFO("parallel lambda small: "<<lambda);

        // relative stopping criteria
        if(fabs(lcp_err - lcp_old_err)/fabs(lcp_old_err) < 1e-6)
        {
            lcp_flag = 3;
            pvfmm::Profile::Toc();
            break;
        }
        
        // absolute stopping criteria
        if(lcp_err < 1e-21)
        {
            lcp_flag =4;
            pvfmm::Profile::Toc();
            break;
        }
                
        pvfmm::Profile::Tic("SolveLCPLiSys", &comm, true);
        // solve linear system
        Error_t err_lcp = lcp_parallel_linear_solver_->Solve(parallel_rhs_, parallel_u_);
        pvfmm::Profile::Toc();
        
        // copy back lambda
        CHK(parallel_u_->GetArray(pvec_i, p_sz));
        Arr_t::getDevice().Memcpy(dlambda.begin(), pvec_i, p_sz*sizeof(value_type), device_type::MemcpyDeviceToHost);
        CHK(parallel_u_->RestoreArray(pvec_i));
        
        // view
        lcp_parallel_linear_solver_->ViewReport();

        // TODO: the gradient of meric function \nabla\theta is lcp_phi^T*lcp_matrix, 
        // which we can't get with matrix free version lcp solver, either form lcp matrix
        // explicitly or do some approximate \nabla\theta calculation.
        // So lcp_flag 6,7 is not tested, and we use Newton's method withou line search for now.
        // (Line search requires \nabla\theta)

        value_type dlambda_norm;
        parallel_u_->Norm(dlambda_norm);
        if(dlambda_norm < lcp_eps)
        {
            lcp_flag = 5;
            pvfmm::Profile::Toc();
            break;
            // could use dlambda = -nabla_theta
        }
        
        // TODO: test for whether dropped into a local minima flag 6
        // TODO: test for sufficient descent direction flag 7

        // update solution with direction calculated
        // TODO: do line search with \nabla\theta for acceptable lcp_tau
        value_type lcp_tau = 1.0;

        Arr_t::getDevice().axpy(lcp_tau, dlambda.begin(), lambda.begin(), 
                dlambda.size(), lambda.begin());
        
        // check lambda if all >= 0
        value_type* tmp_i = lambda.begin();
        for(size_t idx=0; idx < lambda.size(); idx++)
        {
            if(tmp_i[idx] < 0)
                tmp_i[idx] = 0;
        }
        // end of update solution
        lcp_iter++;
        pvfmm::Profile::Toc();
    }
    pvfmm::Profile::Toc();

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ParallelCVJacobianTrans(const Arr_t &lambda, Vec_t &f_col, bool simple) const
{
    //COUT("begin cvjt");
    // begin of MPI communication
    int myrank,np;
    MPI_Comm comm=MPI_COMM_WORLD;
    MPI_Comm_rank(comm,&myrank);
    MPI_Comm_size(comm,&np);
    std::map<int, std::vector<int>*>::const_iterator iter_i;
    std::vector< std::pair<size_t, size_t> > pid_cvid;
    int nv = nv_;
    // prepare pid lambda pairs need to send out
    //COUT("ghost vgrad ind size: "<<ghost_vgrad_ind_.size());
    for(iter_i=ghost_vgrad_ind_.begin(); iter_i!=ghost_vgrad_ind_.end(); iter_i++)
    {
        const std::vector<int> &value_i = *(iter_i->second);
        size_t pid = floor(iter_i->first/nv);
        ASSERT(pid != myrank, "wrong ghost vgrad id");
        if(pid != myrank)
        {
            for(size_t i=0; i<value_i.size(); i++)
            {
                if(value_i[i]>0)
                    pid_cvid.push_back(std::make_pair(pid, value_i[i]));
            }
        }
    }
    // sort by pid, and remove duplicate
    std::sort(pid_cvid.begin(), pid_cvid.end());
    pid_cvid.erase(std::unique(pid_cvid.begin(),pid_cvid.end()), pid_cvid.end());
    
    // prepare all to all communication for lambdas
    s_ind_cnt_.SetZero(); s_ind_dsp_.SetZero(); r_ind_cnt_.SetZero(); r_ind_dsp_.SetZero();
    s_ind_.clear(); r_ind_.clear(); s_value_.clear(); r_value_.clear();
    
    // set send count for vid and vesicle coordinate, set ves_id data to send
    for(size_t i=0; i<pid_cvid.size(); i++)
    {
        s_ind_cnt_[pid_cvid[i].first]++;
        s_ind_.push_back(pid_cvid[i].second);

        ASSERT(pid_cvid[i].second>cvid_dsp_, "wrong cvid in cvjacobiantrans");
        ASSERT(pid_cvid[i].second<=(cvid_dsp_+num_cvs_), "wrong cvid in cvjacobiantrans");

        size_t local_i = pid_cvid[i].second - cvid_dsp_ - 1;
        s_value_.push_back(lambda.begin()[local_i]);
    }

    // send and receive count
    MPI_Alltoall(&s_ind_cnt_[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &r_ind_cnt_[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    s_ind_dsp_[0]=0; pvfmm::omp_par::scan(&s_ind_cnt_[0], &s_ind_dsp_[0], s_ind_cnt_.Dim());
    r_ind_dsp_[0]=0; pvfmm::omp_par::scan(&r_ind_cnt_[0], &r_ind_dsp_[0], r_ind_cnt_.Dim());

    size_t recv_size = r_ind_cnt_[np-1] + r_ind_dsp_[np-1];
    r_ind_.resize(recv_size, 0); 
    r_value_.resize(recv_size, 0);

    // TODO replace with Mpi_Alltoallv_sparse
    // send and receive ind and values
    MPI_Alltoallv(&s_ind_[0], &s_ind_cnt_[0], &s_ind_dsp_[0], pvfmm::par::Mpi_datatype<size_t>::value(), 
                  &r_ind_[0], &r_ind_cnt_[0], &r_ind_dsp_[0], pvfmm::par::Mpi_datatype<size_t>::value(), comm);
    MPI_Alltoallv(&s_value_[0], &s_ind_cnt_[0], &s_ind_dsp_[0], pvfmm::par::Mpi_datatype<value_type>::value(), 
                  &r_value_[0], &r_ind_cnt_[0], &r_ind_dsp_[0], pvfmm::par::Mpi_datatype<value_type>::value(), comm);
    
    // insert to ghost_lambda_ map
    ghost_lambda_.clear();
    for(size_t i=0; i<r_ind_.size(); i++)
    {
        ghost_lambda_[r_ind_[i]] = r_value_[i];
    }
    //COUT("ghost lambda size in CVJacobianTrans: "<<ghost_lambda_.size());
    // end of MPI communication

    // similar to CVJacobianTrans calculation
    std::vector<value_type> f_col_host;
    f_col_host.resize(vgrad_.size(), 0.0);

    std::vector<value_type> vgrad_host;
    vgrad_host.resize(vgrad_.size(), 0.0);
    // copy vgrad_ to host
    vgrad_.getDevice().Memcpy(&vgrad_host.front(), vgrad_.begin(),
        vgrad_.size() * sizeof(value_type),
        vgrad_.getDevice().MemcpyDeviceToHost);

    std::vector<value_type> lambda_host;
    lambda_host.resize(lambda.size(), 0.0);
    // copy lambda to host
    lambda.getDevice().Memcpy(&lambda_host.front(), lambda.begin(),
            lambda.size() * sizeof(value_type),
            lambda.getDevice().MemcpyDeviceToHost);

    //COUT("before omp f_col");
    size_t ncount = vgrad_.size();
    #pragma omp parallel for
    for (size_t icount = 0; icount < ncount; icount++)
    {
        if(vgrad_ind_[icount] > 0)
        {
            if(vgrad_ind_[icount] > cvid_dsp_ && vgrad_ind_[icount] <= (cvid_dsp_+num_cvs_))
            {
                size_t local_i = vgrad_ind_[icount] - cvid_dsp_ - 1;
                f_col_host[icount] = lambda_host[local_i] * vgrad_host[icount];
            }
            else
            {
                //ASSERT(ghost_lambda_.find(vgrad_ind_[icount])!=ghost_lambda_.cend(), "missing ghost lambda in cvjacobiantrans");
                if(ghost_lambda_.find(vgrad_ind_[icount]) == ghost_lambda_.end())
                    COUT("what is this????");
                //COUT("ghost lambda mul");
                f_col_host[icount] = ghost_lambda_[vgrad_ind_[icount]] * vgrad_host[icount];
                //COUT("end ghost lambda mul");
            }
        }
    }
    //COUT("after omp f_col");
        
    //filtered f_col_up and downsample
    std::auto_ptr<Vec_t> xwrk = checkoutVec();
    std::auto_ptr<Vec_t> f_col_up = checkoutVec();
    std::auto_ptr<Vec_t> wrk1 = checkoutVec();
    std::auto_ptr<Vec_t> wrk2 = checkoutVec();
    xwrk->replicate(vgrad_);
    f_col_up->replicate(vgrad_);
    wrk1->replicate(vgrad_);
    wrk2->replicate(vgrad_);
    
    xwrk->getDevice().Memcpy(xwrk->begin(), &f_col_host.front(),
        xwrk->size() * sizeof(value_type),
        xwrk->getDevice().MemcpyHostToDevice);

    if(simple)
    {
        xwrk->getDevice().Memcpy(f_col.begin(), xwrk->begin(), xwrk->size()*sizeof(value_type), xwrk->getDevice().MemcpyDeviceToDevice);
    }
    else
    {
        if(params_.col_upsample)
        {
            sht_filter_high(*xwrk, *f_col_up, &sht_upsample_, params_.rep_exponent);
            Resample(*f_col_up, sht_upsample_, sht_, *wrk1, *wrk2, f_col);
        }
        else
            sht_filter_high(*xwrk, f_col, &sht_, params_.rep_exponent);
    }
    
    recycle(xwrk);
    recycle(f_col_up);
    recycle(wrk1);
    recycle(wrk2);
    //end of filtered and downsample

    //COUT("end cvjt");
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ParallelCVJacobian(const Vec_t &x_new, Arr_t &lambda_mat_vec, bool simple) const
{
    // begin of MPI
    int myrank, np;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &np);
    
    // number of cvs of all processes
    pvfmm::Vector<size_t> world_num_cvs(np);
    size_t num_cvs_myrank = num_cvs_;
    MPI_Allgather(&num_cvs_myrank,   1, pvfmm::par::Mpi_datatype<size_t>::value(),
                  &world_num_cvs[0], 1, pvfmm::par::Mpi_datatype<size_t>::value(), comm);

    // calculate cvid displacement for each process
    pvfmm::Vector<size_t> cvid_dsp(np);
    cvid_dsp[0]=0; pvfmm::omp_par::scan(&world_num_cvs[0], &cvid_dsp[0], world_num_cvs.Dim());
    cvid_dsp_ = cvid_dsp[myrank];
    // cv_id of current process is (cvid_dsp, cvid_dsp+num_cvs_]
    // end of MPI

    std::auto_ptr<Vec_t> x_vgrad = checkoutVec();
    std::auto_ptr<Vec_t> x_new_up = checkoutVec();
    std::auto_ptr<Vec_t> wrk1 = checkoutVec();
    std::auto_ptr<Vec_t> wrk2 = checkoutVec();
    x_vgrad->replicate(vgrad_);
    x_new_up->replicate(vgrad_);
    wrk1->replicate(vgrad_);
    wrk2->replicate(vgrad_);

    if(params_.col_upsample && (!simple))
    {
        Resample(x_new, sht_, sht_upsample_, *wrk1, *wrk2, *x_new_up);
        xy(*x_new_up, vgrad_, *x_vgrad);
    }
    else
        xy(x_new, vgrad_, *x_vgrad);

    std::vector<value_type> x_vgrad_host(x_vgrad->size(), 0.0);
    // copy x_vgrad to host
    x_vgrad->getDevice().Memcpy(&x_vgrad_host.front(), x_vgrad->begin(),
        x_vgrad->size() * sizeof(value_type),
        x_vgrad->getDevice().MemcpyDeviceToHost);

    std::vector<value_type> lambda_mat_vec_host(lambda_mat_vec.size(), 0.0);
    std::map<size_t, value_type> ghost_lambda_mat_vec;
    
    size_t ncount = vgrad_.size();
    //#pragma omp parallel for
    for (size_t icount = 0; icount < ncount; icount++)
    {
        if(vgrad_ind_[icount] > 0)
        {
            if(vgrad_ind_[icount]>cvid_dsp_ && vgrad_ind_[icount]<=(cvid_dsp_+num_cvs_))
            {
                size_t local_i = vgrad_ind_[icount] - cvid_dsp_ - 1;
                lambda_mat_vec_host[local_i] += x_vgrad_host[icount];
            }
            else
            {
                ghost_lambda_mat_vec[vgrad_ind_[icount]] += x_vgrad_host[icount];
            }
            //#pragma omp atomic
            //{
            //}
        }
    }
    //COUT("ghost lambda mat vec size in CVJacobian: "<<ghost_lambda_mat_vec.size());
    
    // begin of MPI
    // set pid cvid pair for sending out
    typename std::map<size_t, value_type>::const_iterator iter_i;
    std::vector< std::pair<size_t, size_t> > pid_cvid;
    for(iter_i=ghost_lambda_mat_vec.begin(); iter_i!=ghost_lambda_mat_vec.end(); iter_i++)
    {
        int pid = std::lower_bound(&cvid_dsp[0], &cvid_dsp[0]+np, iter_i->first) - &cvid_dsp[0] - 1;
        ASSERT(pid!=myrank, "wrong pid in cvjacobian");
        pid_cvid.push_back(std::make_pair(pid, iter_i->first));
    }
    // sort by pid, and remove duplicate
    std::sort(pid_cvid.begin(), pid_cvid.end());
    pid_cvid.erase(std::unique(pid_cvid.begin(),pid_cvid.end()), pid_cvid.end());
    
    // prepare all to all communication for lambdas
    s_ind_cnt_.SetZero(); s_ind_dsp_.SetZero(); r_ind_cnt_.SetZero(); r_ind_dsp_.SetZero();
    s_ind_.clear(); r_ind_.clear(); s_value_.clear(); r_value_.clear();
    
    // set send count for vid and vesicle coordinate, set ves_id data to send
    for(size_t i=0; i<pid_cvid.size(); i++)
    {
        s_ind_cnt_[pid_cvid[i].first]++;
        s_ind_.push_back(pid_cvid[i].second);
        s_value_.push_back(ghost_lambda_mat_vec[pid_cvid[i].second]);
    }

    // send and receive count
    MPI_Alltoall(&s_ind_cnt_[0], 1, pvfmm::par::Mpi_datatype<int>::value(),
                 &r_ind_cnt_[0], 1, pvfmm::par::Mpi_datatype<int>::value(), comm);

    s_ind_dsp_[0]=0; pvfmm::omp_par::scan(&s_ind_cnt_[0], &s_ind_dsp_[0], s_ind_cnt_.Dim());
    r_ind_dsp_[0]=0; pvfmm::omp_par::scan(&r_ind_cnt_[0], &r_ind_dsp_[0], r_ind_cnt_.Dim());

    size_t recv_size = r_ind_cnt_[np-1] + r_ind_dsp_[np-1];
    r_ind_.resize(recv_size, 0); 
    r_value_.resize(recv_size, 0);

    // TODO replace with Mpi_Alltoallv_sparse
    // send and receive inds and vlaues
    MPI_Alltoallv(&s_ind_[0], &s_ind_cnt_[0], &s_ind_dsp_[0], pvfmm::par::Mpi_datatype<size_t>::value(), 
                  &r_ind_[0], &r_ind_cnt_[0], &r_ind_dsp_[0], pvfmm::par::Mpi_datatype<size_t>::value(), comm);
    MPI_Alltoallv(&s_value_[0], &s_ind_cnt_[0], &s_ind_dsp_[0], pvfmm::par::Mpi_datatype<value_type>::value(), 
                  &r_value_[0], &r_ind_cnt_[0], &r_ind_dsp_[0], pvfmm::par::Mpi_datatype<value_type>::value(), comm);
    
    // add received lambda_mat_vec from other processes
    for(size_t i=0; i<r_ind_.size(); i++)
    {
        ASSERT(r_ind_[i]>cvid_dsp_, "wrong received cvid");
        ASSERT(r_ind_[i]<=(cvid_dsp_+num_cvs_), "wrong received cvid");
        size_t local_i = r_ind_[i] - cvid_dsp_ - 1;
        lambda_mat_vec_host[local_i] += r_value_[i];
    }
    //end of MPI

    // copy lambda_mat_vec to device as return
    lambda_mat_vec.getDevice().Memcpy(lambda_mat_vec.begin(), &lambda_mat_vec_host.front(),
            lambda_mat_vec.size() * sizeof(value_type),
            lambda_mat_vec.getDevice().MemcpyHostToDevice);

    recycle(x_vgrad);
    recycle(x_new_up);
    recycle(wrk1);
    recycle(wrk2);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ConfigureLCPSolver() const
{
    PROFILESTART();
    COUTDEBUG("Configuring the parallel LCP linear system solver");

    typedef typename PSolver_t::matvec_type POp;
    typedef typename PSolver_t::vec_type PVec;
    typedef typename PVec::size_type size_type;

    size_type sz(num_cvs_); size_type SZ(sum_num_cvs_);
    
    // set new parallel linear solver
    if(lcp_parallel_linear_solver_)
        delete lcp_parallel_linear_solver_;
    lcp_parallel_linear_solver_ = new ParallelLinSolverPetsc<value_type>(VES3D_COMM_WORLD);

    // Setting up the operator
    if(parallel_matvec_ != NULL)
        delete parallel_matvec_;
    parallel_matvec_ = NULL;
    CHK(lcp_parallel_linear_solver_->LinOpFactory(&parallel_matvec_));
    CHK(parallel_matvec_->SetSizes(sz,sz,SZ,SZ));
    CHK(parallel_matvec_->SetName("LCP matrix"));
    CHK(parallel_matvec_->SetContext(static_cast<const void*>(this)));
    CHK(parallel_matvec_->SetApply(ParallelLCPApply));
    CHK(parallel_matvec_->Configure());

    // setting up the rhs
    if(parallel_rhs_ != NULL)
        delete parallel_rhs_;
    parallel_rhs_ = NULL;
    CHK(lcp_parallel_linear_solver_->VecFactory(&parallel_rhs_));
    CHK(parallel_rhs_->SetSizes(sz,SZ));
    CHK(parallel_rhs_->SetName("rhs"));
    CHK(parallel_rhs_->Configure());

    if(parallel_u_ !=NULL)
        delete parallel_u_;
    parallel_u_ = NULL;
    CHK(parallel_rhs_->ReplicateTo(&parallel_u_));
    CHK(parallel_u_->SetSizes(sz,SZ));
    CHK(parallel_u_->SetName("solution"));
    
    // setting up the solver
    CHK(lcp_parallel_linear_solver_->SetOperator(parallel_matvec_));
    CHK(lcp_parallel_linear_solver_->SetTolerances(params_.time_tol*1e-2*100,
                PSolver_t::PLS_DEFAULT,
                PSolver_t::PLS_DEFAULT,
                params_.time_iter_max));
    CHK(lcp_parallel_linear_solver_->Configure());
    CHK(lcp_parallel_linear_solver_->SetPrecondContext(static_cast<const void*>(this)));
    CHK(lcp_parallel_linear_solver_->UpdatePrecond(ParallelLCPPrecond));

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
ParallelLCPMatvec(Arr_t &lambda) const
{
    PROFILESTART();
    MPI_Comm comm = MPI_COMM_WORLD;
    // prepare send values of lambda
    for(size_t i=0; i<s_ind_.size(); i++)
    {
        size_t ind_i = s_ind_[i];
        size_t local_i = ind_i - cvid_dsp_ - 1;
        ASSERT(ind_i > cvid_dsp_, "wrong send cvid"); 
        ASSERT(ind_i <= cvid_dsp_+num_cvs_, "wrong send cvid"); 
        s_value_[i] = lambda.begin()[local_i];
    }

    // send and receive lambdas
    MPI_Alltoallv(&s_value_[0], &s_ind_cnt_[0], &s_ind_dsp_[0], pvfmm::par::Mpi_datatype<value_type>::value(), 
                  &r_value_[0], &r_ind_cnt_[0], &r_ind_dsp_[0], pvfmm::par::Mpi_datatype<value_type>::value(), comm);

    // add received lambda to ghost_lambda_ map
    ghost_lambda_.clear();
    for(size_t i=0; i<r_ind_.size(); i++)
    {
        ghost_lambda_[r_ind_[i]] = r_value_[i];
    }
    //COUT("ghost lambda mat vec size in lcpmatvec: "<<ghost_lambda_.size());

    // do lcp matvec
    if(num_cvs_>0)
    {
        Arr_t matvec(num_cvs_);
        Arr_t::getDevice().Memset(matvec.begin(), 0, sizeof(value_type)*matvec.size());
        value_type *lambda_i = lambda.begin();
        value_type *matvec_i = matvec.begin();

        typename std::map<std::pair<size_t, size_t>, value_type>::iterator iter_i;
        for(iter_i = parallel_lcp_matrix_.begin(); iter_i != parallel_lcp_matrix_.end(); iter_i++)
        {
            size_t ind_i = iter_i->first.first;
            size_t ind_j = iter_i->first.second;
            value_type val_tmp = iter_i->second;

            ASSERT(ind_i>cvid_dsp_, "wrong ind_i in matvec");
            ASSERT(ind_i<=(cvid_dsp_+num_cvs_), "wrong ind_i in matvec");

            size_t local_i = ind_i - cvid_dsp_ - 1;

            if(ind_j>cvid_dsp_ && ind_j<=(cvid_dsp_+num_cvs_))
            {
                size_t local_j = ind_j - cvid_dsp_ - 1;
                matvec_i[local_i] += val_tmp*lambda_i[local_j];
            }
            else
            {
                ASSERT(ghost_lambda_.find(ind_j)!=ghost_lambda_.end(), "missing ghost lambda in matvec");
                matvec_i[local_i] += val_tmp*ghost_lambda_[ind_j];
            }
        }
        pvfmm::Profile::Add_FLOP( ((long long)parallel_lcp_matrix_.size())*4);

        // apply PA
        LCPSelect(lambda, matvec);
        // copy to matvec to lambda as return
        lambda.getDevice().Memcpy(lambda.begin(), matvec.begin(), 
                lambda.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
    }

    PROFILEEND("",0);
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
GetStartPosAll(const Vec_t &xs_in, const Vec_t &xe_in, Vec_t &xs_out) const
{
    xs_out.replicate(xs_in);
/*
    size_t p1 = params_.sh_order;
    size_t N_ves = xs_in.getNumSubs(); // Number of vesicles

    axpy(static_cast<value_type>(-1.0), xs_in, xe_in, xs_out);

    std::vector<value_type> cx(N_ves, 0.0);
    std::vector<value_type> cy(N_ves, 0.0);
    std::vector<value_type> cz(N_ves, 0.0);
    
    #pragma omp parallel for
    for(size_t k=0;k<N_ves;k++){
        for(size_t i=0;i<p1+1;i++){
            for(size_t j=0;j<2*p1;j++){
                cx[k]+=xs_out.begin()[j+2*p1*(i+(p1+1)*(0+k*COORD_DIM))];
                cy[k]+=xs_out.begin()[j+2*p1*(i+(p1+1)*(1+k*COORD_DIM))];
                cz[k]+=xs_out.begin()[j+2*p1*(i+(p1+1)*(2+k*COORD_DIM))];
            }
        }
        cx[k] /= (2*p1*(p1+1));
        cy[k] /= (2*p1*(p1+1));
        cz[k] /= (2*p1*(p1+1));
    }

    double C[COORD_DIM]={0,0,0};
    double C_glb[COORD_DIM]={0,0,0};
    for(size_t k=0;k<N_ves;k++)
    {
        C[0] += cx[k];
        C[1] += cy[k];
        C[2] += cz[k];
    }
    C[0] /= N_ves; C[1] /= N_ves; C[2] /= N_ves;
    
    int npros;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &npros);
    MPI_Allreduce(C, C_glb, 3, MPI_DOUBLE, MPI_SUM, comm);
    
    C_glb[0] /= npros; C_glb[1] /= npros; C_glb[2] /= npros;
*/

    xs_out.getDevice().Memcpy(xs_out.begin(), xs_in.begin(), xs_in.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);

/*
    double c_len = sqrt(C_glb[0]*C_glb[0] + C_glb[1]*C_glb[1] + C_glb[2]*C_glb[2]);
    if(c_len <= 0.04)
    {
	return ErrorEvent::Success;
    }
    else
    {   
        double c_scale = (c_len - 0.04)/c_len;
        C_glb[0] *= c_scale;
        C_glb[1] *= c_scale;
        C_glb[2] *= c_scale;
    }

    #pragma omp parallel for
    for(size_t k=0;k<N_ves;k++){
        for(size_t i=0;i<p1+1;i++){
            for(size_t j=0;j<2*p1;j++){
                for(size_t l=0;l<COORD_DIM;l++){
                    xs_out.begin()[j+2*p1*(i+(p1+1)*(l+k*COORD_DIM))] += C_glb[l];
                }
            }
        }
    }
*/
    
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::
GetStartPosPair(const Vec_t &xs_in, const Vec_t &xe_in, Vec_t &xs_out) const
{
    xs_out.replicate(xs_in);
/*
    size_t p1 = params_.sh_order;
    size_t N_ves = xs_in.getNumSubs(); // Number of vesicles

    axpy(static_cast<value_type>(-1.0), xs_in, xe_in, xs_out);

    std::vector<value_type> cx(N_ves, 0.0);
    std::vector<value_type> cy(N_ves, 0.0);
    std::vector<value_type> cz(N_ves, 0.0);
    
    for(size_t k=0;k<N_ves;k++){
        for(size_t i=0;i<p1+1;i++){
            for(size_t j=0;j<2*p1;j++){
                cx[k]+=xs_out.begin()[j+2*p1*(i+(p1+1)*(0+k*COORD_DIM))];
                cy[k]+=xs_out.begin()[j+2*p1*(i+(p1+1)*(1+k*COORD_DIM))];
                cz[k]+=xs_out.begin()[j+2*p1*(i+(p1+1)*(2+k*COORD_DIM))];
            }
        }
        cx[k] /= (2*p1*(p1+1));
        cy[k] /= (2*p1*(p1+1));
        cz[k] /= (2*p1*(p1+1));
    }

    value_type C[COORD_DIM]={0,0,0};
    for(size_t k=0;k<N_ves;k++)
    {
        C[0] += cx[k];
        C[1] += cy[k];
        C[2] += cz[k];
    }
    C[0] /= N_ves; C[1] /= N_ves; C[2] /= N_ves;
*/

    xs_out.getDevice().Memcpy(xs_out.begin(), xs_in.begin(), xs_in.size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);

/*
    double c_len = sqrt(C[0]*C[0] + C[1]*C[1] + C[2]*C[2]);
    if(c_len <= 0.04)
    {
	return ErrorEvent::Success;
    }
    else
    {   
        double c_scale = (c_len - 0.04)/c_len;
        C[0] *= c_scale;
        C[1] *= c_scale;
        C[2] *= c_scale;
    }

    for(size_t k=0;k<N_ves;k++){
        for(size_t i=0;i<p1+1;i++){
            for(size_t j=0;j<2*p1;j++){
                for(size_t l=0;l<COORD_DIM;l++){
                    xs_out.begin()[j+2*p1*(i+(p1+1)*(l+k*COORD_DIM))] += C[l];
                }
            }
        }
    }
*/
    
    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
typename InterfacialVelocity<SurfContainer, Interaction>::value_type InterfacialVelocity<SurfContainer, Interaction>::
StokesError(const Vec_t &x) const
{
    PROFILESTART();
    stokes_.SetDensitySL(NULL);
    stokes_.SetDensityDL(NULL);
    stokes_.SetSrcCoord(x);
    value_type stokes_error=stokes_.MonitorError(params_.time_tol*0.1);
    PROFILEEND("",0);

    return stokes_error;
}

template <class Vec_t, class SHT>
static std::vector<typename Vec_t::value_type> inner_prod(const Vec_t& v1, const Vec_t& v2, SHT* sh_trans, int rep_exp){
  typedef typename Vec_t::value_type value_type;

  static Vec_t w, v1_, v2_;
  { // Set v1_
    v1_.replicate(v1);
    w  .replicate(v1);
    sh_trans->forward(v1, w, v1_);
  }
  { // Set v2_
    v2_.replicate(v2);
    w  .replicate(v2);
    sh_trans->forward(v2, w, v2_);
  }
  size_t p=v1.getShOrder();
  int ns_x = v1.getNumSubFuncs();

  assert(p<256);
  assert(rep_exp<128);
  static std::vector<value_type> A_[256*128];
  std::vector<value_type>& A=A_[rep_exp*256+p];
  if(!A.size()){
    A.resize(p+1);
    long filter_freq_=(rep_exp?p/2:p/3);
    for(int ii=0; ii<= p; ++ii){
      value_type a = 1.0 - (rep_exp?std::pow(ii*1.0/filter_freq_,rep_exp):0);
      a *= (ii > filter_freq_ ? 0.0 : 1.0 );
      A[ii] = 1.0 - a;
    }
  }

  std::vector<value_type> E(ns_x/COORD_DIM,0);
  for(int ii=0; ii<= p; ++ii){
    value_type* inPtr_v1 = v1_.begin() + ii;
    value_type* inPtr_v2 = v2_.begin() + ii;
    int len = 2*ii + 1 - (ii/p);
    for(int jj=0; jj< len; ++jj){
      int dist = (p + 1 - (jj + 1)/2);
      for(int ss=0; ss<ns_x; ++ss){
        E[ss/COORD_DIM] += A[ii]*(*inPtr_v1)*(*inPtr_v2);
        inPtr_v1 += dist;
        inPtr_v2 += dist;
      }
      inPtr_v1--;
      inPtr_v2--;
      inPtr_v1 += jj%2;
      inPtr_v2 += jj%2;
    }
  }

  return E;
}

template <class Vec_t, class SHT>
static void sht_filter(const Vec_t& v1, Vec_t& v2, SHT* sh_trans, int rep_exp){
  typedef typename Vec_t::value_type value_type;

  static Vec_t w, v1_;
  { // Set v1_
    v1_.replicate(v1);
    w  .replicate(v1);
    sh_trans->forward(v1, w, v1_);
  }
  size_t p=v1.getShOrder();
  int ns_x = v1.getNumSubFuncs();

  assert(p<256);
  assert(rep_exp<128);
  static std::vector<value_type> A_[256*128];
  std::vector<value_type>& A=A_[rep_exp*256+p];
  if(!A.size()){
    A.resize(p+1);
    long filter_freq_=(rep_exp?p/2:p/3);
    for(int ii=0; ii<= p; ++ii){
      value_type a = 1.0 - (rep_exp?std::pow(ii*1.0/filter_freq_,rep_exp):0);
      a *= (ii > filter_freq_ ? 0.0 : 1.0 );
      A[ii] = 1.0 - a;
    }
  }

  value_type E=0;
  for(int ii=0; ii<= p; ++ii){
    value_type* inPtr_v1 = v1_.begin() + ii;
    int len = 2*ii + 1 - (ii/p);
    for(int jj=0; jj< len; ++jj){
      int dist = (p + 1 - (jj + 1)/2);
      for(int ss=0; ss<ns_x; ++ss){
        inPtr_v1[0] *= -A[ii];
        inPtr_v1 += dist;
      }
      inPtr_v1--;
      inPtr_v1 += jj%2;
    }
  }

  { // Set v2
    v2 .replicate(v1);
    sh_trans->backward(v1_, w, v2);
  }
}

template <class Vec_t, class SHT>
static void sht_filter_high(const Vec_t& v1, Vec_t& v2, SHT* sh_trans, int rep_exp){
  typedef typename Vec_t::value_type value_type;

  //static Vec_t wH, v1H_;
  Vec_t wH, v1H_;
  { // Set v1_
    v1H_.replicate(v1);
    wH  .replicate(v1);
    sh_trans->forward(v1, wH, v1H_);
  }
  size_t p=v1.getShOrder();
  int ns_x = v1.getNumSubFuncs();

  //assert(p<256);
  //assert(rep_exp<128);
  //static std::vector<value_type> AH_[256*128];
  //std::vector<value_type>& AH=AH_[rep_exp*256+p];
  std::vector<value_type> AH;
//#pragma omp critical(ShtFilterHigh)
//{
  if(!AH.size()){
    AH.resize(p+1);
    long filter_freq_=(rep_exp?p/2:p/3);
    for(int ii=0; ii<= p; ++ii){
      value_type a = 1.0 - (rep_exp?std::pow(ii*1.0/filter_freq_,rep_exp):0);
      a *= (ii > filter_freq_ ? 0.0 : 1.0 );
      AH[ii] = a;
    }
  }
//}

  value_type E=0;
  for(int ii=0; ii<= p; ++ii){
    value_type* inPtr_v1 = v1H_.begin() + ii;
    int len = 2*ii + 1 - (ii/p);
    for(int jj=0; jj< len; ++jj){
      int dist = (p + 1 - (jj + 1)/2);
      for(int ss=0; ss<ns_x; ++ss){
        inPtr_v1[0] *= AH[ii];
        inPtr_v1 += dist;
      }
      inPtr_v1--;
      inPtr_v1 += jj%2;
    }
  }

  { // Set v2
    v2 .replicate(v1);
    sh_trans->backward(v1H_, wH, v2);
  }
}

template <class Sca_t, class SHT>
static void sht_filter_high_sca(const Sca_t& v1, Sca_t& v2, SHT* sh_trans, int rep_exp){
  typedef typename Sca_t::value_type value_type;

  static Sca_t wHsca, v1Hsca_;
  { // Set v1_
    v1Hsca_.replicate(v1);
    wHsca  .replicate(v1);
    sh_trans->forward(v1, wHsca, v1Hsca_);
  }
  size_t p=v1.getShOrder();
  int ns_x = v1.getNumSubFuncs();

  assert(p<256);
  assert(rep_exp<128);
  static std::vector<value_type> AH_[256*128];
  std::vector<value_type>& AH=AH_[rep_exp*256+p];
  if(!AH.size()){
    AH.resize(p+1);
    long filter_freq_=(rep_exp?p/2:p/3);
    for(int ii=0; ii<= p; ++ii){
      value_type a = 1.0 - (rep_exp?std::pow(ii*1.0/filter_freq_,rep_exp):0);
      a *= (ii > filter_freq_ ? 0.0 : 1.0 );
      AH[ii] = a;
    }
  }

  value_type E=0;
  for(int ii=0; ii<= p; ++ii){
    value_type* inPtr_v1 = v1Hsca_.begin() + ii;
    int len = 2*ii + 1 - (ii/p);
    for(int jj=0; jj< len; ++jj){
      int dist = (p + 1 - (jj + 1)/2);
      for(int ss=0; ss<ns_x; ++ss){
        inPtr_v1[0] *= AH[ii];
        inPtr_v1 += dist;
      }
      inPtr_v1--;
      inPtr_v1 += jj%2;
    }
  }

  { // Set v2
    v2 .replicate(v1);
    sh_trans->backward(v1Hsca_, wHsca, v2);
  }
}

template<typename SurfContainer, typename Interaction>
Error_t InterfacialVelocity<SurfContainer, Interaction>::reparam()
{
    //std::cout<<"begin reparam.\n";
    PROFILESTART();

    value_type ts(params_.rep_ts);
    value_type rep_tol(params_.rep_tol);
    long rep_maxit(params_.rep_maxit);
    long rep_exp=params_.rep_exponent;
    if(params_.rep_type==BoxReparam) rep_exp=0;

    // Set default values
    if(ts<0) ts=1e-3;
    if(rep_tol<0) rep_tol=1e-6;
    if(rep_maxit<0) rep_maxit=4000;

    SurfContainer* Surf;
    SHtrans_t* sh_trans;
    if (params_.rep_upsample){
        //COUT("Upsampling for reparametrization");
        S_.resample(params_.upsample_freq, &S_up_);
        Surf = S_up_;
        sh_trans = &sht_upsample_;
    } else {
        Surf = &S_;
        sh_trans = &sht_;
    }

    // Storage for resolving collision
    std::auto_ptr<Vec_t> x_old = checkoutVec();
    std::auto_ptr<Vec_t> u1_down = checkoutVec();
    x_old ->replicate(S_.getPosition());
    u1_down ->replicate(S_.getPosition());
    
    // prepare for upsample tension
    std::auto_ptr<Sca_t> twrk1 = checkoutSca();
    std::auto_ptr<Sca_t> twrk2 = checkoutSca();
    std::auto_ptr<Sca_t> tencpy = checkoutSca();
    twrk1->replicate(Surf->getPosition());
    twrk2->replicate(Surf->getPosition());
    tencpy->replicate(S_.getPosition());

    std::auto_ptr<Vec_t> u1 = checkoutVec();
    std::auto_ptr<Vec_t> u2 = checkoutVec();
    std::auto_ptr<Vec_t> x_old_up = checkoutVec();
    std::auto_ptr<Sca_t> wrk = checkoutSca();
    u1 ->replicate(Surf->getPosition());
    u2 ->replicate(Surf->getPosition());
    x_old_up ->replicate(Surf->getPosition());
    wrk->replicate(Surf->getPosition());
    long N_ves = u1->getNumSubs();
    
    // store collision free state in x_old
    if (params_.rep_upsample)
    {
        axpy(static_cast<value_type>(-0.0001), Surf->getNormal(), Surf->getPosition(), *x_old_up);
        Resample(*x_old_up, sht_upsample_, sht_, *u1, *u2, *x_old);
    }
    else
    {
        axpy(static_cast<value_type>(-0.0001), Surf->getNormal(), Surf->getPosition(), *x_old);
    }
    // End of storage for resolving collision

    value_type E0=0;
    { // Compute energy E0
        std::vector<value_type>  x2=inner_prod(Surf->getPosition(), Surf->getPosition(), sh_trans, rep_exp);
        for(long i=0;i<x2.size();i++) E0+=x2[i];
    }
        
    //ShuffleVector(S_.fc_, *x_old, 1);
    //ShuffleVector(pos_vel_, *u1_down, 1);
    int ii(0);
    std::vector<value_type> E;
    while ( ii < rep_maxit )
    {
        //Surf->getSmoothedShapePositionReparam(*u1);
        //axpy(static_cast<value_type>(-1), Surf->getPosition(), *u1, *u1);
        sht_filter(Surf->getPosition(), *u1, sh_trans, rep_exp);
        Surf->mapToTangentSpace(*u1, false /* upsample */);
        { // Filter u1
            static Vec_t u_, w;
            u_.replicate(*u1);
            w .replicate(*u1);
            sh_trans->forward(*u1, w, u_);
            sh_trans->backward(u_, w, *u1);
        }
        Surf->mapToTangentSpace(*u1, false /* upsample */);
        { // Filter u1
            static Vec_t u_, w;
            u_.replicate(*u1);
            w .replicate(*u1);
            sh_trans->forward(*u1, w, u_);
            sh_trans->backward(u_, w, *u1);
        }

        { // normalize u1 for each vesicle
            long N = u1->getNumSubFuncs();
            long M=u1->size()/N;
            value_type* u=u1->begin();
            for(long i=0;i<N/COORD_DIM;i++){
                value_type max_v=0;
                for(long j=0;j<M;j++){
                    value_type x=u[j+M*(0+i*COORD_DIM)];
                    value_type y=u[j+M*(1+i*COORD_DIM)];
                    value_type z=u[j+M*(2+i*COORD_DIM)];
                    max_v=std::max(max_v, sqrt(x*x+y*y+z*z));
                }
                for(long j=0;j<M;j++){
                    u[j+M*(0+i*COORD_DIM)]/=max_v;
                    u[j+M*(1+i*COORD_DIM)]/=max_v;
                    u[j+M*(2+i*COORD_DIM)]/=max_v;
                }
            }
        }

        std::vector<value_type>  x_dot_x =inner_prod(Surf->getPosition(), Surf->getPosition(), sh_trans, rep_exp);
        std::vector<value_type>  x_dot_u1=inner_prod(Surf->getPosition(),                 *u1, sh_trans, rep_exp);
        std::vector<value_type> u1_dot_u1=inner_prod(                *u1,                 *u1, sh_trans, rep_exp);

        value_type dt_max(0);
        std::vector<value_type> dt(x_dot_u1.size(),0);
        for(long i=0; i<N_ves; i++){
            dt[i]=std::min(ts, -x_dot_u1[i]/u1_dot_u1[i]);
            if( dt[i]<rep_tol || (E.size() && (E[i]==0 /*|| E[i]-x_dot_x[i]<rep_tol*rep_tol*E[i]*/ )) ){
                x_dot_x[i]=0;
                dt[i]=0;
            }
            dt_max=std::max(dt_max,dt[i]);

            long N=u1->getStride()*VES3D_DIM;
            value_type* u1_=u1->getSubN_begin(i);
            for(long j=0;j<N;j++) u1_[j]*=dt[i];
        }
        if(dt_max==0) break;
        E=x_dot_x;

        //Advecting tension (useless for implicit)
        if (params_.scheme != GloballyImplicit){
            if (params_.rep_upsample){
                // upsample tension
                Resample(tension_, sht_, sht_upsample_, *twrk1, *twrk2, *wrk);
                
                // advect tension
                Surf->grad(*wrk, *u2);
                GeometricDot(*u2, *u1, *twrk1);
                axpy(1.0, *twrk1, *wrk, *wrk);
                
                // downsample tension
                Resample(*wrk, sht_upsample_, sht_, *twrk1, *twrk2, tension_);
                
                // Vm update
                Resample(el_force->Vm_, sht_, sht_upsample_, *twrk1, *twrk2, *wrk);
                Surf->grad(*wrk, *u2);
                GeometricDot(*u2, *u1, *twrk1);
                axpy(1.0, *twrk1, *wrk, *wrk);
                Resample(*wrk, sht_upsample_, sht_, *twrk1, *twrk2, el_force->Vm_);
/*
                // velocity
                for(int i=0; i<VES3D_DIM; i++)
                {
                    tencpy->getDevice().Memcpy(tencpy->begin(), &(u1_down->begin()[i*tencpy->size()]), 
                            tencpy->size() * sizeof(value_type), device_type::MemcpyDeviceToDevice);

                    // upsample tension
                    Resample(*tencpy, sht_, sht_upsample_, *twrk1, *twrk2, *wrk);
                
                    // advect tension
                    Surf->grad(*wrk, *u2);
                    GeometricDot(*u2, *u1, *twrk1);
                    axpy(1.0, *twrk1, *wrk, *wrk);
                
                    // downsample tension
                    Resample(*wrk, sht_upsample_, sht_, *twrk1, *twrk2, *tencpy);

                    u1_down->getDevice().Memcpy(&(u1_down->begin()[i*tencpy->size()]), tencpy->begin(),
                            tencpy->size() * sizeof(value_type), device_type::MemcpyDeviceToDevice);
                }
                // fc
                for(int i=0; i<VES3D_DIM; i++)
                {
                    tencpy->getDevice().Memcpy(tencpy->begin(), &(x_old->begin()[i*tencpy->size()]), 
                            tencpy->size() * sizeof(value_type), device_type::MemcpyDeviceToDevice);

                    // upsample tension
                    Resample(*tencpy, sht_, sht_upsample_, *twrk1, *twrk2, *wrk);
                
                    // advect tension
                    Surf->grad(*wrk, *u2);
                    GeometricDot(*u2, *u1, *twrk1);
                    axpy(1.0, *twrk1, *wrk, *wrk);
                
                    // downsample tension
                    Resample(*wrk, sht_upsample_, sht_, *twrk1, *twrk2, *tencpy);

                    x_old->getDevice().Memcpy(&(x_old->begin()[i*tencpy->size()]), tencpy->begin(),
                            tencpy->size() * sizeof(value_type), device_type::MemcpyDeviceToDevice);
                }
*/
            }
            else {
                // tension
                Surf->grad(tension_, *u2);
                GeometricDot(*u2, *u1, *wrk);
                axpy(1.0, *wrk, tension_, tension_);
                
                Surf->grad(el_force->Vm_, *u2);
                GeometricDot(*u2, *u1, *wrk);
                axpy(1.0, *wrk, el_force->Vm_, el_force->Vm_);
/*
                // velocity
                for(int i=0; i<VES3D_DIM; i++)
                {
                    tencpy->getDevice().Memcpy(tencpy->begin(), &(u1_down->begin()[i*tencpy->size()]), 
                            tencpy->size() * sizeof(value_type), device_type::MemcpyDeviceToDevice);
                    Surf->grad(*tencpy, *u2);
                    GeometricDot(*u2, *u1, *wrk);
                    axpy(1.0, *wrk, *tencpy, *tencpy);
                    u1_down->getDevice().Memcpy(&(u1_down->begin()[i*tencpy->size()]), tencpy->begin(),
                            tencpy->size() * sizeof(value_type), device_type::MemcpyDeviceToDevice);
                }
                // fc
                for(int i=0; i<VES3D_DIM; i++)
                {
                    tencpy->getDevice().Memcpy(tencpy->begin(), &(x_old->begin()[i*tencpy->size()]), 
                            tencpy->size() * sizeof(value_type), device_type::MemcpyDeviceToDevice);
                    Surf->grad(*tencpy, *u2);
                    GeometricDot(*u2, *u1, *wrk);
                    axpy(1.0, *wrk, *tencpy, *tencpy);
                    x_old->getDevice().Memcpy(&(x_old->begin()[i*tencpy->size()]), tencpy->begin(),
                            tencpy->size() * sizeof(value_type), device_type::MemcpyDeviceToDevice);
                }
*/
            }
        }

        //updating position
        axpy(1.0, *u1, Surf->getPosition(), Surf->getPositionModifiable());

        COUTDEBUG("Iteration = "<<ii<<", dt = "<<dt_max);
        ++ii;
    }
    //ShuffleVector(*x_old, S_.fc_, 0);
    //ShuffleVector(*u1_down, pos_vel_, 0);

    value_type E1=0;
    { // Compute energy E1
        std::vector<value_type>  x2=inner_prod(Surf->getPosition(), Surf->getPosition(), sh_trans, rep_exp);
        for(long i=0;i<x2.size();i++) E1+=x2[i];
    }
    INFO("Iterations = "<<ii<<", Energy = "<<E1<<", dE = "<<E1-E0);
    { // print log(coeff)
      std::auto_ptr<Vec_t> x = checkoutVec();
      { // Set x
        std::auto_ptr<Vec_t> w   = checkoutVec();
        x  ->replicate(Surf->getPosition());
        w  ->replicate(Surf->getPosition());
        sh_trans->forward(Surf->getPosition(), *w, *x);
        recycle(w);
      }
      {
          size_t p=x->getShOrder();
          int ns_x = x->getNumSubFuncs();
          std::vector<value_type> coeff_norm0(p+1,0);
          for(int ii=0; ii<= p; ++ii){
              value_type* inPtr_x = x->begin() + ii;

              int len = 2*ii + 1 - (ii/p);
              for(int jj=0; jj< len; ++jj){

                  int dist = (p + 1 - (jj + 1)/2);
                  for(int ss=0; ss<ns_x; ++ss){
                      coeff_norm0[ii] = std::max(coeff_norm0[ii], (*inPtr_x)*(*inPtr_x));
                      inPtr_x += dist;
                  }
                  inPtr_x--;
                  inPtr_x += jj%2;
              }
          }

          std::stringstream ss;
          ss<<"SPH-Coeff0: ";
          for(int ii=0; ii<= p; ++ii){
            ss<<-(int)(0.5-10*log(sqrt(coeff_norm0[ii]))/log(10.0))*0.1<<' ';
          }
          ss<<'\n';
          INFO(ss.str());
      }
      recycle(x);
    }

    if (params_.rep_upsample)
        Resample(Surf->getPosition(), sht_upsample_, sht_, *u1, *u2,
            S_.getPositionModifiable());

    // begin for collision
    // project u1 to collision free
    INFO("Begin Project reparam direction to without contact.");
    /*
    if (params_.rep_upsample)
    {
        axpy(static_cast<value_type>(-0.002), Surf->getNormal(), Surf->getPosition(), Surf->getPositionModifiable());
        Resample(Surf->getPositionModifiable(), sht_upsample_, sht_, *u1, *u2, *x_old);
    }
    else
    {
        axpy(static_cast<value_type>(-0.002), Surf->getNormal(), Surf->getPosition(), *x_old);
    }
    */
    axpy(static_cast<value_type>(-1.0), *x_old, S_.getPosition(), *u1_down);
    ParallelRemoveContactSimple(*u1_down, *x_old);
    u1_down->getDevice().Memcpy(S_.getPositionModifiable().begin(), u1_down->begin(), 
            u1_down->size()*sizeof(value_type), device_type::MemcpyDeviceToDevice);
    INFO("End Project reparam direction to without contact.");
    // end for collision

    recycle(twrk1);
    recycle(twrk2);
    recycle(u1);
    recycle(u2);
    recycle(x_old_up);
    recycle(x_old);
    recycle(u1_down);
    recycle(wrk);
    recycle(tencpy);
    PROFILEEND("",0);

    return ErrorEvent::Success;
}

template<typename SurfContainer, typename Interaction>
std::auto_ptr<typename SurfContainer::Sca_t> InterfacialVelocity<SurfContainer, Interaction>::
checkoutSca() const
{
    std::auto_ptr<Sca_t> scp;

    if(scalar_work_q_.empty())
        scp = static_cast<std::auto_ptr<Sca_t> >(new Sca_t);
    else
    {
        scp = static_cast<std::auto_ptr<Sca_t> >(scalar_work_q_.front());
        scalar_work_q_.pop();
    }

    scp->replicate(S_.getPosition());
    ++checked_out_work_sca_;
    return(scp);
}
template<typename SurfContainer, typename Interaction>
void InterfacialVelocity<SurfContainer, Interaction>::
recycle(std::auto_ptr<Sca_t> scp) const
{
    scalar_work_q_.push(scp.release());
    --checked_out_work_sca_;
}

template<typename SurfContainer, typename Interaction>
std::auto_ptr<typename SurfContainer::Vec_t> InterfacialVelocity<SurfContainer, Interaction>::
checkoutVec() const
{
    std::auto_ptr<Vec_t> vcp;

    if(vector_work_q_.empty())
        vcp = static_cast<std::auto_ptr<Vec_t> >(new Vec_t);
    else
    {
        vcp = static_cast<std::auto_ptr<Vec_t> >(vector_work_q_.front());
        vector_work_q_.pop();
    }

    vcp->replicate(S_.getPosition());
    ++checked_out_work_vec_;

    return(vcp);
}

template<typename SurfContainer, typename Interaction>
void InterfacialVelocity<SurfContainer, Interaction>::
recycle(std::auto_ptr<Vec_t> vcp) const
{
    vector_work_q_.push(vcp.release());
    --checked_out_work_vec_;
}

template<typename SurfContainer, typename Interaction>
void InterfacialVelocity<SurfContainer, Interaction>::
purgeTheWorkSpace() const
{
    while ( !scalar_work_q_.empty() )
    {
         delete scalar_work_q_.front();
        scalar_work_q_.pop();
    }

    while ( !vector_work_q_.empty() )
    {
        delete vector_work_q_.front();
        vector_work_q_.pop();
    }
}

template<typename SurfContainer, typename Interaction>
void InterfacialVelocity<SurfContainer, Interaction>::
ShuffleVector(const Vec_t &u_in, Vec_t &u_out, int mode)
{
    assert(u_in.size() == u_out.size());
    size_t stride = u_in.getStride();
    size_t num_sub_funcs = u_in.getNumSubFuncs();
    size_t num_subs = u_in.getNumSubs();
    if(mode == 1)
    {
        #pragma omp parallel for
        for(size_t ss=0; ss<num_sub_funcs; ss++)
        {
            size_t i = ss%3;
            size_t j = ss/3;
            u_out.getDevice().Memcpy(&(u_out.begin()[i*stride*num_subs+j*stride]), &(u_in.begin()[ss*stride]), 
                    stride*sizeof(value_type), device_type::MemcpyDeviceToDevice);
        }
    }
    else
    {
        #pragma omp parallel for
        for(size_t ss=0; ss<num_sub_funcs; ss++)
        {
            size_t i = ss%3;
            size_t j = ss/3;
            u_out.getDevice().Memcpy(&(u_out.begin()[ss*stride]), &(u_in.begin()[i*stride*num_subs+j*stride]), 
                    stride*sizeof(value_type), device_type::MemcpyDeviceToDevice);
        }
    }

}
