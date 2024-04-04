template<typename ScalarContainer, typename VectorContainer>
void write_vtk(const VectorContainer &X, const VectorContainer &v, const ScalarContainer &s, const std::string &filename){
    // test write vector field

    char* cstr = const_cast<char*>(filename.c_str());

    // sh order
    int p0 = X.getShOrder();
    typedef sctl::Vector<typename VectorContainer::value_type> SCTL_Vec;

    SCTL_Vec Xgrid(X.size(), (typename VectorContainer::value_type*)X.begin(), false);
    SCTL_Vec Xgrid_shc;
    sctl::SphericalHarmonics<typename VectorContainer::value_type>::Grid2SHC(Xgrid, p0+1, 2*p0, p0, Xgrid_shc, sctl::SHCArrange::ROW_MAJOR);

    SCTL_Vec V(v.size(), (typename VectorContainer::value_type*)v.begin(), false);
    SCTL_Vec V_shc;
    sctl::SphericalHarmonics<typename VectorContainer::value_type>::Grid2SHC(V, p0+1, 2*p0, p0, V_shc, sctl::SHCArrange::ROW_MAJOR);

    sctl::SphericalHarmonics<typename VectorContainer::value_type>::WriteVTK(cstr, &Xgrid_shc, &V_shc, sctl::SHCArrange::ROW_MAJOR, p0, 2*p0);

}



template<typename EvolveSurface>
MonitorBase<EvolveSurface>::~MonitorBase()
{}

/////////////////////////////////////////////////////////////////////////////////////
///@todo move the buffer size to the parameters
template<typename EvolveSurface>
Monitor<EvolveSurface>::Monitor(const Parameters<value_type> *params) :
    checkpoint_flag_(params->checkpoint),
    checkpoint_stride_(params->checkpoint_stride),
    A0_(-1),
    V0_(-1),
    last_checkpoint_(-1),
    time_idx_(-1),
    params_(params)
{}

template<typename EvolveSurface>
Monitor<EvolveSurface>::~Monitor()
{}

template<typename EvolveSurface>
Error_t Monitor<EvolveSurface>::operator()(const EvolveSurface *state,
    const value_type &t, value_type &dt)
{
    const typename EvolveSurface::Sca_t::device_type& device=EvolveSurface::Sca_t::getDevice();
    size_t N_ves=state->S_->getPosition().getNumSubs();

    v_work.replicate(state->S_->getPosition());
    state->S_->grad(state->F_->tension_, v_work);

    area_new.replicate(state->S_->getPosition());
    vol_new .replicate(state->S_->getPosition());
    state->S_->area  (area_new);
    state->S_->volume( vol_new);
    value_type v0_new = vol_new.begin()[0];

    if(A0_ < 0){ // Initialize area0_, vol0_
        area0_.replicate(state->S_->getPosition());
        vol0_ .replicate(state->S_->getPosition());
        device.Memcpy(area0_.begin(), area_new.begin(), N_ves*sizeof(value_type), device.MemcpyDeviceToDevice);
        device.Memcpy( vol0_.begin(),  vol_new.begin(), N_ves*sizeof(value_type), device.MemcpyDeviceToDevice);
        A0_=(device.MaxAbs(area0_.begin(), N_ves));
        V0_=(device.MaxAbs( vol0_.begin(), N_ves));
    }

    device.axpy(-1.0, area0_.begin(), area_new.begin(), N_ves, area_new.begin());
    device.axpy(-1.0,  vol0_.begin(),  vol_new.begin(), N_ves,  vol_new.begin());
    value_type DA(device.MaxAbs(area_new.begin(), N_ves));
    value_type DV(device.MaxAbs( vol_new.begin(), N_ves));

#pragma omp critical (monitor)
    {
        INFO(emph<<"Monitor: thread = "<<omp_get_thread_num()<<"/"<<omp_get_num_threads()
             <<", progress = "<<static_cast<int>(100*t/params_->time_horizon)<<"%"
             <<", t = "<<SCI_PRINT_FRMT<<t
             <<", dt = "<<SCI_PRINT_FRMT<<dt
             <<", bdry_centrosome_pulling = "<<SCI_PRINT_FRMT<<state->F_->bdry_centrosome_pulling_.begin()[0]<<" "<<state->F_->bdry_centrosome_pulling_.begin()[1]<<" "<<state->F_->bdry_centrosome_pulling_.begin()[2]
             <<", centrosome_pulling = "<<SCI_PRINT_FRMT<<state->F_->centrosome_pulling_.begin()[0]<<" "<<state->F_->centrosome_pulling_.begin()[1]<<" "<<state->F_->centrosome_pulling_.begin()[2]
             <<", centrosome_pushing = "<<SCI_PRINT_FRMT<<state->F_->centrosome_pushing_.begin()[0]<<" "<<state->F_->centrosome_pushing_.begin()[1]<<" "<<state->F_->centrosome_pushing_.begin()[2]
             <<", centrosome_position = "<<SCI_PRINT_FRMT<<state->F_->centrosome_pos_[0]<<" "<<state->F_->centrosome_pos_[1]<<" "<<state->F_->centrosome_pos_[2]
             <<", centrosome_velocity = "<<SCI_PRINT_FRMT<<state->F_->centrosome_vel_[0]<<" "<<state->F_->centrosome_vel_[1]<<" "<<state->F_->centrosome_vel_[2]
             <<", area error = "<<SCI_PRINT_FRMT<<(DA/A0_)
             <<", volume error = "<<SCI_PRINT_FRMT<<(DV/V0_)
             <<", volume = "<<SCI_PRINT_FRMT<<v0_new<<emph);
        INFO(emph<<"Monitor: mass integral before: " << std::setprecision(8) << state->mass_before_
             <<", mass integral after: " << std::setprecision(8) << state->mass_after_
             <<", rel err in mass cons: " << std::setprecision(3) << abs(state->mass_after_-state->mass_before_)/abs(state->mass_before_)
             <<", binding prob integral: " << std::setprecision(8) << state->int_binding_
             <<", contact area: " << std::setprecision(8) << state->contact_area_
             <<", min distance: " << std::setprecision(8) << state->F_->min_dist_
             <<", boundary min distance: " << std::setprecision(8) << state->F_->bdry_min_dist_
             <<", max C: " << std::setprecision(8) << Max(state->F_->density_)
             <<", min C: " << std::setprecision(8) << Min(state->F_->density_)
             <<", max mean curvature: " << std::setprecision(8) << Max(state->S_->getMeanCurv())
             <<", min mean curvature: " << std::setprecision(8) << Min(state->S_->getMeanCurv())
             <<", max tension: " << std::setprecision(8) << Max(state->F_->tension_)
             <<", min tension: " << std::setprecision(8) << Min(state->F_->tension_)
             <<", max flux: " << std::setprecision(8) << Max(state->F_->flux_) << emph);

        int checkpoint_index(checkpoint_stride_ <= 0 ? last_checkpoint_+1 : t/checkpoint_stride_);

        if ( checkpoint_flag_ && checkpoint_index > last_checkpoint_ )
        {
            ++time_idx_;
            std::string fname(params_->checkpoint_file_name);
            char suffix[7];
            sprintf(suffix, "%06d", time_idx_);
            d_["time_idx"] = std::string(suffix);
            expand_template(&fname, d_);

            //This order of packing is used when loading checkpoints
            //in ves3d_simulation file
            std::stringstream ss;
            ss<<std::scientific<<std::setprecision(16);
            params_->pack(ss, Streamable::ASCII);
            state->pack(ss, Streamable::ASCII);

            INFO("Writing data to file "<<fname);
            IO_.DumpFile(fname.c_str(), ss);
            ++last_checkpoint_;

            if(params_->write_vtk.size()){
                std::string vtkfbase_centrosome(params_->write_vtk);
                std::string centrosome_suffix("centrosome_");
                vtkfbase_centrosome += centrosome_suffix;
                vtkfbase_centrosome += suffix;
                std::string vtkfbase_boundary(params_->write_vtk);
                vtkfbase_boundary += "boundary_";
                vtkfbase_boundary += suffix;
                std::string vtkfbase(params_->write_vtk);
                vtkfbase += suffix;
                INFO("Writing VTK file "<<vtkfbase);
                WriteVTK(*state->S_, vtkfbase.c_str(),
                        {&(state->F_->pulling_force_), &(state->F_->pushing_force_), &(state->F_->bending_force_), &(state->F_->tensile_force_), &(state->F_->flux_), &(state->F_->pos_vel_), &v_work},
                        {"f_pull","f_push","f_bending","f_tensile","flux","membrane_vel","surf_grad_tension"},
                        {&(state->F_->density_), &(state->F_->binding_probability_), &(state->F_->impingement_rate_), &(state->F_->tension_), &(state->S_->contact_indicator_), &(state->S_->concentration_drag_)},
                        {"concentration","binding_prob","impingement_rate","tension","contact_indicator","concentration_drag"},
                        -1, params_->periodic_length, state->F_->centrosome_pos_, 1, vtkfbase_centrosome.c_str());

                WriteVTK(*state->F_->pulling_boundary_.S_, vtkfbase_boundary.c_str(),
                        {&(state->F_->pulling_boundary_.Fpull_)},
                        {"f_pull"},
                        {&(state->F_->pulling_boundary_.binding_probability_),&(state->F_->pulling_boundary_.impingement_rate_),&(state->F_->pulling_boundary_.visible_zone_)},
                        {"binding_prob","impingement_rate","visible_zone"},
                        -1, params_->periodic_length);
            }

#if HAVE_PVFMM
            if(params_->write_vtk.size()){
                std::string vtkfbase(params_->write_vtk);
                vtkfbase += suffix;
                INFO("Writing VTK file");
                WriteVTK(*state->S_,vtkfbase.c_str(), MPI_COMM_WORLD, NULL, -1, params_->periodic_length);
            }
#endif // HAVE_PVFMM
        }
    }

    Error_t return_val(ErrorEvent::Success);
    if ( (DA/A0_) > params_->error_factor  || (DV/V0_) > params_->error_factor )
        return_val = ErrorEvent::AccuracyError;

    return return_val;
}
