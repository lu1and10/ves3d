#ifndef _PULLINGBOUNDARY_H_
#define _PULLINGBOUNDARY_H_


//template<typename SurfContainer, typename Interaction>
class PullingBoundary
{
  public:
      PullingBoundary();
      ~PullingBoundary();
      void EvalPotential(int num_target_points, double* target_address, double* target_potential);
      void Solve();

      PatchSurfFaceMap* surface;
      SolverGMRESDoubleLayer* solver;
      MPI_Comm comm;

      Vec solved_density;
      Vec solved_density_tmp;
      Vec boundary_data;
      Vec boundary_flow;
      Vec boundary_flow_tmp;

      double* GetSamplePoints(int& num_sample_points);
      void RestoreSamplePoints(double **local_sample_points);
      void SetBoundaryData(double* boundary_data_address);
      void SetTriData();
      void SetBoundaryFlow();
      void BuildInOutLets();
      void FillVesicle(double cell_size);
      void LoadDensity(bool flag);
      void SaveDensity();

  private:
};

#include "PullingBoundary.cc"

#endif // _PULLINGBOUNDARY_H_
