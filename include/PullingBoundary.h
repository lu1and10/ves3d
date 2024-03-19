#ifndef _PULLINGBOUNDARY_H_
#define _PULLINGBOUNDARY_H_

#include "Parameters.h"
#include "DataIO.h"
#include "OperatorsMats.h"
#include "GLIntegrator.h"

template<typename SurfContainer>
class PullingBoundary
{
  public:
      typedef typename SurfContainer::value_type value_type;
      typedef typename SurfContainer::device_type device_type;
      typedef typename SurfContainer::Arr_t Arr_t;
      typedef typename SurfContainer::Sca_t Sca_t;
      typedef typename SurfContainer::Vec_t Vec_t;
      typedef Parameters<value_type> Params_t;
      typedef OperatorsMats<Arr_t> Mats_t;

      PullingBoundary(const Params_t &params, const Mats_t &mats);
      ~PullingBoundary();

      void Solve();
      void GetCentrosomePulling(const value_type* centrosome_position, const value_type* centrosome_velocity, Vec_t *Fcpull, value_type *min_dist);

      SurfContainer* S_;

      GaussLegendreIntegrator<Sca_t> integrator_;

      Vec_t solved_density_;
      Vec_t Fpull_;
      Sca_t binding_probability_;
      Sca_t impingement_rate_;
      value_type M0_;
      value_type area_;
      const Params_t &params_;
  private:
};

#include "PullingBoundary.cc"

#endif // _PULLINGBOUNDARY_H_
