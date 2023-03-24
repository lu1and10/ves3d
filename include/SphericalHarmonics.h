#ifndef _SPHERICAL_HARMONICS_H_
#define _SPHERICAL_HARMONICS_H_

//#include <matrix.hpp>
#include "sctl.hpp"
#define SHMAXDEG 256

template <class Real>
class SphericalHarmonics{

  public:

    static void SHC2Grid(const sctl::Vector<Real>& S, long p0, long p1, sctl::Vector<Real>& X, sctl::Vector<Real>* X_theta=NULL, sctl::Vector<Real>* X_phi=NULL);

    static void Grid2SHC(const sctl::Vector<Real>& X, long p0, long p1, sctl::Vector<Real>& S);

    static void SHC2GridTranspose(const sctl::Vector<Real>& X, long p0, long p1, sctl::Vector<Real>& S);

    static void SHC2Pole(const sctl::Vector<Real>& S, long p0, sctl::Vector<Real>& P);

    static void RotateAll(const sctl::Vector<Real>& S, long p0, long dof, sctl::Vector<Real>& S_);

    static void RotateTranspose(const sctl::Vector<Real>& S_, long p0, long dof, sctl::Vector<Real>& S);

    static sctl::Vector<Real>& LegendreNodes(long p1);

    static sctl::Vector<Real>& LegendreWeights(long p1);

    static sctl::Vector<Real>& SingularWeights(long p1);

    static sctl::Matrix<Real>& MatFourier(long p0, long p1);

    static sctl::Matrix<Real>& MatFourierInv(long p0, long p1);

    static sctl::Matrix<Real>& MatFourierGrad(long p0, long p1);

    static std::vector<sctl::Matrix<Real> >& MatLegendre(long p0, long p1);

    static std::vector<sctl::Matrix<Real> >& MatLegendreInv(long p0, long p1);

    static std::vector<sctl::Matrix<Real> >& MatLegendreGrad(long p0, long p1);

    static std::vector<sctl::Matrix<Real> >& MatRotate(long p0);

    static void StokesSingularInteg(const sctl::Vector<Real>& S, long p0, long p1, sctl::Vector<Real>* SLMatrix=NULL, sctl::Vector<Real>* DLMatrix=NULL);

  private:

    /**
     * \brief Computes all the Associated Legendre Polynomials (normalized) upto the specified degree.
     * \param[in] degree The degree upto which the legendre polynomials have to be computed.
     * \param[in] X The input values for which the polynomials have to be computed.
     * \param[in] N The number of input points.
     * \param[out] poly_val The output array of size (degree+1)*(degree+2)*N/2 containing the computed polynomial values.
     * The output values are in the order:
     * P(n,m)[i] => {P(0,0)[0], P(0,0)[1], ..., P(0,0)[N-1], P(1,0)[0], ..., P(1,0)[N-1],
     * P(2,0)[0], ..., P(degree,0)[N-1], P(1,1)[0], ...,P(2,1)[0], ..., P(degree,degree)[N-1]}
     */
    static void LegPoly(Real* poly_val, const Real* X, long N, long degree);

    static void LegPolyDeriv(Real* poly_val, const Real* X, long N, long degree);

    template <bool SLayer, bool DLayer>
    static void StokesSingularInteg_(const sctl::Vector<Real>& X0, long p0, long p1, sctl::Vector<Real>& SL, sctl::Vector<Real>& DL);

    static struct MatrixStorage{
      MatrixStorage(int size){
        Qx_ .resize(size);
        Qw_ .resize(size);
        Sw_ .resize(size);
        Mf_ .resize(size*size);
        Mdf_.resize(size*size);
        Ml_ .resize(size*size);
        Mdl_.resize(size*size);
        Mr_ .resize(size);
        Mfinv_ .resize(size*size);
        Mlinv_ .resize(size*size);
      }
      std::vector<sctl::Vector<Real> > Qx_;
      std::vector<sctl::Vector<Real> > Qw_;
      std::vector<sctl::Vector<Real> > Sw_;
      std::vector<sctl::Matrix<Real> > Mf_ ;
      std::vector<sctl::Matrix<Real> > Mdf_;
      std::vector<std::vector<sctl::Matrix<Real> > > Ml_ ;
      std::vector<std::vector<sctl::Matrix<Real> > > Mdl_;
      std::vector<std::vector<sctl::Matrix<Real> > > Mr_;
      std::vector<sctl::Matrix<Real> > Mfinv_ ;
      std::vector<std::vector<sctl::Matrix<Real> > > Mlinv_ ;
    } matrix;

};

template<> SphericalHarmonics<double>::MatrixStorage SphericalHarmonics<double>::matrix(SHMAXDEG);
template<> SphericalHarmonics<float >::MatrixStorage SphericalHarmonics<float >::matrix(SHMAXDEG);

#include "SphericalHarmonics.cc"

#endif // _SPHERICAL_HARMONICS_H_

