template<typename lhsContainer, typename rhsContainer>
inline bool AreCompatible(const lhsContainer &lhs, const  rhsContainer &rhs)
{
    return(lhs.getDevice() == rhs.getDevice()
        && lhs.getStride() == rhs.getStride()
        && lhs.getNumSubs() == rhs.getNumSubs());
}

template<typename ScalarContainer>
void Sqrt(const ScalarContainer &x_in, ScalarContainer &sqrt_out)
{
    ASSERT(AreCompatible(x_in, sqrt_out),"Incompatible containers");

    x_in.getDevice().Sqrt(x_in.begin(), x_in.size(), sqrt_out.begin());
}

template<typename ScalarContainer>
void Exp(const ScalarContainer &x_in, ScalarContainer &exp_out)
{
    ASSERT(AreCompatible(x_in, exp_out),"Incompatible containers");

    x_in.getDevice().Exp(x_in.begin(), x_in.size(), exp_out.begin());
}

template<typename ScalarContainer>
void xy(const ScalarContainer &x_in, const ScalarContainer &y_in,
    ScalarContainer &xy_out)
{
    ASSERT(AreCompatible(x_in,y_in),"Incompatible containers");
    ASSERT(AreCompatible(y_in,xy_out),"Incompatible containers");

    x_in.getDevice().xy(x_in.begin(), y_in.begin(), x_in.size(), xy_out.begin());
}

template<typename ScalarContainer>
void xyInv(const ScalarContainer &x_in, const ScalarContainer &y_in,
    ScalarContainer &xyInv_out)
{
    ASSERT(AreCompatible(x_in,y_in),"Incompatible containers");
    ASSERT(AreCompatible(y_in,xyInv_out),"Incompatible containers");

    x_in.getDevice().xyInv(x_in.begin(), y_in.begin(), x_in.size(),
        xyInv_out.begin());
}

template<typename ScalarContainer>
void xInv(const ScalarContainer &x_in, ScalarContainer &xInv_out)
{
    ASSERT(AreCompatible(x_in,xInv_out),"Incompatible containers");

    x_in.getDevice().xyInv(static_cast<typename
        ScalarContainer::value_type*>(NULL), x_in.begin(),
        x_in.size(), xInv_out.begin());
}

template<typename ScalarContainer>
void axpy(typename ScalarContainer::value_type a_in,
    const ScalarContainer &x_in, const ScalarContainer &y_in,
    ScalarContainer &axpy_out)
{
    ASSERT(AreCompatible(x_in,y_in),"Incompatible containers");
    ASSERT(AreCompatible(y_in,axpy_out),"Incompatible containers");

    x_in.getDevice().axpy(a_in, x_in.begin(), y_in.begin(), x_in.size(),
        axpy_out.begin());
}

template<typename ScalarContainer>
void axpy(typename ScalarContainer::value_type a_in,
    const ScalarContainer &x_in, ScalarContainer &axpy_out)
{
    ASSERT(AreCompatible(x_in,axpy_out),"Incompatible containers");

    x_in.getDevice().axpy(
        a_in, x_in.begin(), (typename ScalarContainer::value_type*)NULL, x_in.size(), axpy_out.begin());
}


template<typename ScalarContainer, typename IntegrandContainer>
void Reduce(const IntegrandContainer &x_in,
    const ScalarContainer &w_in, const ScalarContainer &quad_w_in,
    IntegrandContainer &x_dw)
{
    ASSERT(AreCompatible(x_in,w_in),"Incompatible containers");
    ASSERT(quad_w_in.getStride() == w_in.getStride(),
        "Incompatible containers");
    ASSERT(quad_w_in.getNumSubFuncs() >= 1,"Incompatible containers");

    COUTDEBUG("x=(dim:"<<
        x_in.getTheDim()<<", numsubs:"
        <<x_in.getNumSubs()<<", subfunc:"
        <<x_in.getNumSubFuncs()<<", stride:"
        <<x_in.getStride()<<", sublength:"
        <<x_in.getSubLength()<<", size:"
        <<x_in.size()<<")"
              );

    COUTDEBUG("w=(dim:"<<
        w_in.getTheDim()<<", numsubs:"
        <<w_in.getNumSubs()<<", subfunc:"
        <<w_in.getNumSubFuncs()<<", stride:"
        <<w_in.getStride()<<", sublength:"
        <<w_in.getSubLength()<<", size:"
        <<w_in.size()<<")"
              );

    COUTDEBUG("quad.size="<<quad_w_in.size()<<", x_dw.size="<<x_dw.size());

    x_in.getDevice().Reduce(x_in.begin(),
        x_in.getTheDim(),
        w_in.begin(),
        quad_w_in.begin(),
        x_in.getStride(),
        x_in.getNumSubs(),
        x_dw.begin());
}

template<typename Container>
void Reduce(const Container &w_in, const Container &quad_w_in,
    Container &dw)
{
    ASSERT(quad_w_in.getStride() == w_in.getStride(),"Incompatible containers");
    ASSERT(quad_w_in.getNumSubFuncs() >= 1,"Incompatible containers");

    w_in.getDevice().Reduce(static_cast<typename Container::value_type* >(0),
        0, w_in.begin(), quad_w_in.begin(), w_in.getStride(),
        w_in.getNumSubs(), dw.begin());
}

template<typename ScalarContainer>
typename ScalarContainer::value_type Max(const ScalarContainer &x_in)
{
    return(x_in.getDevice().Max(x_in.begin(),
            x_in.getStride() * x_in.getNumSubFuncs()));
}

template<typename Container>
inline void fillRand(Container &x_in){
    x_in.getDevice().fillRand(x_in.begin(),x_in.size());
}


template<typename ScalarContainer, typename VectorContainer>
inline void GeometricDot(const VectorContainer &u_in,
    const VectorContainer &v_in, ScalarContainer &x_out)
{
    ASSERT(AreCompatible(u_in,v_in),"Incompatible containers");
    ASSERT(AreCompatible(v_in,x_out),"Incompatible containers");

    COUTDEBUG("u("<<u_in.getNumSubFuncs()
        <<","<<u_in.getStride()<<")"
        <<", v=("<<v_in.getNumSubFuncs()<<","<<v_in.getStride()<<")"
        ", x=("<<x_out.getNumSubFuncs()<<","<<x_out.getStride()<<")");

    u_in.getDevice().DotProduct(u_in.begin(), v_in.begin(),
        u_in.getStride(), u_in.getNumSubs(), x_out.begin());
}

template<typename VectorContainer>
inline void GeometricCross(const VectorContainer &u_in,
    const VectorContainer &v_in, VectorContainer &w_out)
{
    ASSERT(AreCompatible(u_in,v_in),"Incompatible containers");
    ASSERT(AreCompatible(v_in,w_out),"Incompatible containers");

    u_in.getDevice().CrossProduct(u_in.begin(), v_in.begin(),
        u_in.getStride(), u_in.getNumSubs(), w_out.begin());
}

template<typename ScalarContainer, typename VectorContainer>
inline void uyInv(const VectorContainer &u_in,
    const ScalarContainer &y_in, VectorContainer &uyInv_out)
{
    ASSERT(AreCompatible(u_in, y_in),"Incompatible containers");
    ASSERT(AreCompatible(y_in,uyInv_out),"Incompatible containers");

    u_in.getDevice().uyInv(u_in.begin(), y_in.begin(),
        u_in.getStride(), u_in.getNumSubs(), uyInv_out.begin());
}

template<typename ScalarContainer, typename VectorContainer>
inline void avpw(const ScalarContainer &a_in,
    const VectorContainer &v_in,  const VectorContainer &w_in,
    VectorContainer &avpw_out)
{
    ASSERT(a_in.getNumSubs() == v_in.getNumSubs(),"Incompatible containers");
    ASSERT(AreCompatible(v_in,w_in),"Incompatible containers");
    ASSERT(AreCompatible(w_in,avpw_out),"Incompatible containers");

    CERR_LOC("THIS NEEDS TO BE IMPLEMENTED",std::endl, exit(1));
    //u_in.getDevice().avpw(a_in.begin(), v_in.begin(),
    //    w_in.begin(), v_in.getStride(), u_in.getNumSubs(), avpw_out.begin());
}

template<typename Arr_t, typename Vec_t>
inline void av(const Arr_t &a_in, const Vec_t &v_in, Vec_t &av_out)
{
    ASSERT(a_in.size() == v_in.getNumSubs(),"Incompatible containers");
    ASSERT(AreCompatible(v_in,av_out),"Incompatible containers");

    Arr_t::getDevice().avpw(a_in.begin(),
					       v_in.begin(),
					       (typename Arr_t::value_type*)NULL,
					       v_in.getStride(),
					       v_in.getNumSubs(),
					       av_out.begin());
}

template<typename ScalarContainer, typename VectorContainer>
inline void xvpw(const ScalarContainer &x_in,
    const VectorContainer &v_in, const VectorContainer &w_in,
    VectorContainer &xvpw_out)
{
    ASSERT(AreCompatible(x_in,v_in),"Incompatible containers");
    ASSERT(AreCompatible(v_in,w_in),"Incompatible containers");
    ASSERT(AreCompatible(w_in,xvpw_out),"Incompatible containers");

    x_in.getDevice().xvpw(x_in.begin(), v_in.begin(),
        w_in.begin(), v_in.getStride(), v_in.getNumSubs(), xvpw_out.begin());
}

template<typename Container>
inline void ax(const Container& a, const Container& x, Container& ax_out)
{
    ASSERT( a.getStride() == x.getStride() ,"Incompatible containers");
    ASSERT( AreCompatible(x, ax_out) ,"Incompatible containers");
    Container::getDevice().ax(a.begin(), x.begin(), x.getStride(),
        x.getNumSubFuncs(), ax_out.begin());
}

template<typename ScalarContainer, typename VectorContainer>
inline void xv(const ScalarContainer &x_in,
    const VectorContainer &v_in, VectorContainer &xv_out)
{
    ASSERT(AreCompatible(x_in,v_in),"Incompatible containers");
    ASSERT(AreCompatible(v_in,xv_out),"Incompatible containers");

    x_in.getDevice().xvpw(
        x_in.begin(), v_in.begin(), (typename ScalarContainer::value_type*)NULL, v_in.getStride(),
        v_in.getNumSubs(), xv_out.begin());
}

template<typename VectorContainer>
inline void ShufflePoints(const VectorContainer &u_in,
    VectorContainer &u_out)
{
    ASSERT(AreCompatible(u_in,u_out),"Incompatible containers");
    size_t stride = u_in.getStride();

    int dim = u_in.getTheDim();

    size_t dim1 = (u_in.getPointOrder() == AxisMajor) ? dim : stride;
    size_t dim2 = (u_in.getPointOrder() == AxisMajor) ? stride : dim;

    for(size_t ss=0; ss<u_in.getNumSubs(); ++ss)
        u_in.getDevice().Transpose(u_in.begin() + dim * stride *ss
            , dim1, dim2, u_out.begin() + dim * stride * ss);

    u_out.setPointOrder((u_in.getPointOrder() == AxisMajor) ? PointMajor : AxisMajor);
}

///@todo This need a new data structure for the matrix, etc
template<typename ScalarContainer>
inline void CircShift(const typename ScalarContainer::value_type *x_in,
    int shift, ScalarContainer &x_out)
{
    typedef typename ScalarContainer::value_type VT;
    typedef typename ScalarContainer::device_type DT;

    int vec_length = x_out.getStride();
    int n_vecs = x_out.getNumSubs();
    shift = shift % vec_length;
    shift += (shift < 0) ?  vec_length : 0;

    int base_in, base_out;
    for (int ii = 0; ii < n_vecs; ii++) {
        base_out = ii * vec_length;
        base_in = base_out + vec_length - shift;
        x_out.getDevice().Memcpy(
            (VT*) x_out.begin() + base_out,
            x_in + base_in,
            sizeof(VT) * shift,
            DT::MemcpyDeviceToDevice);
        base_in = base_out;
        base_out += shift;
        x_out.getDevice().Memcpy(
            (VT*)
            x_out.begin() + base_out,
            x_in + base_in,
            sizeof(VT) * (vec_length - shift),
            DT::MemcpyDeviceToDevice);
    }
}

template<typename VectorContainer, typename CentCont>
inline void Populate(VectorContainer &x, const CentCont &centers)
{
    typedef typename VectorContainer::value_type VT;
    typedef typename VectorContainer::device_type DT;

    size_t cpysize = x.getSubLength();
    cpysize *=sizeof(typename VectorContainer::value_type);

    if ( cpysize == 0)
        return;

    for(int ii=1; ii<x.getNumSubs(); ++ii)
        x.getDevice().Memcpy(
            x.getSubN_begin(ii),
            x.begin(),
            cpysize,
            DT::MemcpyDeviceToDevice);

    x.getDevice().apx(
        centers.begin(),
        x.begin(),
        x.getStride(),
        x.getNumSubFuncs(),
        x.begin());
}

template<typename val_t, typename E>
void _rotation_matrix_zyz(val_t rot_z1, val_t rot_y, val_t rot_z2, E &rot)
{
    /* utility function for initializaiton */
    val_t R1[] = { cos(rot_z1),-sin(rot_z1), 0.0,
                   sin(rot_z1), cos(rot_z1), 0.0,
                   0.0        , 0.0        , 1.0};
    val_t R2[] = { cos(rot_y) , 0.0        , sin(rot_y),
                   0.0        , 1.0        , 0.0,
                  -sin(rot_y) , 0.0        , cos(rot_y)};
    val_t R3[] = { cos(rot_z2),-sin(rot_z2), 0.0,
                   sin(rot_z2), cos(rot_z2), 0.0,
                   0.0        , 0.0        , 1.0};

    val_t Ri[VES3D_DIM*VES3D_DIM], R[VES3D_DIM*VES3D_DIM];

    /* these can be coded in, but leaving as is for debugging */
    for (int i(0);i<VES3D_DIM;++i)
        for (int j(0);j<VES3D_DIM;++j){
            Ri[i*VES3D_DIM+j] = 0;
            for (int k(0);k<VES3D_DIM;++k)
                Ri[i*VES3D_DIM+j] += R2[i*VES3D_DIM+k]*R1[k*VES3D_DIM+j];
        }
    for (int i(0);i<VES3D_DIM;++i)
        for (int j(0);j<VES3D_DIM;++j){
            R[i*VES3D_DIM+j] = 0;
            for (int k(0);k<VES3D_DIM;++k)
                R[i*VES3D_DIM+j] += R3[i*VES3D_DIM+k]*Ri[k*VES3D_DIM+j];
        }

    for (int i(0);i<VES3D_DIM*VES3D_DIM;++i)
        rot.push_back(R[i]);
}

template<typename val_t>
void _transform_point(val_t &x, val_t &y, val_t &z,
    val_t scale, const val_t *rot, const val_t *cen)
{
    /* utility function for initializaiton */
    val_t xyz[] = {x*scale, y*scale, z*scale};
    val_t XYZ[] = {0,0,0};

    for (int i(0);i<VES3D_DIM;++i)
        for (int j(0);j<VES3D_DIM;++j)
            XYZ[i] += rot[i*VES3D_DIM+j]*xyz[j];

    x = XYZ[0] + cen[0];
    y = XYZ[1] + cen[1];
    z = XYZ[2] + cen[2];
}

template<typename Vec_t, typename E>
inline void InitializeShapes(Vec_t &x, const E &shape_gallery,
    const E &geo_spec)
{
    typedef typename E::value_type val_t;
    typedef typename E::size_type  sz_t;
    typedef typename Vec_t::device_type  device_type;

    int nves(x.getNumSubs());
    int stride(x.getStride());

    // extract info
    int nfields(8);
    E shape, scale, cen, rot;
    for (int iV(0); iV<nves; ++iV){
        sz_t offset(iV*nfields);
        shape.push_back(geo_spec[offset++]);
        scale.push_back(geo_spec[offset++]);

        for (int iC(0); iC<VES3D_DIM; ++iC)
            cen.push_back(geo_spec[offset++]);

        _rotation_matrix_zyz(
            geo_spec[offset++] /* rz1 */,
            geo_spec[offset++] /* ry  */,
            geo_spec[offset++] /* rz2 */,
            rot);
    }

    // transform and fill
    E xi(nves*stride*VES3D_DIM);
    for (int iV(0); iV<nves; ++iV) {
        sz_t shape_offset(shape[iV]*VES3D_DIM*stride);
        sz_t ves_offset  (iV       *VES3D_DIM*stride);

        val_t *R = &(rot[iV*VES3D_DIM*VES3D_DIM]);
        val_t *C = &(cen[iV*VES3D_DIM    ]);

        for (int iP(0);iP<stride;++iP) {
            val_t x = shape_gallery[shape_offset+         iP];
            val_t y = shape_gallery[shape_offset+  stride+iP];
            val_t z = shape_gallery[shape_offset+2*stride+iP];
            _transform_point(x,y,z,scale[iV],R,C);

            xi[ves_offset+         iP] = x;
            xi[ves_offset+  stride+iP] = y;
            xi[ves_offset+2*stride+iP] = z;
        }
    }

    // copy
    ASSERT(x.size()==xi.size(), "incorrect initialization");
    x.getDevice().Memcpy(x.begin(),
        &(xi[0]),
        x.size()*sizeof(val_t),
        device_type::MemcpyHostToDevice);
}

template<typename ScalarContainer>
typename ScalarContainer::value_type AlgebraicDot(const ScalarContainer &x,
    const ScalarContainer &y)
{
    return(ScalarContainer::getDevice().AlgebraicDot(x.begin(), y.begin(), x.size()));
}

template<typename Container>
typename Container::value_type MaxAbs(Container &x)
{
    return(x.getDevice().MaxAbs(x.begin(), x.size()));
}

template<typename Container>
typename Container::value_type MinAbs(Container &x)
{
    return(x.getDevice().MinAbs(x.begin(), x.size()));
}

template<typename Container, typename SHT>
void Resample(const Container &xp, const SHT &shtp, const SHT &shtq,
    Container &shcpq, Container &wrkpq, Container &xq)
{
    typedef typename Container::value_type value_type;
    typedef typename Container::device_type DT;

    int p = shtp.getShOrder();
    int q = shtq.getShOrder();

    COUTDEBUG("resample from "<<p<<" to "<<q);
    if(p == q)
    {
        COUTDEBUG("p==q, no need to resample");
        Container::getDevice().Memcpy(xq.begin(), xp.begin(),
            xp.size() * sizeof(value_type), DT::MemcpyDeviceToDevice);
        return;
    }

    value_type *shc_p, *shc_q;

    shtp.forward(xp, shcpq, wrkpq);
    shc_p = wrkpq.begin();
    shc_q = shcpq.begin();

    if(p < q){
        COUTDEBUG("q>p filling with zeros");
        Container::getDevice().Memset(shcpq.begin(), 0,
            shcpq.size() * sizeof(value_type));
    }

    int len_p, len_q, cpy_len;
    int minfreq = ( p > q ) ? q : p;
    int n_funs = xp.getNumSubFuncs();
    COUTDEBUG("copying each coeff set (n_fun="
        <<n_funs<<")");

    for(int ii=0; ii< 2 * minfreq; ++ii)
    {
        len_p   = p  + 1 - (ii+1)/2;
        len_q   = q  + 1 - (ii+1)/2;
        cpy_len = minfreq + 1 - (ii+1)/2;

        COUTDEBUG("copying coeffs of same degree "<<ii
            <<" advancing p by "<<len_p
            <<" advancing q by "<<len_q);
        for(int jj = 0; jj<n_funs; ++jj)
        {
            Container::getDevice().Memcpy(shc_q, shc_p,
                cpy_len * sizeof(value_type), DT::MemcpyDeviceToDevice);

            shc_p += len_p;
            shc_q += len_q;
        }
    }

    shtq.backward(shcpq, wrkpq, xq);
}

template<typename Container>
void set_zero(Container &x){
    x.getDevice().Memset(x.begin(), 0, sizeof(typename Container::value_type)*x.size());
}

template<typename Container>
void set_one(Container &x){
    std::fill_n(x.begin(), x.size(), 1);
}

template<typename ScalarContainer, typename VectorContainer>
void set_exp(const VectorContainer &x, const typename ScalarContainer::value_type *x0, ScalarContainer &out){
  // This writes scalar function exp(-||x-x0||) into "out", at nodes "x".

    // store point -x0
    typename ScalarContainer::array_type a_tmp(x.getNumSubFuncs());
    for(int i=0; i<x.getNumSubFuncs(); i++){
        a_tmp.begin()[i] = -x0[i%3];
    }

    // init v_tmp for output of apx
    VectorContainer v_tmp;
    v_tmp.replicate(x);
    set_zero(v_tmp);

    // store x - x0 in v_tmp          (apx does "a plus x" where a is const vec)
    x.getDevice().apx(a_tmp.begin(),x.begin(),x.getStride(),x.getNumSubFuncs(),v_tmp.begin());
    // The point of doing this in an abstracted way is could be CPU or GPU, real/double.

    // init s_tmp
    ScalarContainer s_tmp;
    s_tmp.replicate(x);
    set_zero(s_tmp);

    // store (x - x0) dot (x - x0) in s_tmp
    GeometricDot(v_tmp, v_tmp, s_tmp);

    // store sqrt(s_tmp) in s_tmp
    Sqrt(s_tmp, s_tmp);

    // [maybe exp for ScalarContainer should be added to help functions? or not and use loop]
#pragma omp parallel for
    for(int idx = 0; idx < out.size(); idx++){
      // out.begin() is C++ vec style for 1st element. a[idx] is C-style
      out.begin()[idx] = std::exp(-s_tmp.begin()[idx]);
    }
}

template<typename ScalarContainer, typename VectorContainer>
void set_exp_simple(const VectorContainer &x, const typename ScalarContainer::value_type *x0, ScalarContainer &out){
  // This writes scalar function exp(-||x-x0||) into "out", at nodes "x".
  // Would not be abstracted (not on GPU)!

  int N = out.size();         // # nodes on surf of vesicle
#pragma omp parallel for
    for(int idx = 0; idx < N; idx++){
      typename ScalarContainer::value_type d[3];  // displacement vector x[idx]-x0
      for (int i=0; i<3; i++)
        d[i] = x.begin()[idx+N*i] - x0[i];      // note a flattened array xxx...yyy..zzz
      typename ScalarContainer::value_type d2 = d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
      out.begin()[idx] = std::exp(-std::sqrt(d2));
    }
}

template<typename ScalarContainer>
void chi(const ScalarContainer &x_in, const ScalarContainer &one_in,
    ScalarContainer &chi_out)
{
    ASSERT(AreCompatible(x_in,y_in),"Incompatible containers");
    ASSERT(AreCompatible(y_in,xyInv_out),"Incompatible containers");

    xy(x_in, x_in, chi_out); // chi_out= x_in^2
    axpy(1.0, one_in, chi_out, chi_out); // chi_out = 1 + x_in^2
    Sqrt(chi_out, chi_out); // chi_out = sqrt(1+x_in^2)
    xyInv(one_in, chi_out, chi_out); // chi_out = 1/sqrt(1+x_in^2)
    axpy(-1.0, chi_out, one_in, chi_out); // chi_out = 1 - 1/sqrt(1+x_in^2)
    axpy(0.5, chi_out, chi_out); // chi_out = 0.5*(1 - 1/sqrt(1+x_in^2))
}
