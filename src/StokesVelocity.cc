#include <omp.h>
#include <iostream>
#include <SphericalHarmonics.h>

template<class Real>
StokesVelocity<Real>::StokesVelocity(int sh_order_, int sh_order_up_, Real box_size_):
  sh_order(sh_order_), sh_order_up_self(sh_order_up_), sh_order_up(sh_order_up_), box_size(box_size_), trg_is_surf(true)
{
  add_repul=false;
  fmm_setup=false;
}

template <class Real>
StokesVelocity<Real>::~StokesVelocity(){
}


template <class Real>
void StokesVelocity<Real>::SetSrcCoord(const SCTLVec& S, int sh_order_up_self_, int sh_order_up_){
  if(sh_order_up_self_>0) sh_order_up_self=sh_order_up_self_;
  if(sh_order_up_     >0) sh_order_up     =sh_order_up_     ;
  scoord.ReInit(S.Dim(), (Real*)&S[0], true);
  fmm_setup=false;

  SLMatrix.ReInit(0);
  DLMatrix.ReInit(0);

  scoord_far.ReInit(0);
  tcoord_repl.ReInit(0);
  scoord_norm.ReInit(0);
  scoord_area.ReInit(0);

  rforce_single.ReInit(0);
  qforce_single.ReInit(0);
  uforce_double.ReInit(0);
  qforce_double.ReInit(0);

  S_vel.ReInit(0);
  S_vel_up.ReInit(0);
  fmm_vel.ReInit(0);
  trg_vel.ReInit(0);
}

template <class Real>
template <class Vec>
void StokesVelocity<Real>::SetSrcCoord(const Vec& S, int sh_order_up_self_, int sh_order_up_){
  SCTLVec tmp(S.size(), (Real*)S.begin(), false);
  SetSrcCoord(tmp, sh_order_up_self_, sh_order_up_);
}

template <class Real>
void StokesVelocity<Real>::SetDensitySL(const SCTLVec* f, bool add_repul_){
  if(f){
    if(force_single.Dim()!=f->Dim()) fmm_setup=false;
    force_single.ReInit(f->Dim(), (Real*)&f[0][0], true);
  }else if(force_single.Dim()!=0){
    fmm_setup=false;
    force_single.ReInit(0);
  }

  rforce_single.ReInit(0);
  qforce_single.ReInit(0);
  add_repul=add_repul_;

  S_vel.ReInit(0);
  S_vel_up.ReInit(0);
  fmm_vel.ReInit(0);
  trg_vel.ReInit(0);
}

template <class Real>
template <class Vec>
void StokesVelocity<Real>::SetDensitySL(const Vec* f, bool add_repul_){
  if(f){
    SCTLVec tmp(f->size(), (Real*)f->begin(), false);
    SetDensitySL(&tmp, add_repul_);
  }else SetDensitySL((const SCTLVec*)NULL, add_repul_);
}

template <class Real>
void StokesVelocity<Real>::SetDensityDL(const SCTLVec* f){
  if(f){
    if(force_double.Dim()!=f->Dim()) fmm_setup=false;
    force_double.ReInit(f->Dim(), (Real*)&f[0][0], true);
  }else if(force_double.Dim()!=0){
    fmm_setup=false;
    force_double.ReInit(0);
  }

  uforce_double.ReInit(0);
  qforce_double.ReInit(0);

  S_vel.ReInit(0);
  S_vel_up.ReInit(0);
  fmm_vel.ReInit(0);
  trg_vel.ReInit(0);
}

template <class Real>
template <class Vec>
void StokesVelocity<Real>::SetDensityDL(const Vec* f){
  if(f){
    SCTLVec tmp(f->size(), (Real*)f->begin(), false);
    SetDensityDL(&tmp);
  }else SetDensityDL(NULL);
}

template <class Real>
void StokesVelocity<Real>::SetTrgCoord(const SCTLVec* T){
  if(T){
    trg_is_surf=false;
    tcoord.ReInit(T->Dim(),const_cast<Real*>(&T[0][0]));
  }else{
    trg_is_surf=true;
    tcoord.ReInit(0);
  }

  fmm_setup=false;
  S_vel.ReInit(0);
  S_vel_up.ReInit(0);
  fmm_vel.ReInit(0);
  trg_vel.ReInit(0);
}

template <class Real>
const typename StokesVelocity<Real>::SCTLVec& StokesVelocity<Real>::SelfInteraction(){
  setup_self();

  if(!S_vel.Dim()){ // Compute self interaction
    static SCTLVec Vcoef;
    { // Compute Vcoeff
      long Ncoef =   sh_order*(sh_order+2);
      long Ngrid = 2*sh_order*(sh_order+1);
      static SCTLVec SL_vel, DL_vel;
      SL_vel.ReInit(0);
      DL_vel.ReInit(0);

      if(rforce_single.Dim()){ // Set SL_vel
        static sctl::Vector<Real> F;
        SphericalHarmonics<Real>::Grid2SHC(rforce_single,sh_order,sh_order,F);

        long nv = rforce_single.Dim()/Ngrid/VES3D_DIM;
        SL_vel.ReInit(nv*VES3D_DIM*Ncoef);
        #pragma omp parallel
        { // mat-vec
          long tid=omp_get_thread_num();
          long omp_p=omp_get_num_threads();

          long a=(tid+0)*nv/omp_p;
          long b=(tid+1)*nv/omp_p;
          for(long i=a;i<b;i++){
            sctl::Matrix<Real> Mv(1,VES3D_DIM*Ncoef,&SL_vel[i*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real> Mf(1,VES3D_DIM*Ncoef,&F     [i*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real> M(VES3D_DIM*Ncoef,VES3D_DIM*Ncoef,&SLMatrix[i*VES3D_DIM*Ncoef*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real>::GEMM(Mv,Mf,M);
          }
        }
      }
      if(force_double.Dim()){ // Set DL_vel
        static sctl::Vector<Real> F;
        SphericalHarmonics<Real>::Grid2SHC(force_double,sh_order,sh_order,F);

        long nv = force_double.Dim()/Ngrid/VES3D_DIM;
        DL_vel.ReInit(nv*VES3D_DIM*Ncoef);
        #pragma omp parallel
        { // mat-vec
          long tid=omp_get_thread_num();
          long omp_p=omp_get_num_threads();

          long a=(tid+0)*nv/omp_p;
          long b=(tid+1)*nv/omp_p;
          for(long i=a;i<b;i++){
            sctl::Matrix<Real> Mv(1,VES3D_DIM*Ncoef,&DL_vel[i*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real> Mf(1,VES3D_DIM*Ncoef,&F     [i*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real> M(VES3D_DIM*Ncoef,VES3D_DIM*Ncoef,&DLMatrix[i*VES3D_DIM*Ncoef*VES3D_DIM*Ncoef],false);
            sctl::Matrix<Real>::GEMM(Mv,Mf,M);
          }
        }
      }
      if(SL_vel.Dim() && DL_vel.Dim()){ // Vcoef=SL_vel+DL_vel
        Vcoef.ReInit(SL_vel.Dim());
        #pragma omp parallel for
        for(long i=0;i<Vcoef.Dim();i++) Vcoef[i]=SL_vel[i]+DL_vel[i];
      }else{
        if(SL_vel.Dim()) Vcoef.ReInit(SL_vel.Dim(),&SL_vel[0]);
        else if(DL_vel.Dim()) Vcoef.ReInit(DL_vel.Dim(),&DL_vel[0]);
        else Vcoef.ReInit(0);
      }
    }
    SphericalHarmonics<Real>::SHC2Grid(Vcoef, sh_order, sh_order   , S_vel);
  }

  return S_vel;
}

template <class Real>
template <class Vec>
void StokesVelocity<Real>::SelfInteraction(Vec& vel){
  SCTLVec self_vel_(vel.size(),vel.begin(),false);

  const SCTLVec& self_vel_tmp=this->SelfInteraction();
  assert(self_vel_tmp.Dim()==self_vel_.Dim());
  self_vel_=self_vel_tmp;
}

template <class Real>
void StokesVelocity<Real>::setup_self(){

  if(!SLMatrix.Dim() || !DLMatrix.Dim()){
    if(!SLMatrix.Dim() && !DLMatrix.Dim() && force_single.Dim() && force_double.Dim()){
      if(1){
        sctl::Vector<Real> tmp1; tmp1.Swap(SLMatrix);
        sctl::Vector<Real> tmp2; tmp2.Swap(DLMatrix);
      }
      SphericalHarmonics<Real>::StokesSingularInteg(scoord, sh_order, sh_order_up_self, &SLMatrix, &DLMatrix);
    }else if(!SLMatrix.Dim() && force_single.Dim()){
      if(1){
        sctl::Vector<Real> tmp1; tmp1.Swap(SLMatrix);
      }
      SphericalHarmonics<Real>::StokesSingularInteg(scoord, sh_order, sh_order_up_self, &SLMatrix, NULL);
    }else if(!DLMatrix.Dim() && force_double.Dim()){
      if(1){
        sctl::Vector<Real> tmp2; tmp2.Swap(DLMatrix);
      }
      SphericalHarmonics<Real>::StokesSingularInteg(scoord, sh_order, sh_order_up_self, NULL, &DLMatrix);
    }
  }

  if(!rforce_single.Dim() && force_single.Dim()){
    rforce_single.ReInit(force_single.Dim(),&force_single[0],false);
  }

}

template <class Real>
void WriteVTK(const sctl::Vector<Real>& S, long p0, long p1, const char* fname, Real period=0, const sctl::Vector<Real>* v_ptr=NULL, const sctl::Vector<Real>* s_ptr=NULL, Real* centrosome_pos=NULL, long n_centrosome=0){
  typedef double VTKReal;
  int data__dof=VES3D_DIM;

  sctl::Vector<Real> X, Xp, V, Vp, Sca, Scap;
  { // Upsample X
    const sctl::Vector<Real>& X0=S;
    sctl::Vector<Real> X1;
    SphericalHarmonics<Real>::Grid2SHC(X0,p0,p0,X1);
    SphericalHarmonics<Real>::SHC2Grid(X1,p0,p1,X);
    SphericalHarmonics<Real>::SHC2Pole(X1, p0, Xp);
  }
  if(v_ptr){ // Upsample V
    const sctl::Vector<Real>& X0=*v_ptr;
    sctl::Vector<Real> X1;
    SphericalHarmonics<Real>::Grid2SHC(X0,p0,p0,X1);
    SphericalHarmonics<Real>::SHC2Grid(X1,p0,p1,V);
    SphericalHarmonics<Real>::SHC2Pole(X1, p0, Vp);
  }
  if(s_ptr){ // Upsample Sca
    const sctl::Vector<Real>& X0=*s_ptr;
    sctl::Vector<Real> X1;
    SphericalHarmonics<Real>::Grid2SHC(X0,p0,p0,X1);
    SphericalHarmonics<Real>::SHC2Grid(X1,p0,p1,Sca);
    SphericalHarmonics<Real>::SHC2Pole(X1, p0, Scap);
  }

  std::vector<VTKReal> point_coord;
  std::vector<VTKReal> point_value_vec1;
  std::vector<VTKReal> point_value_sca1;
  std::vector< int32_t> poly_connect;
  std::vector< int32_t> poly_offset;
  { // Set point_coord, point_values, poly_connect
    size_t N_ves = X.Dim()/(2*p1*(p1+1)*VES3D_DIM); // Number of vesicles
    assert(Xp.Dim() == N_ves*2*VES3D_DIM);
    for(size_t k=0;k<N_ves;k++){ // Set point_coord
      Real C[VES3D_DIM]={0,0,0};
      if(period>0){
        for(long l=0;l<VES3D_DIM;l++) C[l]=0;
        for(size_t i=0;i<p1+1;i++){
          for(size_t j=0;j<2*p1;j++){
            for(size_t l=0;l<VES3D_DIM;l++){
              C[l]+=X[j+2*p1*(i+(p1+1)*(l+k*VES3D_DIM))];
            }
          }
        }
        for(size_t l=0;l<VES3D_DIM;l++) C[l]+=Xp[0+2*(l+k*VES3D_DIM)];
        for(size_t l=0;l<VES3D_DIM;l++) C[l]+=Xp[1+2*(l+k*VES3D_DIM)];
        for(long l=0;l<VES3D_DIM;l++) C[l]/=2*p1*(p1+1)+2;
        for(long l=0;l<VES3D_DIM;l++) C[l]=(round(C[l]/period))*period;
      }

      for(size_t i=0;i<p1+1;i++){
        for(size_t j=0;j<2*p1;j++){
          for(size_t l=0;l<VES3D_DIM;l++){
            point_coord.push_back(X[j+2*p1*(i+(p1+1)*(l+k*VES3D_DIM))]-C[l]);
          }
        }
      }
      for(size_t l=0;l<VES3D_DIM;l++) point_coord.push_back(Xp[0+2*(l+k*VES3D_DIM)]-C[l]);
      for(size_t l=0;l<VES3D_DIM;l++) point_coord.push_back(Xp[1+2*(l+k*VES3D_DIM)]-C[l]);
    }

    if(v_ptr){
      data__dof = V.Dim() / (N_ves * 2*p1*(p1+1));
      for(size_t k=0;k<N_ves;k++){ // Set point_value_vec1
        for(size_t i=0;i<p1+1;i++){
          for(size_t j=0;j<2*p1;j++){
            for(size_t l=0;l<data__dof;l++){
              point_value_vec1.push_back(V[j+2*p1*(i+(p1+1)*(l+k*data__dof))]);
            }
          }
        }
        for(size_t l=0;l<data__dof;l++) point_value_vec1.push_back(Vp[0+2*(l+k*data__dof)]);
        for(size_t l=0;l<data__dof;l++) point_value_vec1.push_back(Vp[1+2*(l+k*data__dof)]);
      }
    }

    if(s_ptr){
      data__dof = Sca.Dim() / (N_ves * 2*p1*(p1+1));
      for(size_t k=0;k<N_ves;k++){ // Set point_value_sca1
        for(size_t i=0;i<p1+1;i++){
          for(size_t j=0;j<2*p1;j++){
            for(size_t l=0;l<data__dof;l++){
              point_value_sca1.push_back(Sca[j+2*p1*(i+(p1+1)*(l+k*data__dof))]);
            }
          }
        }
        for(size_t l=0;l<data__dof;l++) point_value_sca1.push_back(Scap[0+2*(l+k*data__dof)]);
        for(size_t l=0;l<data__dof;l++) point_value_sca1.push_back(Scap[1+2*(l+k*data__dof)]);
      }
    }

    for(size_t k=0;k<N_ves;k++){
      for(size_t j=0;j<2*p1;j++){
        size_t i0= 0;
        size_t i1=p1;
        size_t j0=((j+0)       );
        size_t j1=((j+1)%(2*p1));

        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*(p1+1)+0);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j0);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j1);
        poly_offset.push_back(poly_connect.size());

        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*(p1+1)+1);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j0);
        poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j1);
        poly_offset.push_back(poly_connect.size());
      }
      for(size_t i=0;i<p1;i++){
        for(size_t j=0;j<2*p1;j++){
          size_t i0=((i+0)       );
          size_t i1=((i+1)       );
          size_t j0=((j+0)       );
          size_t j1=((j+1)%(2*p1));
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j0);
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j0);
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i1+j1);
          poly_connect.push_back((2*p1*(p1+1)+2)*k + 2*p1*i0+j1);
          poly_offset.push_back(poly_connect.size());
        }
      }
    }
  }

  int myrank, np;
  np = 1;
  myrank = 0;

  std::vector<VTKReal>& coord=point_coord;
  std::vector<VTKReal>& value_vec1=point_value_vec1;
  std::vector<VTKReal>& value_sca1=point_value_sca1;
  std::vector<int32_t> connect=poly_connect;
  std::vector<int32_t> offset=poly_offset;
  std::vector<int32_t> connect_vert;
  std::vector<int32_t> offset_vert;

  for(int i=0; i<n_centrosome; i++){
    connect_vert.push_back(coord.size()/VES3D_DIM);
    offset_vert.push_back(connect_vert.size());
    value_sca1.push_back(0);
    for(size_t l=0;l<VES3D_DIM;l++){
      coord.push_back(centrosome_pos[VES3D_DIM*i+l]);
      value_vec1.push_back(0);
    }
  }
  int pt_cnt=coord.size()/VES3D_DIM;
  int poly_cnt=poly_offset.size();
  int vert_cnt=n_centrosome;

  //Open file for writing.
  std::stringstream vtufname;
  vtufname<<fname<<"_"<<std::setfill('0')<<std::setw(6)<<myrank<<".vtp";
  std::ofstream vtufile;
  vtufile.open(vtufname.str().c_str());
  if(vtufile.fail()) return;

  bool isLittleEndian;
  {
    uint16_t number = 0x1;
    uint8_t *numPtr = (uint8_t*)&number;
    isLittleEndian=(numPtr[0] == 1);
  }

  //Proceed to write to file.
  size_t data_size=0;
  vtufile<<"<?xml version=\"1.0\"?>\n";
  if(isLittleEndian) vtufile<<"<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  else               vtufile<<"<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  //===========================================================================
  vtufile<<"  <PolyData>\n";
  vtufile<<"    <Piece NumberOfPoints=\""<<pt_cnt<<"\" NumberOfVerts=\""<<vert_cnt<<"\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""<<poly_cnt<<"\">\n";

  //---------------------------------------------------------------------------
  vtufile<<"      <Points>\n";
  vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<VES3D_DIM<<"\" Name=\"Position\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+coord.size()*sizeof(VTKReal);
  vtufile<<"      </Points>\n";
  //---------------------------------------------------------------------------
  if(value_vec1.size() || value_sca1.size())
    vtufile<<"      <PointData>\n";
  if(value_vec1.size()){ // value
    vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value_vec1.size()/pt_cnt<<"\" Name=\""<<"value vec1"<<"\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+value_vec1.size()*sizeof(VTKReal);
  }
  if(value_sca1.size()){ // value
    vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value_sca1.size()/pt_cnt<<"\" Name=\""<<"value sca1"<<"\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+value_sca1.size()*sizeof(VTKReal);
  }
  if(value_vec1.size() || value_sca1.size())
    vtufile<<"      </PointData>\n";

  //---------------------------------------------------------------------------
  vtufile<<"      <Polys>\n";
  vtufile<<"        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+connect.size()*sizeof(int32_t);
  vtufile<<"        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+offset.size() *sizeof(int32_t);
  vtufile<<"      </Polys>\n";
  //---------------------------------------------------------------------------
  if(vert_cnt>0){
    vtufile<<"      <Verts>\n";
    vtufile<<"        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+connect_vert.size()*sizeof(int32_t);
    vtufile<<"        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+offset_vert.size() *sizeof(int32_t);
    vtufile<<"      </Verts>\n";
  }

  vtufile<<"    </Piece>\n";
  vtufile<<"  </PolyData>\n";
  //===========================================================================
  vtufile<<"  <AppendedData encoding=\"raw\">\n";
  vtufile<<"    _";

  int32_t block_size;
  block_size=coord.size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&coord  [0], coord.size()*sizeof(VTKReal));
  if(value_vec1.size()){ // value
    block_size=value_vec1.size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&value_vec1  [0], value_vec1.size()*sizeof(VTKReal));
  }
  if(value_sca1.size()){ // value
    block_size=value_sca1.size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&value_sca1  [0], value_sca1.size()*sizeof(VTKReal));
  }
  block_size=connect.size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&connect[0], connect.size()*sizeof(int32_t));
  block_size=offset .size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&offset [0], offset .size()*sizeof(int32_t));
  if(vert_cnt>0){
    block_size=connect_vert.size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&connect_vert[0], connect_vert.size()*sizeof(int32_t));
    block_size=offset_vert .size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&offset_vert [0], offset_vert .size()*sizeof(int32_t));
  }

  vtufile<<"\n";
  vtufile<<"  </AppendedData>\n";
  //===========================================================================
  vtufile<<"</VTKFile>\n";
  vtufile.close();


  if(myrank) return;
  std::stringstream pvtufname;
  pvtufname<<fname<<".pvtp";
  std::ofstream pvtufile;
  pvtufile.open(pvtufname.str().c_str());
  if(pvtufile.fail()) return;
  pvtufile<<"<?xml version=\"1.0\"?>\n";
  pvtufile<<"<VTKFile type=\"PPolyData\">\n";
  pvtufile<<"  <PPolyData GhostLevel=\"0\">\n";
  pvtufile<<"      <PPoints>\n";
  pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<VES3D_DIM<<"\" Name=\"Position\"/>\n";
  pvtufile<<"      </PPoints>\n";
  if(value_vec1.size() || value_sca1.size())
    pvtufile<<"      <PPointData>\n";
  if(value_vec1.size()){ // value
    pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value_vec1.size()/pt_cnt<<"\" Name=\""<<"value vec1"<<"\"/>\n";
  }
  if(value_sca1.size()){ // value
    pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value_sca1.size()/pt_cnt<<"\" Name=\""<<"value sca1"<<"\"/>\n";
  }
  if(value_vec1.size() || value_sca1.size())
    pvtufile<<"      </PPointData>\n";

  {
    // Extract filename from path.
    std::stringstream vtupath;
    vtupath<<'/'<<fname;
    std::string pathname = vtupath.str();
    unsigned found = pathname.find_last_of("/\\");
    std::string fname_ = pathname.substr(found+1);
    for(int i=0;i<np;i++) pvtufile<<"      <Piece Source=\""<<fname_<<"_"<<std::setfill('0')<<std::setw(6)<<i<<".vtp\"/>\n";
  }
  pvtufile<<"  </PPolyData>\n";
  pvtufile<<"</VTKFile>\n";
  pvtufile.close();
}

template <class Surf>
void WriteVTK(const Surf& S, const char* fname, const typename Surf::Vec_t* v_ptr=NULL, const typename Surf::Sca_t* s_ptr=NULL, int order=-1, typename Surf::value_type period=0, typename Surf::value_type* centrosome_pos=NULL, int n_centrosome=0){
  typedef typename Surf::value_type Real;
  typedef typename Surf::Vec_t Vec;
  size_t p0=S.getShOrder();
  size_t p1=(order>0?order:p0); // upsample

  sctl::Vector<Real> S_, v_, s_;
  S_.ReInit(S.getPosition().size(),(Real*)S.getPosition().begin(),false);
  if(v_ptr) v_.ReInit(v_ptr->size(),(Real*)v_ptr->begin(),false);
  if(s_ptr) s_.ReInit(s_ptr->size(),(Real*)s_ptr->begin(),false);
  WriteVTK(S_, p0, p1, fname, period, (v_ptr?&v_:NULL), (s_ptr?&s_:NULL), centrosome_pos, n_centrosome);
}

