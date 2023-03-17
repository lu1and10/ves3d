function [fRot sRot]= movePole(f,thetaTarg,lambdaTarg, isReal) 
% [cc shcNew]= movePole(f,thetaTarg,lambdaTarg) move the pole of the
%parameters to the target point with coordinate thetaTarg (elevation
%[-pi/2,pi/2]) and lambdaTarg (azimuth [0,2pi)).  MOVEPOLE returns the
%Cartesian coordinates of the points with now parametrization (fRot) and
%their corresponding spherical harmonic coefficients (shcRot).

  if(nargin==0), testMovePole(); return;end
  if(nargin<4), isReal = true; end
  [d1 d2] = size(f);
  N = (sqrt(2*d1+1)-1)/2;

  %% lambdaTarg rotation
  mMat = exp(1i*repmat(-N:N,N+1,1)*lambdaTarg);
  shc = repmat(shrinkShVec(mMat(:)),1,d2).*shAna(f);

  %% thetaTarg rotation
  %Parametrization of the equator in the rotated from. We need 2*N+2
  %equispaced points in [0,2pi).
  phiR = (0:2*N+1)*2*pi/(2*N+2) + pi/(2*N+2);
  %Finding the corresponding material point
  [phi theta] = cart2sph(cos(thetaTarg)*cos(phiR),sin(phiR),-sin(thetaTarg)*cos(phiR));
  %Mapping phi to [0,2pi) 
  phi = mod(phi,2*pi)';
  theta = pi/2-theta'; 
  %Coefficients for G
  c1 = sin(thetaTarg)*cos(theta).*cos(phi) - cos(thetaTarg)*sin(theta);
  c2 = sin(thetaTarg)*sin(phi)./sin(theta);
  %Calculating f and g and finding their Fourier coefficients.
  for idx=1:d2
    for n=0:N    
      [Yn Hn] = Ynm(n,[],theta(:),phi(:));
      coeff = repmat(shc(n^2+(1:2*n+1),idx).',2*N+2,1);
    
      Y2 = (c2*(1i*(-n:n))).*Yn;
      Hn = -repmat(c1,1,2*n+1).*Hn;
    
      f = sum(coeff.* Yn      ,2);
      g = sum(coeff.*(Hn + Y2),2);
    
      for m =-n:n
        ff(n+1,N+m+1) = 2*pi/(2*N+2)*sum(f.*exp(-1i*m*phiR'))/2/pi;
        gg(n+1,N+m+1) = 2*pi/(2*N+2)*sum(g.*exp(-1i*m*phiR'))/2/pi;
      end      
    end
  
    %Calculation new shc coeffiecents
    for n=0:N
      [P Q] = Ynm(n,[],pi/2,0);
      for m=-n:n
        P0 = P(n+1+m);
        Q0 = Q(n+1+m);
        shcRot(n+1,N+m+1) = (ff(n+1,N+m+1)*P0 + gg(n+1,N+m+1)*Q0)/(P0^2+Q0^2);
      end
    end
    sRot(:,idx) = shcRot(:);
  end
  %% lambdaTarg rotation -- back 
  mMat = exp(-1i*repmat(-N:N,N+1,1)*lambdaTarg);
  sRot = repmat(mMat(:),1,d2).*sRot;
  sRot = shrinkShVec(sRot);
  
  %% Calculation point values 
  fRot = shSyn(sRot, isReal);
 
function testMovePole()
  np = 16;
  
  [u v] = parDomain(np); u = pi/2-u;
  rho = @(theta,phi) 1+.1*cos(6*theta);%.5*real(Ynm(4,3,theta,phi));
  [x y z] = sph2cart(v,u,rho(u,v));
  X =[x y z];
  theta = 3*pi/4; phi = 0;
  
  XX = movePole(X,theta,phi);
  
  subplot(1,2,1); plotb(X(:));
  subplot(1,2,2); plotb(XX(:));
  
  [v u r]=cart2sph(XX(:,1),XX(:,2),XX(:,3));
  e = abs(r - rho(u,v));
  disp([' First shape error = ' num2str(max(e(:)))]);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  pause; clear; clf;
  np = 16;
  [u v] = parDomain(np); u = u-pi/2;
  rho = @(theta,phi) exp(.5*cos(theta).^4.*sin(theta).*cos(4*phi));
  [x y z] = sph2cart(v,u,rho(u,v));
  X =[x y z];
  theta = 3*pi/4; phi = 0;
  
  XX = movePole(X,theta,phi);
  subplot(1,2,1); plotb(X(:),[],'plain');
  subplot(1,2,2); plotb(XX(:),[],'plain');
  
  [v u r]=cart2sph(XX(:,1),XX(:,2),XX(:,3));
  e = abs(r - rho(u,v));
  disp([' Second shape error = ' num2str(max(e(:)))]);
  pause;

  XX = movePole(XX,-theta,-phi);
  clf;
  subplot(1,2,1); plotb(X(:),[],'plain');
  subplot(1,2,2); plotb(XX(:),[],'plain');

  e = abs(XX(:)-X(:));
  disp([' Rotation back to original error = ' num2str(max(e))]);
