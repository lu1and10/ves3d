function [V, A, RA] = reducedVolume(S)
%REDUCED_VOLUMEX - Given the shape S, it returns
%the volume V, area A and reduced volume RA.

persistent p wt

if(isempty(wt) || p ~= S.p)
  p = S.p;
  [trash gwt]=grule(p+1);
  wt = pi/p*repmat(gwt', 2*p, 1)./sin(parDomain(p));
  wt = wt(:)';
end

W = interpsh(S.geoProp.W,p);
X = interpsh(S.cart,p);
nor = interpsh(S.geoProp.nor,p);

A = wt*W;
V = 1/3*wt*(dot(X,nor).*W);

RA = 6*sqrt(pi)*V./A.^(1.5);
  
