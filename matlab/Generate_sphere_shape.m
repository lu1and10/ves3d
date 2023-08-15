p = 16;
[u,v] = parDomain(p);
a = 1.0; b = 1.0; c = 1.0;
X = a * sin(u) .* cos(v);
Y = b * sin(u) .* sin(v);
Z = c * cos(u);

S = vesicle([X;Y;Z]);
rv = S.reducedVol;
disp(['reducedVol: ' num2str(rv)]);

x = reshape(X,p+1,[])';
y = reshape(Y,p+1,[])';
z = reshape(Z,p+1,[])';
data_out = [x(:); y(:); z(:)];
fileName = ['sphere_p' num2str(p) '_ra_' num2str(rv) '.txt'];
save(fileName,'data_out','-ascii', '-double');
