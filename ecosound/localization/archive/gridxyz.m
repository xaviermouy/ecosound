isdev = 0;                  % 0=use knowledge of data stdev; 1=estimate data stdev
B     = 0.99;               % Percent for CIs
xtrue = [5.2, 15.7, -5.1];  % true source location
sdev  = 2e-5;               % data std dev (puts errors on dobs)
v     = 1485;               % water sound velocity

Nhyd = 7;      % Number of hydrophones

xh(1,:) = [ -1.04, -0.10,  0.23 ];    % Our outside HPs
xh(2,:) = [ -2.17,  0.40, -0.25 ];   
xh(3,:) = [ -2.56,  0.12, -1.39 ];   
xh(4,:) = [ -1.74,  0.05, -1.63 ];    %Ch 21
xh(5,:) = [  0.53,  0.07, -1.42 ];    %Ch 22
xh(6,:) = [  0.52, -0.11, -0.40 ];    %Ch 23
xh(7,:) = [  0.85,  0.16, -0.19 ];    %Ch 24

%rng(2637)   % set random number generator

%------------------------------------------------------------------------------
%  Compute (simulated) observed data (h/ph 1 is reference)
%------------------------------------------------------------------------------

Ndat = Nhyd-1;          % Number of data
dobs = zeros(Ndat,1);

r1 = sqrt((xtrue(1)-xh(1,1))^2+(xtrue(2)-xh(1,2))^2+(xtrue(3)-xh(1,3))^2);
for i=2:Nhyd
   r = sqrt((xtrue(1)-xh(i,1))^2+(xtrue(2)-xh(i,2))^2+(xtrue(3)-xh(i,3))^2);
   dobs(i-1) = (r-r1)/v+randn*sdev;
end

%------------------------------------------------------------------------------
%  Set up search grid in x, y, z
%------------------------------------------------------------------------------

x0 = -20;   % -20 to 20 by .2 m
dx = 0.2;
Nx = 201;
xgrid = x0+(0:Nx-1)*dx;
    
y0 = -50;      % -50 to 50 by .25 m
dy = 0.25;
Ny = 401;
ygrid = y0+(0:Ny-1)*dy; 
    
z0 = 0;      % 0 to -24 by .25 m
dz = -0.25;
Nz = 101;
zgrid = z0+(0:Nz-1)*dz; 
 
%------------------------------------------------------------------------------
%  Compute sum-of-squared-misft (ssq) on grid (vectorized over x)
%------------------------------------------------------------------------------
    
ssq = zeros(Nx,Ny,Nz);
r1  = zeros(Nx,1);
r   = zeros(Nx,1);
    
for iy=1:Ny
    for iz=1:Nz   
        r1 = sqrt((xgrid-xh(1,1)).^2+(ygrid(iy)-xh(1,2))^2+(zgrid(iz)-xh(1,3)).^2)';            
        for i=2:Nhyd
            r = sqrt((xgrid-xh(i,1)).^2+(ygrid(iy)-xh(i,2))^2+(zgrid(iz)-xh(i,3)).^2)';
            ssq(:,iy,iz) = ssq(:,iy,iz)+((r-r1)/v-dobs(i-1)).^2;
        end
    end
end

%------------------------------------------------------------------------------
%  Compute normalized PPD 
%------------------------------------------------------------------------------

if (isdev == 1)
    sdest = sqrt(min(min(min(ssq)))/Ndat)   % Estimate std dev from residuals
else
    sdest = sdev;
end

L   = exp(-ssq./(2*sdest^2));            % unormalized likelihood
PPD = L./sum(sum(sum(L)));               % normalized PPD
   
%------------------------------------------------------------------------------
%  Compute 1D marginal probability distributions in x, y, z.
%------------------------------------------------------------------------------

Px = zeros(Nx,1);
for ix=1:Nx
    Px(ix) = sum(sum(PPD(ix,:,:)));
end
Px = Px/sum(Px);

Py = zeros(Ny,1);
for iy=1:Ny
    Py(iy) = sum(sum(PPD(:,iy,:)));
end
Py = Py/sum(Py);

Pz = zeros(Nz,1);
for iz=1:Nz
    Pz(iz) = sum(sum(PPD(:,:,iz)));
end
Pz = Pz/sum(Pz);

%------------------------------------------------------------------------------
%  Find location of 3-D maximum probability estimate
%------------------------------------------------------------------------------

ixmax = 1; 
iymax = 1;
izmax = 1;
Pmax  = -1000;
for iy=1:Ny
    for iz=1:Nz
        [Ptry ixtry] = max(squeeze(PPD(:,iy,iz)));
        if (Ptry > Pmax)
            Pmax = Ptry;
            ixmax = ixtry;
            iymax = iy;
            izmax = iz;
        end
    end
end
xopt = xgrid(ixmax);
yopt = ygrid(iymax);
zopt = zgrid(izmax);

%------------------------------------------------------------------------------
%  Compute standard deviations of optimal estimates
%------------------------------------------------------------------------------

sdx = 0;
for ix=1:Nx
    sdx = sdx+(xopt-xgrid(ix))^2*Px(ix);
end
sdx = sqrt(sdx);

sdy = 0;
for iy=1:Ny
    sdy = sdy+(yopt-ygrid(iy))^2*Py(iy);
end
sdy = sqrt(sdy);

sdz = 0;
for iz=1:Nz
    sdz = sdz+(zopt-zgrid(iz))^2*Pz(iz);
end
sdz = sqrt(sdz);


%------------------------------------------------------------------------------
%  Compute B% highest-probability density credibility intervas
%------------------------------------------------------------------------------
    
iCItry = zeros(Nx-1,2);    %  CI for x
itry  = 1;
for istart=1:Nx-1
    for iend=istart+1:Nx
        Psum = sum(Px(istart:iend));
       if (Psum >= B)
            iCItry(itry,1) = istart;
            iCItry(itry,2) = iend;            
            break
       end
    end
    if (Psum < B)
        iCItry(itry,1) = 1;
        iCItry(itry,2) = Nx; 
    end
    itry = itry+1;
end
[CImin,imin] = min(iCItry(:,2)-iCItry(:,1));
CIx1 = xgrid(iCItry(imin,1));
CIx2 = xgrid(iCItry(imin,2));
CIx = sum(Px(iCItry(imin,1):iCItry(imin,2)));

iCItry = zeros(Ny-1,2);    %  CI for y
itry  = 1;
for istart=1:Ny-1
    for iend=istart+1:Ny
        Psum = sum(Py(istart:iend));
       if (Psum >= B)
            iCItry(itry,1) = istart;
            iCItry(itry,2) = iend;            
            break
       end
    end
    if (Psum < B)
        iCItry(itry,1) = 1;
        iCItry(itry,2) = Ny; 
    end
    itry = itry+1;
end
[CImin,imin] = min(iCItry(:,2)-iCItry(:,1));
CIy1 = ygrid(iCItry(imin,1));
CIy2 = ygrid(iCItry(imin,2));
CIy  = sum(Py(iCItry(imin,1):iCItry(imin,2)));

iCItry = zeros(Nz-1,2);     %  CI for z
itry  = 1;
for istart=1:Nz-1
    for iend=istart+1:Nz
        Psum = sum(Pz(istart:iend));
       if (Psum >= B)
            iCItry(itry,1) = istart;
            iCItry(itry,2) = iend;            
            break
       end
    end
    if (Psum < B)
        iCItry(itry,1) = 1;
        iCItry(itry,2) = Nz; 
    end
    itry = itry+1;
end
[CImin,imin] = min(iCItry(:,2)-iCItry(:,1));
CIz1 = zgrid(iCItry(imin,1));
CIz2 = zgrid(iCItry(imin,2));
CIz  = sum(Pz(iCItry(imin,1):iCItry(imin,2)));    

%------------------------------------------------------------------------------
%  Compute 2D Marginals
%------------------------------------------------------------------------------

Pxy = zeros(Nx,Ny);
for iy=1:Ny
    Pxy(:,iy) = sum(PPD(:,iy,:),3);
end
Pxy = Pxy/sum(sum(Pxy));

Pxz = zeros(Nx,Nz);
for iz=1:Nz
    Pxz(:,iz) = sum(PPD(:,:,iz),2);
end
Pxz = Pxz/sum(sum(Pxz));

Pyz = zeros(Ny,Nz);
for iz=1:Nz
    Pyz(:,iz) = sum(PPD(:,:,iz),1);
end
Pyz = Pyz/sum(sum(Pyz));

%------------------------------------------------------------------------------
%  Plot 1D Marginals with credibility intervals
%------------------------------------------------------------------------------

figure(1)

x0 = .1;
y0 = .1;
xw = .85;
yw = .225;
xs = .08;
ys = .1;

lwd       = 1.25;
fontsize  = 12;
  
subplot('position',[x0,y0+2*(yw+ys),xw,yw])
plot(xgrid,Px,'k','LineWidth',lwd); grid
xlabel('x (m)');
ylabel('P(x)');
hold on
plot([xtrue(1) xtrue(1)],[0 0],'rX','MarkerSize',10,'LineWidth',2)
plot([xopt xopt],[0 0],'kX','MarkerSize',10,'LineWidth',2)
plot([CIx1 CIx1],ylim,'k:','LineWidth',2)
plot([CIx2 CIx2],ylim,'k:','LineWidth',2)
hold off

subplot('position',[x0,y0+1*(yw+ys),xw,yw]) 
plot(ygrid,Py,'k','LineWidth',lwd); grid
xlabel('y (m)');
ylabel('P(y)');
hold on
plot([xtrue(2) xtrue(2)],[0 0],'rX','MarkerSize',10,'LineWidth',2)
plot([yopt yopt],[0 0],'kX','MarkerSize',10,'LineWidth',2)
plot([CIy1 CIy1],ylim,'k:','LineWidth',2)
plot([CIy2 CIy2],ylim,'k:','LineWidth',2)
hold off

subplot('position',[x0,y0+0*(yw+ys),xw,yw]) 
plot(zgrid,Pz,'k','LineWidth',lwd); grid
xlabel('z (m)');
ylabel('P(z)');
hold on
plot([xtrue(3) xtrue(3)],[0 0],'rX','MarkerSize',10,'LineWidth',2)
plot([zopt zopt],[0 0],'kX','MarkerSize',10,'LineWidth',2)
plot([CIz1 CIz1],ylim,'k:','LineWidth',2)
plot([CIz2 CIz2],ylim,'k:','LineWidth',2)
hold off

axh = findobj(gcf); % get the handles associated with the current figure
allaxes  = findall(axh,'Type','axes');
alllines = findall(axh,'Type','line');
alltext  = findall(axh,'Type','text');
set(allaxes,'FontName','Arial','LineWidth',lwd,'FontSize',fontsize);
%set(alllines,'Linewidth',linewidth);
set(alltext,'FontName','Arial','FontSize',fontsize);
set(allaxes,'TickDir','out');

%------------------------------------------------------------------------------
%  Plot 2D Marginals
%------------------------------------------------------------------------------

figure(2)

subplot('position',[x0,y0+2*(yw+ys),xw,yw])    % Plot x-y marginal
imagesc(xgrid,ygrid,Pxy(:,:)'); grid; colorbar
set(gca,'YDir','normal');  % Flip so y increases upward
xlabel('x (m)');
ylabel('y (m)');
hold on
plot([xtrue(1)],[xtrue(2)],'w+','LineWidth',3,'MarkerSize',12)
plot([xtrue(1)],[xtrue(2)],'k+','LineWidth',2,'MarkerSize',8)
hold off

subplot('position',[x0,y0+1*(yw+ys),xw,yw])    % Plot x-z marginal 
imagesc(xgrid,zgrid,Pxz(:,:)'); grid; colorbar
set(gca,'YDir','normal');  % Flip so y increases upward
xlabel('x (m)');
ylabel('z (m)');
hold on
plot([xtrue(1)],[xtrue(3)],'w+','LineWidth',3,'MarkerSize',12)
plot([xtrue(1)],[xtrue(3)],'k+','LineWidth',2,'MarkerSize',8)
hold off

subplot('position',[x0,y0+0*(yw+ys),xw,yw])    % Plot y-z marginal  
imagesc(ygrid,zgrid,Pyz(:,:)'); grid; colorbar
set(gca,'YDir','normal');  % Flip so y increases upward
xlabel('y (m)');
ylabel('z (m)');
hold on
plot([xtrue(2)],[xtrue(3)],'w+','LineWidth',3,'MarkerSize',12)
plot([xtrue(2)],[xtrue(3)],'k+','LineWidth',2,'MarkerSize',8)
hold off

MyNewColorMap = jet;       % Set zero to white; linearly interpolate into jet
iblue = find(ismember(MyNewColorMap, [0 0 1], 'rows'));
yy = [1 0]; 
xx = [1 iblue];
blend = interp1(xx,yy,1:iblue);
for i=1:iblue
   MyNewColorMap(i,:) = [blend(i) blend(i) 1.0000];
end
colormap(MyNewColorMap);





