function gridxyz_xm()
% This code take TDOAs from python localization code and returns the optimal
% localizations (which were already calculated) and the confidence
% intervals. Results are save to a csv file.

clear all
close all
clc

%isdev = 0;                 % 0=use knowledge of data stdev; 1=estimate data stdev
B     = 0.99 ;              % Percent for CIs
sdev  = 1.8893e-05;         % data std dev (puts errors on dobs) 
v     = 1484;               % water sound velocity
Nhyd = 4;      % Number of hydrophones
ref_hyd = 3;   % reference hydrophone

% %original  mobile array
% xh(1,:) = [ -0.46, 0.00, 0.00 ];
% xh(2,:) = [  0.00, 0.19, 0.54 ];
% xh(3,:) = [  0.00, 0.49, 0.00 ];
% xh(4,:) = [  0.48, 0.00, 0.00 ];

%original mini array
xh(1,:) = [  0.63, 0.00, 0.00 ];
xh(2,:) = [  0.13, 0.57, 0.14 ];
xh(3,:) = [  0.00, 0.00, 0.54 ];
xh(4,:) = [  -0.50, 0.00, 0.00 ];

% mobile Array:
% sdev  = 1.2576e-05 % Projector Mcauley POint
% sdev  = 3.3665e-05;% Copper exemple #1 (with blackeye goby in front)
% sdev  = 1.2418e-04 % Copper exemple #2 (on top of rock)
% sdev  = 1.1126e-04 % Goby

% mini array:
% sdev = 3.5460e-06 % ROV
% sdev = 1.8893e-05 % Copper

doplot=1;
%dobs=[-0.0006334, -0.000232, -4.2e-05]';

indir = 'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\mini-array_copper\';
%indir = 'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\mini-array_ROV\';
infile = 'localizations_python.csv';
%infile = 'localizations_python_corrected.csv';
%infile = 'localizations_matlab_with_CI.csv';


table = readtable([indir infile]);
table.x_min_CI99 = zeros(height(table),1);
table.x_max_CI99 = zeros(height(table),1);
table.y_min_CI99 = zeros(height(table),1);
table.y_max_CI99 = zeros(height(table),1);
table.z_min_CI99 = zeros(height(table),1);
table.z_max_CI99 = zeros(height(table),1);

%table = table(1:15,:);

channels = 1:Nhyd;
channels(ref_hyd)=[];
Ndat = Nhyd-1;          % Number of data

% % estimate Estimate std dev from residuals
% res=[];
% for ii = 1:height(table)    
%     r1 = sqrt((table.x(ii)-xh(ref_hyd,1)).^2+(table.y(ii)-xh(ref_hyd,2))^2+(table.z(ii)-xh(ref_hyd,3)).^2)';
%     idx=0;
%     dobs = [table.tdoa_sec_1(ii), table.tdoa_sec_2(ii), table.tdoa_sec_3(ii)]';
%     for i=channels
%         idx=idx+1;
%         r = sqrt( (table.x(ii)-xh(i,1)).^2+(table.y(ii)-xh(i,2))^2+(table.z(ii)-xh(i,3)).^2)';        
%         %res = res+((r-r1)/v-dobs(idx)).^2;
%         res = [res,((r-r1)/v-dobs(idx)).^2];
%     end %for i=channels
% end %for i = 1:height(table)
% %sdev = sqrt(res/(height(table)*(Ndat-1))); 
% sdev = sqrt(sum(res)/(height(table)*(Ndat-1))) 


% add std err that was used to the table (for book keeping)
table.tdoa_errors_std(:) = sdev;
for i = 1:height(table) 
    dobs = [table.tdoa_sec_1(i), table.tdoa_sec_2(i), table.tdoa_sec_3(i)]';
    [x,y,z,x_min, x_max,y_min, y_max,z_min, z_max]=solve_gridsearch(dobs,sdev, Nhyd, ref_hyd,xh,v,B,Ndat, channels,doplot);
    table.x_min_CI99(i) = x_min;
    table.x_max_CI99(i) = x_max;
    table.y_min_CI99(i) = y_min;
    table.y_max_CI99(i) = y_max;
    table.z_min_CI99(i) = z_min;
    table.z_max_CI99(i) = z_max;
    table.x(i) = x;
    table.y(i) = y;
    table.z(i) = z;
    disp(i)
    close all
    x
    y
    z    
end
writetable(table,[indir,'localizations_matlab_with_CI.csv'])

% %% plot single observation
% doplot=1;
% dobs = [0.0002747, 1e-07, -0.0003867]';
% [x,y,z,x_min, x_max,y_min, y_max,z_min, z_max]=solve_gridsearch(dobs,sdev, Nhyd, ref_hyd,xh,v,B,Ndat, channels,doplot);

end % function

function [x,y,z,x_min, x_max,y_min, y_max,z_min, z_max] = solve_gridsearch(dobs,sdev, Nhyd, ref_hyd,xh,v,B,Ndat, channels,doplot)

%------------------------------------------------------------------------------
%  Compute (simulated) observed data (h/ph 1 is reference)
%------------------------------------------------------------------------------



%------------------------------------------------------------------------------
%  Set up search grid in x, y, z
%------------------------------------------------------------------------------

%% -3 to 3 by .02m
x0 = -3;   % -8 to 8 by .02 m
dx = 0.02;
Nx = 301;
xgrid = x0+(0:Nx-1)*dx;

y0 = -3;      % -3 to 3 by .02 m
dy = 0.02;
Ny = 301;
ygrid = y0+(0:Ny-1)*dy;

z0 = -0.2;      % -1 to 3 by .02 m
dz = 0.02;
Nz = 161;
zgrid = z0+(0:Nz-1)*dz;

% %% -8 to 8 by .02m
% x0 = -8;   % -8 to 8 by .02 m
% dx = 0.02;
% Nx = 801;
% xgrid = x0+(0:Nx-1)*dx;
% 
% y0 = -8;      % -8 to 8 by .02 m
% dy = 0.02;
% Ny = 801;
% ygrid = y0+(0:Ny-1)*dy;
% 
% z0 = -2;      % -1 to 8 by .02 m
% dz = 0.02;
% Nz = 501;
% zgrid = z0+(0:Nz-1)*dz;

%------------------------------------------------------------------------------
%  Compute sum-of-squared-misft (ssq) on grid (vectorized over x)
%------------------------------------------------------------------------------

ssq = zeros(Nx,Ny,Nz);
r1  = zeros(Nx,1);
r   = zeros(Nx,1);

for iy=1:Ny
    for iz=1:Nz
        r1 = sqrt((xgrid-xh(ref_hyd,1)).^2+(ygrid(iy)-xh(ref_hyd,2))^2+(zgrid(iz)-xh(ref_hyd,3)).^2)';
        idx=0;
        for i=channels
            idx=idx+1;
            r = sqrt((xgrid-xh(i,1)).^2+(ygrid(iy)-xh(i,2))^2+(zgrid(iz)-xh(i,3)).^2)';
            ssq(:,iy,iz) = ssq(:,iy,iz)+((r-r1)/v-dobs(idx)).^2;
        end
    end
end

%------------------------------------------------------------------------------
%  Compute normalized PPD
%------------------------------------------------------------------------------

% if (isdev == 1)
%     sdest = sqrt(min(min(min(ssq)))/Ndat)   % Estimate std dev from residuals
% else
    sdest = sdev;
% end

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

x=xopt;
y=yopt;
z=zopt;
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
x_min = CIx1;
x_max = CIx2;

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
y_min = CIy1;
y_max = CIy2;

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

z_min = CIz1;
z_max = CIz2;

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
if doplot ==1
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
    %plot([xtrue(1) xtrue(1)],[0 0],'rX','MarkerSize',10,'LineWidth',2)
    plot([xopt xopt],[0 0],'kX','MarkerSize',10,'LineWidth',2)
    plot([CIx1 CIx1],ylim,'k:','LineWidth',2)
    plot([CIx2 CIx2],ylim,'k:','LineWidth',2)
    hold off
    
    subplot('position',[x0,y0+1*(yw+ys),xw,yw])
    plot(ygrid,Py,'k','LineWidth',lwd); grid
    xlabel('y (m)');
    ylabel('P(y)');
    hold on
    %plot([xtrue(2) xtrue(2)],[0 0],'rX','MarkerSize',10,'LineWidth',2)
    plot([yopt yopt],[0 0],'kX','MarkerSize',10,'LineWidth',2)
    plot([CIy1 CIy1],ylim,'k:','LineWidth',2)
    plot([CIy2 CIy2],ylim,'k:','LineWidth',2)
    hold off
    
    subplot('position',[x0,y0+0*(yw+ys),xw,yw])
    plot(zgrid,Pz,'k','LineWidth',lwd); grid
    xlabel('z (m)');
    ylabel('P(z)');
    hold on
    %plot([xtrue(3) xtrue(3)],[0 0],'rX','MarkerSize',10,'LineWidth',2)
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
    %plot([xtrue(1)],[xtrue(2)],'w+','LineWidth',3,'MarkerSize',12)
    %plot([xtrue(1)],[xtrue(2)],'k+','LineWidth',2,'MarkerSize',8)
    hold off
    
    subplot('position',[x0,y0+1*(yw+ys),xw,yw])    % Plot x-z marginal
    imagesc(xgrid,zgrid,Pxz(:,:)'); grid; colorbar
    set(gca,'YDir','normal');  % Flip so y increases upward
    xlabel('x (m)');
    ylabel('z (m)');
    hold on
    %plot([xtrue(1)],[xtrue(3)],'w+','LineWidth',3,'MarkerSize',12)
    %plot([xtrue(1)],[xtrue(3)],'k+','LineWidth',2,'MarkerSize',8)
    hold off
    
    subplot('position',[x0,y0+0*(yw+ys),xw,yw])    % Plot y-z marginal
    imagesc(ygrid,zgrid,Pyz(:,:)'); grid; colorbar
    set(gca,'YDir','normal');  % Flip so y increases upward
    xlabel('y (m)');
    ylabel('z (m)');
    hold on
    %plot([xtrue(2)],[xtrue(3)],'w+','LineWidth',3,'MarkerSize',12)
    %plot([xtrue(2)],[xtrue(3)],'k+','LineWidth',2,'MarkerSize',8)
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
    
end % doplot

end % function


