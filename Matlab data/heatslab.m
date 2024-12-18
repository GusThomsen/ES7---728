%HEATSLAB  -  heater on a slab
%  Script to demonstrate 3-omega heating effect.
%

clear;
close all;


%% Init -- constants, parameters and settings
HEATER = 1;
SLAB = 2;
ABOVE = 3;
BELOW = 4;
DATAFILE = 'triWdata';
showAnimation = false;
nextPlot = [5 10 100 120];
Nsamples = 2^10;
tend = 10;
fmax = 4;


% Heater (wire)
d = 20e-6;                          % diameter [m]
l = 0.13;                           % length [m]
kh = 77.8;                          % Pt thermal conductivity [W/m K]
rhoh = 21.45e3;                     % Pt mass density [kg/m^3]
ch = 133;                           % Pt specific heat [J/kg K]
resistivity = 10.6e-8;              % Pt [ohm m]

R = l*resistivity/(pi*d^2/4);       % Resistance [ohm] 
I = 75e-3;                          % Current amplitude [A]
dRdT = 0.00385*R;                   % Platinum 
omega = 2*pi*1;                     % applied AC frequency [rad/s]
sigma = 0*sqrt(3e-6);                 % noise standard deviation [V] 

% Slab/substrate
hs = 4.0e-3;                        % thickness [m]
ws = 20.0e-3;                       % width [m]
ks = 100;                           % SiC thermal conductivity [W/m K]
rhos = 3.16e3;                      % SiC mass density [kg/m^3]
cs = 670;                           % SiC specific heat [J/kg K]

% Water/air above
hw = 10.0e-3;                       % thickness [m]
kw = 0.598;                         % H2O thermal conductivity [W/m K]
rhow = 997;                         % H2O mass density [kg/m^3]
cw = 4.2e3;                         % H2O specific heat [J/kg K]
ka = 0.024;                         % air thermal conductivity [W/m K]
rhoa = 1.225;                       % air mass density [kg/m^3]
ca = 1.012e3;                       % air specific heat [J/kg K]

% Water/air below: reuses data from water/air above



%% Create model
thermalmodel = createpde('thermal', 'transient');

% Geometry
slabG = [3; 4; -ws/2; -ws/2; ws/2; ws/2; -hs; 0; 0; -hs];
aboveG = [3; 4; -ws/2; -ws/2; ws/2; ws/2; 0; hw; hw; 0];
belowG = [3; 4; -ws/2; -ws/2; ws/2; ws/2; -hs; -hw; -hw; -hs];
heaterG = [1; 0; d/2; d/2; zeros(length(slabG)-4,1)]; 
geometryDef = [heaterG, slabG, aboveG, belowG];
setFormula = 'heaterG+slabG+aboveG+belowG';
nameSpace = char('heaterG', 'slabG','aboveG','belowG')';
geometryFromEdges(thermalmodel, decsg(geometryDef,setFormula,nameSpace));

% Check that the geometry is specified correctly
figure(1)
pdegplot(thermalmodel,'EdgeLabels','on','FaceLabels','on')
title('Geometry for simulation')
xlabel('x [m]')
ylabel('y [m]')

% Thermal properties
thermalProperties(thermalmodel, 'ThermalConductivity', kh, ...
                                'MassDensity', rhoh, ...
                                'SpecificHeat', ch, ...
                                'Face', HEATER);
thermalProperties(thermalmodel, 'ThermalConductivity', ks, ...
                                'MassDensity', rhos, ...
                                'SpecificHeat', cs, ...
                                'Face', SLAB);
thermalProperties(thermalmodel, 'ThermalConductivity', ka, ...
                                'MassDensity', rhoa, ...
                                'SpecificHeat', ca, ...
                                'Face', ABOVE);
thermalProperties(thermalmodel, 'ThermalConductivity', ka, ...
                                'MassDensity', rhoa, ...
                                'SpecificHeat', ca, ...
                                'Face', BELOW);
                            
% Heat source
internalHeatSource(thermalmodel, @jouleHeating, 'Face', HEATER);

% Boundary and initial conditions
thermalBC(thermalmodel, 'Edge', 1:8, 'Temperature', 0); 
thermalIC(thermalmodel, 0);


%% Generate and plot mesh
mesh = generateMesh(thermalmodel);
figure(2)
pdemesh(thermalmodel)
title('Mesh for simulation')
xlabel('x [m]')
ylabel('y [m]')


%% Solve the heat equation and show results
t = linspace(0,tend,Nsamples);
tic
thermalmodel.SolverOptions.ReportStatistics ='on';
result = solve(thermalmodel,t);
toc
T = result.Temperature;
Tmax = max(max(T));
Tmin = min(min(T));

if showAnimation == true
    plotIdx = 1;
    for j = 1:Nsamples
        figure(3)
        pdeplot(thermalmodel,'XYData',T(:,j),'ZData',T(:,j));
        caxis([Tmin Tmax]);
        axis([-1.1*ws 1.1*ws -1.1*hs 1.1*hw 0 Tmax]);
        axis equal
        title('Temperature field')
        xlabel('x [m]')
        ylabel('y [m]')
        view([0 90])
        drawnow
        %Mv(j) = getframe;
        if j == nextPlot(plotIdx)
            figure(14)
            subplot(2,2,plotIdx)
            pdeplot(thermalmodel,'XYData',T(:,j),'ZData',T(:,j));
            caxis([Tmin Tmax]);
            axis([-1.1*ws 1.1*ws -1.1*hs 1.1*hw 0 Tmax]);
            axis equal
            title(['Temperature field, t = ',num2str(t(j)),' s'])
            xlabel('x [m]')
            ylabel('y [m]')
            drawnow
            if plotIdx < length(nextPlot), plotIdx = plotIdx + 1; end
        end
    end
    
    %movie(Mv,1)
end


%% Compute average temperature of heater for each time step
heaterNodes = findNodes(mesh,'region','Face',HEATER);
Th = sum(T(heaterNodes,:))/length(heaterNodes);
figure(5)
plot(t, Th)
title('Average heater temperature (above ambient)')
ylabel('Temperature Th [K]')
xlabel('Time t [s]')


%% Compute voltage drop in heater with added measurement noise
dT = Th - sum(Th(Nsamples/2:end))/(Nsamples/2);
V = I*exp(1i*omega*t).*(R + dRdT*dT) + sigma*randn(size(Th));
V3w = V - I*R*exp(1i*omega*t);
figure(6)
subplot(211)
plot(t, V)
title('Heater voltage')
ylabel('Voltage V [V]')
subplot(212)
plot(t, V3w)
title('Heater voltage, I_0 R_0 exp(i\omega t) subtracted')
ylabel('Voltage V [V]')
xlabel('Time t [s]')

figure(7)
Nfft = 2^nextpow2(Nsamples);
f = (Nsamples/tend)*linspace(0,1,Nfft/2+1)/2;
fmaxIdx = find(f>fmax, 1);
Vfreq = abs(fft(V,Nfft))/Nsamples;
semilogy(f(1:fmaxIdx), 2*Vfreq(1:fmaxIdx))
xlabel('Frequency [Hz]')
ylabel('Amplitude [V]')
title(['Sampling freq. ', num2str(Nsamples/tend), ' Hz, N_s = ', ...
       num2str(Nsamples)])

filename = strcat(DATAFILE, 'fs', num2str(round(Nsamples/tend)),'.dat');
save(filename, 'V', '-ascii');
