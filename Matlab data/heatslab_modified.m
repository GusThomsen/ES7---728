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
sigma = 0*sqrt(3e-6);               % noise standard deviation [V] 

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
% Generate and plot mesh with refined settings
mesh = generateMesh(thermalmodel);

% Plot the mesh
figure(2)
pdemesh(thermalmodel)
title('Mesh for simulation')
xlabel('x [m]')
ylabel('y [m]')

% Extract the node coordinates from the mesh
nodeCoordinates = mesh.Nodes;

% Define the square region bounds
x_min = -0.002;
x_max = 0.002;
y_min = -0.002;
y_max = 0.002;

% Find nodes within the specified bounds
nodesInSquare = (nodeCoordinates(1, :) >= x_min & nodeCoordinates(1, :) <= x_max) & ...
                (nodeCoordinates(2, :) >= y_min & nodeCoordinates(2, :) <= y_max);

% Extract the coordinates of nodes within the square
nodesInSquareCoords = nodeCoordinates(:, nodesInSquare);

% Set a minimum distance threshold for uniqueness
minDistance = 1e-4;

% Initialize an array to store unique points
uniquePoints = nodesInSquareCoords(:, 1);  % Start with the first point

% Loop through each point and add it if it is not too close to existing unique points
for i = 2:size(nodesInSquareCoords, 2)
    % Calculate distances from the current point to all unique points
    distances = sqrt(sum((uniquePoints - nodesInSquareCoords(:, i)).^2, 1));
    
    % Check if this point is farther than the minimum distance from all unique points
    if all(distances > minDistance)
        uniquePoints = [uniquePoints, nodesInSquareCoords(:, i)];  % Add the point to uniquePoints
    end
end

% Plot the thermal nodes as dots on top of the mesh
hold on
plot(nodeCoordinates(1, :), nodeCoordinates(2, :), 'r.', 'MarkerSize', 10)

% Highlight the nodes within the square region with a smaller marker size
plot(nodeCoordinates(1, nodesInSquare), nodeCoordinates(2, nodesInSquare), 'bo', 'MarkerSize', 5)

% Plot the square region outline for reference
plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'k-', 'LineWidth', 1.5)

hold off
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
        axis([-1.1*ws 1.1*ws -1.1*hs 1.1*hs 0 Tmax]);
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
            axis([-1.1*ws 1.1*ws -1.1*hs 1.1*hs 0 Tmax]);
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
%heaterNodes = findNodes(mesh,'region','Face',HEATER);
%Th = sum(T(heaterNodes,:))/length(heaterNodes);
%figure(5)
%plot(t, Th)
%title('Average heater temperature (above ambient)')
%ylabel('Temperature Th [K]')
%xlabel('Time t [s]')

%% Compute average temperature of heater for each time step
ambient_temperature = 293.15;
heaterNodes = findNodes(mesh, 'region', 'Face', HEATER);
Th = sum(T(heaterNodes, :)) / length(heaterNodes) + ambient_temperature;

save('heater.mat', 'Th');

% Plot the average heater temperature with data points
figure(5)
plot(t, Th, '-b', 'LineWidth', 1.5) % Plot the temperature as a line
hold on
plot(t, Th, 'ro', 'MarkerSize', 3)  % Overlay data points as red circles
title('Average Heater Temperature (above ambient)')
ylabel('Temperature Th [K]')
xlabel('Time t [s]')
legend('Temperature Response', 'Data Points')
hold off

% Find the nearest mesh node to the point (X, Y)
targetPoint = [0.0020; 0.0018];
[~, targetNode] = min(vecnorm(mesh.Nodes(1:2,:) - targetPoint, 2, 1));

% Extract the temperature at the target node over time
T_target = T(targetNode, :); % - ambient_temperature;

T_heater_max = max(Th);
T_target_max = max(T_target);

% Plot the temperature at the target point over time
figure(8)
plot(t, T_target)
title('Temperature at (0.002, 0.0018) over time')
ylabel('Temperature T [K]')
xlabel('Time t [s]')

%% Initialize a matrix to store the temperatures at each unique point
numUniquePoints = size(uniquePoints, 2);
temperatureMatrix = zeros(numUniquePoints, length(t));  % Rows: unique points, Columns: time steps

% Loop through each unique point to find the closest node in the mesh
for i = 1:numUniquePoints
    % Find the closest mesh node to the current unique point
    [~, closestNode] = min(vecnorm(mesh.Nodes(1:2,:) - uniquePoints(:, i), 2, 1));
    
    % Extract the temperature at the closest node over time
    temperatureMatrix(i, :) = T(closestNode, :);
end

% Create a table with the coordinates and temperatures over time
% Convert uniquePoints to a table with columns 'X' and 'Y'
coordsTable = array2table(uniquePoints', 'VariableNames', {'X', 'Y'});

% Convert temperatureMatrix to a table, with each column named by its time index
timeCols = arrayfun(@(i) sprintf('T_t%d', i), 1:length(t), 'UniformOutput', false);
tempTable = array2table(temperatureMatrix, 'VariableNames', timeCols);

% Combine coordinates and temperatures into one table
fullTable = [coordsTable, tempTable];

% Optional: Save the table to a CSV file for later analysis
writetable(fullTable, 'UniquePoints_TemperatureOverTime.csv');


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

I_vector = I * ones(size(t));

figure(10)
subplot(211)
plot(t, V)
title('Heater voltage')
ylabel('Voltage V [V]')
subplot(212)
plot(t, I_vector)
title('Current vs. Time')
ylabel('Current I [A]')
xlabel('Time t [s]')

Voltage_Current_Matrix = [I_vector;
                          V;
                          V3w;];

% Step 1: Get variable names from fullTable
variableNames = fullTable.Properties.VariableNames;

% Step 2: Create Voltage_Current_Matrix with zeros and convert to table
Voltage_Current_Matrix_table = array2table([zeros(3, 2), Voltage_Current_Matrix]);

% Step 3: Assign the same variable names to Voltage_Current_Matrix_table
Voltage_Current_Matrix_table.Properties.VariableNames = variableNames;

% Step 5: Concatenate the tables vertically
combinedTable = [fullTable; Voltage_Current_Matrix_table];

save('combinedTable.mat', 'combinedTable');

%% 
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
