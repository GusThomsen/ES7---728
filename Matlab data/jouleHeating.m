%q = jouleHeating(location, state)
%  Computes time-varying heating of the heater element.
%
%  The solver passes location data as a structure array with the fields 
%  location.x and location.y. The state data is a structure array with the 
%  fields state.u, state.ux, state.uy, and state.time. 
%  The state.u field contains the solution vector. 
%  The state.ux, state.uy, state.uz fields are estimates of the solution’s 
%  partial derivatives at the corresponding points of the location structure. 
%  The state.time field contains time at evaluation points.
%
function q = jouleHeating(location, state)
    x = location.x;
    y = location.y;
    t = state.time;
    q = zeros(1,numel(x));
    if(isnan(state.time))
        % Returning a NaN when time=NaN tells the solver that the heat source 
        % is a function of time.
        q(1,:) = NaN;
        return
    end
    
    % NB! The following constants must be updated whenever the geometry
    % changes!
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

    h = 1.2e-4;                        % heater thickness [m]
    w = 1.2e-4;                        % heater width [m]
    R = 45; %l*resistivity/(hh*wh); 
    I = 75e-3;
    P = 1000*R*I^2;                    % artificially inflated peak heat power [W]
    k = 77.8;                          % heat conductivity [W/m K]
    omega = 2*pi*1;                    % applied AC frequency [rad/s]
    q(1,:) = P*(1 - cos(2*omega*t))/(2*pi*d*d*k);
end