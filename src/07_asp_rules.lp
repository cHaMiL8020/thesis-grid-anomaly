% 07_asp_rules.lp
%
% Facts expected per hour T:
%
%   hour(T).
%   pred(solar_elevation, E, T).      % synthetic: >0 day, <0 night
%   pred(cf_solar,        V, T).      % capacity factor for solar (predicted or true)
%   pred(cf_wind,         C, T).      % capacity factor for wind (predicted or true)
%   pred(load_mw,         L, T).      % realized load [MW]
%   pred(load_ref,        R, T).      % reference/expected load [MW]
%   pred(dload_pred,      D, T).      % predicted load ramp [MW]
%   pred(wind_speed_100m, Vw, T).     % wind speed at 100m [m/s]
%
%   anom(Name, T).                    % model says anomaly for target Name at time T
%   bucket(holiday, H, T).            % H = 0 or 1: is public holiday
%
% Optionally:
%   resid(Name, Abs, T).              % absolute residual (not used in rules yet)
%   thr(Name, Thresh).                % thresholds (not used yet)
%
% Target names (Name) are lowercase atoms:
%   cf_solar, cf_wind, load_mw, price
%
% Output:
%   final(Name, T).                   % rule-consistent anomaly

% =========================================================
% Utility predicates
% =========================================================

% abs_val(D, A): absolute value helper
abs_val(D, A) :- D >= 0, A = D.
abs_val(D, A) :- D <  0, A = -D.

% =========================================================
% Night PV rule
% =========================================================
% If sun is below the horizon, we consider it "night".
night(T) :-
    pred(solar_elevation, E, T),
    E < 0.

% If it is daytime and CF_Solar is near zero, flag this as suspicious.
flag(solar_day_violation, T) :-
    not night(T),
    pred(cf_solar, V, T),
    V < 0.02.

% =========================================================
% Wind zero-wind guard
% =========================================================
% If 100m wind speed is very low but CF_Wind is high, flag as implausible.
flag(wind_implausible, T) :-
    pred(wind_speed_100m, V, T), V < 0.5,
    pred(cf_wind,         C, T), C > 0.2.

% =========================================================
% Load ramp guard
% =========================================================
% Large predicted ramp: |ΔLoad| > 2000 MW (tunable).
ramp_violation(T) :-
    pred(dload_pred, D, T),
    abs_val(D, A),
    A > 2000.

% =========================================================
% Holiday low-load rule
% =========================================================
% On public holidays, low load vs. reference is OK (we relax anomalies).
holiday_lowload_ok(T) :-
    bucket(holiday, 1, T),
    pred(load_mw,  L, T),
    pred(load_ref, R, T),
    L < 0.7 * R.

% =========================================================
% Refined anomaly logic
% =========================================================

% "Keep" anomaly if model flags anomaly and any plausibility guard fires.
keep(Name, T) :-
    anom(Name, T),
    (   flag(solar_day_violation, T)
    ;   flag(wind_i_
