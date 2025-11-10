% Facts you will pass per hour as:
%   hour(T).
%   pred(Name, Value, T).  % predicted values
%   resid(Name, Abs, T).   % absolute residual for target Name at time T
%   anom(Name, T).         % model says anomaly for target
%   bucket(holiday,0 or 1,T). % context
%
% Threshold facts can be encoded as:
%   thr(Name, Thresh).

% --- Night PV rule: if night, PV anomalies are true; if day and PV near zero, downweight.
night(T) :- pred(solar_elevation, E, T), E < 0.

flag(solar_day_violation,T) :- not night(T), pred(CF_Solar, V, T), V < 0.02.

% --- Wind zero-wind guard: if wind speed near zero but CF_Wind high, flag
flag(wind_implausible,T) :- pred(wind_speed_100m, V, T), V < 0.5, pred(CF_Wind, C, T), C > 0.2.

% --- Load ramp guard: |ΔLoad| > RAMP
ramp_violation(T) :- pred(dLoad_pred, D, T), abs(D) > 2000.  % tune threshold

% --- Holiday: allow lower loads without penalty
holiday_lowload_ok(T) :- bucket(holiday,1,T), pred(Load_MW, L, T), L < 0.7 * pred(Load_ref, R, T).

% --- Refine anomaly: keep if anomaly and plausibility flags rise
keep(Name,T) :- anom(Name,T), ( flag(solar_day_violation,T); flag(wind_implausible,T); ramp_violation(T) ).
% --- Drop anomaly if holiday-related low load
drop(Load_MW,T) :- anom(Load_MW,T), holiday_lowload_ok(T).

% Default: if an anomaly and not dropped, then final anomaly
final(Name,T) :- anom(Name,T), not drop(Name,T).
