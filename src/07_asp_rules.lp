% 07_asp_rules.lp
%
% =========================================================
% 07_asp_rules.lp - Neuro-Symbolic Refinement Logic
% =========================================================

% --- Utility: Absolute Value ---
abs_val(D, A) :- D >= 0, A = D.
abs_val(D, A) :- D <  0, A = -D.

% =========================================================
% 1. Physical Plausibility Guards (Veto Conditions)
% =========================================================

% Night PV Rule: Solar anomalies are invalid if radiation is zero or negative.
% Factored from: pred(shortwave_radiation, R, T)
night(T) :- 
    pred(shortwave_radiation, R, T), 
    R <= 0.

invalid_solar(T) :- 
    anomaly("cf_solar", T), 
    night(T).

% Wind Guard: Flag as implausible if wind speed is near zero but anomaly is high.
% Factored from: pred(wind_speed_100m, V, T)
invalid_wind(T) :-
    anomaly("cf_wind", T),
    pred(wind_speed_100m, V, T), 
    V < 1.

% Load Ramp Guard: Large sudden jumps are often sensor errors rather than grid events.
% Threshold set to 2000 MW based on Austrian grid capacity.
ramp_violation(T) :-
    pred(dload_pred, D, T),
    abs_val(D, A),
    A > 2000.

% =========================================================
% 2. Market & Calendar Logic
% =========================================================

% Holiday Exception: Low load is expected on public holidays.
% Relax anomalies if load is significantly lower than the reference median.
holiday_exception(T) :-
    anomaly("load_mw", T),
    holiday(T),
    pred(load_mw, L, T),
    pred(load_ref, R, T),
    L < R.

% Co-Anomaly Confirmation: Price spikes are highly credible if they coincide with Load spikes.
confirmed_economic_event(T) :-
    anomaly("price", T),
    anomaly("load_mw", T).

% =========================================================
% 3. Final Refinement (The "Neuro-Symbolic Fusion")
% =========================================================

% An anomaly is "valid" if it is suggested by the Neural layer AND 
% passes all Symbolic plausibility checks.
valid_anomaly(Name, T) :-
    anomaly(Name, T),
    not invalid_solar(T),
    not invalid_wind(T),
    not holiday_exception(T),
    not ramp_violation(T).

% Priority Rule: Co-anomalies are valid even if they might otherwise be flagged.
valid_anomaly("price", T) :-
    confirmed_economic_event(T).

% =========================================================
% 4. Explainable AI (XAI) Output
% =========================================================

% Define a veto for the visualization script to track "The Why"
vetoed(Name, T) :- 
    anomaly(Name, T), 
    not valid_anomaly(Name, T).

% Specific veto reasons for diagnostic reporting
veto_reason(T, "Night-time Solar Noise") :- invalid_solar(T).
veto_reason(T, "Low Wind Plausibility") :- invalid_wind(T).
veto_reason(T, "Holiday Pattern")        :- holiday_exception(T).
veto_reason(T, "Physical Ramp Limit")    :- ramp_violation(T).

#show valid_anomaly/2.
#show vetoed/2.
