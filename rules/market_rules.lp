% -------------------------------------------------------------------
% market_rules.lp (Corrected & Enhanced for Explainable AI)
% -------------------------------------------------------------------

#const min_persist = 3.
#const wind_ramp_limit = 500.

% --- RULE 1: PERSISTENCE FILTER ---
persistent(G, T) :- 
    anomaly(G, T), 
    anomaly(G, T-1), 
    anomaly(G, T-2).

% --- RULE 2: PHYSICS-BASED CONSTRAINTS (Lowercase names) ---
invalid_solar(T) :- 
    anomaly("cf_solar", T),
    pred(shortwave_radiation, R, T),
    R <= 0.

excessive_wind_ramp(T) :-
    anomaly("cf_wind", T),
    pred(wind_mw, V1, T),
    pred(wind_mw, V0, T-1),
    |V1 - V0| > wind_ramp_limit.

% --- RULE 3: CO-ANOMALY LOGIC (Price-Load Correlation) ---
confirmed_economic_spike(T) :-
    anomaly("price", T),
    anomaly("load_mw", T).

% --- RULE 4: FINAL REFINEMENT (The "Fusion") ---
valid_anomaly(G, T) :-
    persistent(G, T),
    not holiday(T),
    not invalid_solar(T),
    not excessive_wind_ramp(T).

% Priority Rule: Co-anomalies bypass persistence check
valid_anomaly("price", T) :-
    confirmed_economic_spike(T),
    not holiday(T).

% --- RULE 5: EXPLAINABLE AI (Veto Tracking) ---
% This allows your visualization scripts to identify and categorize rejections
vetoed(G, T) :- anomaly(G, T), not valid_anomaly(G, T).

veto_reason(T, "night_solar") :- invalid_solar(T).
veto_reason(T, "wind_ramp_limit") :- excessive_wind_ramp(T).
veto_reason(T, "holiday_calendar") :- anomaly(G, T), holiday(T).

% --- SHOW ATOMS ---
#show valid_anomaly/2.
#show vetoed/2.
#show veto_reason/2.