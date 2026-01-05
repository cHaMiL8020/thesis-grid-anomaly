% -------------------------------------------------------------------
% market_rules.lp (Enhanced Neuro-Symbolic Logic)
% -------------------------------------------------------------------
% INPUTS (Factored from Step 07):
%   anomaly(Target, T).         (Statistical flag from dCeNN-ELM)
%   holiday(T).                 (Public holiday calendar facts)
%   pred(Feature, Value, T).    (Physical/Market variables from Step 01/07)
% -------------------------------------------------------------------

#const min_persist = 3.
#const wind_ramp_limit = 500.  % MW/h threshold for Austrian fleet plausibility

% -------------------------------------------------------------------
% RULE 1: PERSISTENCE FILTER
% -------------------------------------------------------------------
% Keep only anomalies that persist for at least 3 consecutive hours.
persistent(G, T) :- 
    anomaly(G, T), 
    anomaly(G, T-1), 
    anomaly(G, T-2).

% -------------------------------------------------------------------
% RULE 2: PHYSICS-BASED CONSTRAINTS (VRE RAMP & NIGHT LOGIC)
% -------------------------------------------------------------------
% Physics check: Solar anomalies are invalid if radiation is zero (Night).
% This filters out neural network noise during hours with no production.
invalid_solar(T) :- 
    anomaly("CF_Solar", T),
    pred(shortwave_radiation, R, T),
    R <= 0.

% Physical Plausibility: Flag Wind anomalies that exceed a physical ramp-rate.
% Excessive jumps (e.g. >500MW/h) are treated as sensor or model errors.
excessive_wind_ramp(T) :-
    pred(wind_mw, V1, T),
    pred(wind_mw, V0, T-1),
    |V1 - V0| > wind_ramp_limit.

% -------------------------------------------------------------------
% RULE 3: CO-ANOMALY LOGIC (PRICE-LOAD CORRELATION)
% -------------------------------------------------------------------
% A Price anomaly is "Confirmed" if it coincides with a Load anomaly,
% indicating a genuine supply/demand event rather than market volatility.
confirmed_economic_spike(T) :-
    anomaly("Price", T),
    anomaly("Load_MW", T).

% -------------------------------------------------------------------
% RULE 4: FINAL REFINEMENT (THE "FUSION")
% -------------------------------------------------------------------
% An anomaly is valid if it is persistent, not a holiday, 
% and not physically impossible based on the rules above.
valid_anomaly(G, T) :-
    persistent(G, T),
    not holiday(T),
    not invalid_solar(T),
    not excessive_wind_ramp(T).

% Special Case: Prioritize Price anomalies that have co-anomaly confirmation.
% These are kept even if they don't meet the standard persistence criteria.
valid_anomaly("Price", T) :-
    confirmed_economic_spike(T),
    not holiday(T).

% -------------------------------------------------------------------
% SHOW ATOMS
% -------------------------------------------------------------------
#show valid_anomaly/2.