% -------------------------------------------------------------------
% market_rules.lp  (safe, arity-2)
% -------------------------------------------------------------------
% INPUT FACTS (from Python):
%   anomaly(Target, HourIndex).
%   holiday(HourIndex).
%
% OUTPUT:
%   valid_anomaly(Target, HourIndex).
% -------------------------------------------------------------------

#const min_persist = 3.  % we conceptually use 3h persistence

% -------------------------------------------------------------------
% Rule 1 — Persistence filter:
% Keep only anomalies that persist for at least 3 consecutive hours.
% -------------------------------------------------------------------
persistent(T, G) :-
    anomaly(G, T),
    anomaly(G, T-1),
    anomaly(G, T-2).

% -------------------------------------------------------------------
% Rule 2 — Ignore anomalies on public holidays
% -------------------------------------------------------------------
valid_anomaly(G, T) :-
    persistent(T, G),
    not holiday(T).

% -------------------------------------------------------------------
% SHOW
% -------------------------------------------------------------------
#show valid_anomaly/2.
