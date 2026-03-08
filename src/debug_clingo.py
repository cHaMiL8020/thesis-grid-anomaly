import clingo
ctl = clingo.Control()
# Manually add a "Solar Anomaly at Night" fact
test_facts = 'anomaly("cf_solar", 100). pred(shortwave_radiation, 0, 100).'
ctl.add("base", [], test_facts)
ctl.load("rules/market_rules.lp")
ctl.ground([("base", [])])
ctl.solve(on_model=lambda m: print("Found valid anomaly:", m))