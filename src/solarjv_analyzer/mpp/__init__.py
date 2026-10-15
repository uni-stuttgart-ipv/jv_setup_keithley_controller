"""Maximum-power-point tracking.

A perturb-and-observe tracker: sit at a voltage, step away from it, and keep
the step only if the power improved. Unlike SPO — which holds one voltage and
watches it drift — MPP follows the maximum as the cell changes.

The package is deliberately self-contained, like `spo/`: the procedure talks
to the Keithley through the helpers it inherits from `JVProcedure`, and the
report writes through `store.finalize`, so nothing here re-implements
instrument control or file publishing.
"""
