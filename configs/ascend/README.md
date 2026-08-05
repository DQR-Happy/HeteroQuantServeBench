# S09 Ascend campaign configuration

`campaign_spec.yaml` freezes the evidence gate and browser contract. It does
not select or guess an Ascend product/version for the current Jetson target.

`scope_adr.yaml` deactivates optional E09-06 while the repository makes no
Ascend INT8/INT4 claim. A real low-precision claim must satisfy the listed
reactivation conditions and create a new formal run; it must not rewrite the
blocked evidence collected on Jetson.
