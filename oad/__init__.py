"""Online Action Derivation (OAD) -- reference implementation.

Modules:
  skeleton, confirm, regularity, gate, engine  -- the system (CPU-only, no trained model)
  workload                                      -- synthetic families + ground-truth oracle
  teacher                                       -- OracleTeacher / NoisyTeacher / OpenAITeacher
  harness                                       -- out-of-loop referee
  experiments_phase1 / _phase2 / _rigor         -- experiment drivers

See README.md and LIMITATIONS.md.
"""
from . import skeleton, confirm, regularity, gate, engine, workload, teacher, harness
