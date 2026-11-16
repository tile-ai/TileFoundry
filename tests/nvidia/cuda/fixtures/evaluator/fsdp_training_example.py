"""The FSDP training example at one and two units, with both residency policies."""

from examples.fsdp_training.model import training_program

Release1 = training_program(1, retain=False)
Retain1 = training_program(1, retain=True)
Release2 = training_program(2, retain=False)
Retain2 = training_program(2, retain=True)
