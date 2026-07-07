#!/bin/bash
N=11433479; S=11433480
for i in $(seq 1 120); do
  NST=$(squeue -j $N -h -o %t 2>/dev/null); SST=$(squeue -j $S -h -o %t 2>/dev/null)
  if [ "$NST" = "R" ]; then echo "$(date) NORMAL started -> cancel scavenger"; scancel $S; exit 0; fi
  if [ "$SST" = "R" ]; then echo "$(date) SCAVENGER started -> cancel normal? NO - keep normal queued as backup vs preemption"; exit 0; fi
  if [ -z "$NST" ] && [ -z "$SST" ]; then echo "$(date) both gone from queue"; exit 1; fi
  sleep 30
done
echo "$(date) timeout: still pending"
