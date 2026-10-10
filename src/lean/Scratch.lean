/-
Copyright (c) 2026 Wayne Brassem. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Wayne Brassem
-/

import Mathlib
import StandingWave.Oscillator

/-!
# Standing Wave — Scratch

Experimental workspace for inspecting Mathlib declarations,
testing examples, and developing proof strategies.

Nothing in this file is part of the formal theory.
Successful results should be moved to their appropriate
StandingWave module.
-/

open StandingWave

/-!
## Inspecting Definitions and Types
-/

#check OscillatorState
#check NormalizedRotation
#check oscillator_energy
#check compose_rotation
#check inverse_rotation

/-!
## Group Operations
-/

#check (1 : NormalizedRotation)
#check (fun r : NormalizedRotation => r⁻¹)
#check (fun r₁ r₂ : NormalizedRotation => r₁ * r₂)

#check mul_assoc
#check mul_comm
#check inv_mul_cancel

/-!
## Matrix Representation
-/

#check Matrix
#check Matrix.det
#check Matrix.transpose
#check Matrix.specialOrthogonalGroup

#check Matrix.mem_specialOrthogonalGroup_fin_two_iff
#print Matrix.mem_specialOrthogonalGroup_fin_two_iff

#check Matrix.specialOrthogonalGroup
#check Function.Injective
#check Function.Surjective
#check MulEquiv

/-!
## Verification Experiments
-/

example (r₁ r₂ : NormalizedRotation) :
    r₁ * r₂ = r₂ * r₁ := by
  exact mul_comm r₁ r₂

example (r : NormalizedRotation) :
    r⁻¹ * r = 1 := by
  exact inv_mul_cancel r

/-!
## SO(2) Group Isomorphism Verification

Verify that the correspondence between our normalized
rotations and the matrix special orthogonal group is
available and preserves the group operations.
-/

#check StandingWave.rotation_to_so2
#check StandingWave.so2_to_rotation

#check StandingWave.rotation_matrix_injective
#check StandingWave.so2_to_rotation_rotation_to_so2
#check StandingWave.rotation_to_so2_so2_to_rotation
#check StandingWave.rotation_to_so2_mul

#check StandingWave.normalized_rotation_equiv_so2

/-!
### Verification of the Group Isomorphism
-/

/--
The isomorphism preserves multiplication.
-/
example (r₁ r₂ : NormalizedRotation) :
    normalized_rotation_equiv_so2 (r₁ * r₂) =
      normalized_rotation_equiv_so2 r₁ *
        normalized_rotation_equiv_so2 r₂ := by
  exact normalized_rotation_equiv_so2.map_mul r₁ r₂

/--
The isomorphism preserves the identity.
-/
example :
    normalized_rotation_equiv_so2 (1 : NormalizedRotation) =
      1 := by
  exact normalized_rotation_equiv_so2.map_one

/--
The isomorphism preserves inverses.
-/
example (r : NormalizedRotation) :
    normalized_rotation_equiv_so2 r⁻¹ =
      (normalized_rotation_equiv_so2 r)⁻¹ := by
  exact normalized_rotation_equiv_so2.map_inv r

/-!
### Axiom Audit
-/

#print axioms StandingWave.oscillator_energy_rotate
#print axioms StandingWave.rotate_sequence_energy
#print axioms StandingWave.compose_rotation_assoc
#print axioms StandingWave.compose_rotation_comm
#print axioms StandingWave.rotation_to_so2_mul
#print axioms StandingWave.normalized_rotation_equiv_so2
