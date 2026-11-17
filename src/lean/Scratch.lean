/-
Copyright (c) 2026 Wayne Brassem. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Wayne Brassem
-/

import StandingWave.CausalDistance
import StandingWave.JointModes

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
#check oscillator_magnitude_sq
#check compose_rotation
#check inverse_rotation

/-!
## Quadratic Magnitude Verification

Inspect the quadratic magnitude, its positive-definiteness
properties, and its invariance under normalized rotations.
-/

#check StandingWave.oscillator_magnitude_sq_nonneg
#check StandingWave.oscillator_magnitude_sq_eq_zero_iff
#check StandingWave.oscillator_magnitude_sq_rotate
#check StandingWave.apply_rotation_magnitude_sq
#check StandingWave.rotate_sequence_magnitude_sq

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

#check Function.Injective
#check Function.Surjective
#check MulEquiv

/-!
## Verification Experiments

Check that the group operations can be used through
their standard Mathlib interfaces.
-/

example (r₁ r₂ : NormalizedRotation) :
    r₁ * r₂ = r₂ * r₁ := by
  exact mul_comm r₁ r₂

example (r : NormalizedRotation) :
    r⁻¹ * r = 1 := by
  exact inv_mul_cancel r

example (r₁ r₂ r₃ : NormalizedRotation) :
    (r₁ * r₂) * r₃ = r₁ * (r₂ * r₃) := by
  exact mul_assoc r₁ r₂ r₃

/-!
## SO(2) Group Isomorphism Verification

Verify that the correspondence between normalized rotations
and the matrix special orthogonal group is available and
preserves the group operations.
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
## Pairwise Oscillator Coupling Examples

Explore concrete instances of interactions between two
oscillator states.

These examples illustrate redistribution of quadratic
magnitude under normalized mixing. They are computational
checks rather than general conservation theorems.

The underlying conservation properties are established
in `StandingWave.Coupling`.
-/

/-
### Complete Magnitude Redistribution

Normalized mixing with coefficients (0, 1) transfers
the entire quadratic magnitude from the first oscillator
to the second.

The combined quadratic magnitude remains conserved.
-/
example :
    let r : NormalizedRotation :=
      ⟨0, 1, by norm_num⟩
    let A : OscillatorState := ⟨1, 0⟩
    let B : OscillatorState := ⟨0, 0⟩
    let output := mix_oscillators r (A, B)
    oscillator_magnitude_sq output.1 = 0 ∧
      oscillator_magnitude_sq output.2 = 1 ∧
      combined_magnitude_sq output.1 output.2 =
        combined_magnitude_sq A B := by
  norm_num [mix_oscillators, oscillator_magnitude_sq,
    combined_magnitude_sq]

/-
### Partial Magnitude Redistribution

Normalized mixing with coefficients (3/5, 4/5)
redistributes the initial unit quadratic magnitude
between both oscillator states.

The resulting individual magnitudes are 9/25 and
16/25, while their combined magnitude remains one.

Neither output oscillator has zero magnitude.
-/
example :
    let r : NormalizedRotation :=
      ⟨3 / 5, 4 / 5, by norm_num⟩
    let A : OscillatorState := ⟨1, 0⟩
    let B : OscillatorState := ⟨0, 0⟩
    let output := mix_oscillators r (A, B)
    oscillator_magnitude_sq output.1 = 9 / 25 ∧
      oscillator_magnitude_sq output.2 = 16 / 25 ∧
      combined_magnitude_sq output.1 output.2 =
        combined_magnitude_sq A B := by
  norm_num [mix_oscillators, oscillator_magnitude_sq,
    combined_magnitude_sq]

/-!
## Oscillator Interaction Monoid Examples

Check that the standard monoid interface is available
for oscillator interactions.
-/

example (F G H : OscillatorInteraction) :
    (F * G) * H = F * (G * H) := by
  exact mul_assoc F G H

/- JointModes verification -/

/-!
## Scratch Experiment: Two Different Phase Flips

Starting with the separable matrix

    M = !![4, 6; 2, 3]

we first negate the lower-right entry (d).

This produces a nonseparable state.

We then negate the upper-right entry (b),
without undoing the first flip.

The resulting state is again separable,
although it is not equal to the original state.
-/

section DoubleFlipExperiment

open StandingWave

-- The initial separable state.
private noncomputable def walkM : JointModeState :=
  !![4, 6; 2, 3]

-- Negate entry b = M 0 1.
private def walkFlipB (M : JointModeState) :
    JointModeState :=
  fun i j =>
    if i = 0 ∧ j = 1 then -(M i j) else M i j

-- First negate d, using our existing operation.
private noncomputable def walkAfterD : JointModeState :=
  joint_mode_phase_flip walkM

-- Then negate b, leaving d negative.
private noncomputable def walkAfterDB : JointModeState :=
  walkFlipB walkAfterD

-- Initial determinant: 12 - 12 = 0.
example : Matrix.det walkM = 0 := by
  norm_num [Matrix.det_fin_two, walkM]

-- After flipping d: -12 - 12 = -24.
example : Matrix.det walkAfterD = -24 := by
  norm_num [Matrix.det_fin_two, walkAfterD,
    joint_mode_phase_flip, walkM]

-- After also flipping b: -12 - (-12) = 0.
example : Matrix.det walkAfterDB = 0 := by
  norm_num [Matrix.det_fin_two, walkAfterDB,
    walkFlipB, walkAfterD,
    joint_mode_phase_flip, walkM]

-- The original state is separable.
example : joint_mode_separable walkM := by
  apply (joint_mode_separable_iff_det_eq_zero walkM).2
  norm_num [Matrix.det_fin_two, walkM]

-- The state after one flip is nonseparable.
example : ¬ joint_mode_separable walkAfterD := by
  intro h
  have hdet :=
    (joint_mode_separable_iff_det_eq_zero walkAfterD).1 h
  norm_num [Matrix.det_fin_two, walkAfterD,
    joint_mode_phase_flip, walkM] at hdet

-- The state after two DIFFERENT flips is separable.
example : joint_mode_separable walkAfterDB := by
  apply (joint_mode_separable_iff_det_eq_zero walkAfterDB).2
  norm_num [Matrix.det_fin_two, walkAfterDB,
    walkFlipB, walkAfterD,
    joint_mode_phase_flip, walkM]

-- But the final state is NOT the original matrix.
example : walkAfterDB ≠ walkM := by
  intro h
  have hb :=
    congrArg (fun M : JointModeState => M 0 1) h
  norm_num [walkAfterDB, walkFlipB, walkAfterD,
    joint_mode_phase_flip, walkM] at hb

end DoubleFlipExperiment

#check StandingWave.joint_mode_flip_sequence_odd_separable_iff
#check StandingWave.joint_mode_mix_oscillators_det

/-!
## Axiom Audit

Inspect the logical dependencies of the principal results.

The expected dependencies are standard foundational axioms
used by Lean and Mathlib, rather than additional axioms
introduced specifically for this development.
-/

#print axioms StandingWave.oscillator_magnitude_sq_nonneg
#print axioms StandingWave.oscillator_magnitude_sq_eq_zero_iff
#print axioms StandingWave.oscillator_magnitude_sq_rotate
#print axioms StandingWave.apply_rotation_magnitude_sq
#print axioms StandingWave.rotate_sequence_magnitude_sq

#print axioms StandingWave.compose_rotation_assoc
#print axioms StandingWave.compose_rotation_comm

#print axioms StandingWave.rotation_to_so2_mul
#print axioms StandingWave.normalized_rotation_equiv_so2

#print axioms StandingWave.execute_history_eq_of_equivalent
#print axioms StandingWave.history_equivalent_has_precedence_equiv

#print axioms StandingWave.mix_oscillators_conserves
#print axioms StandingWave.scale_half_exhibits_loss
#print axioms StandingWave.conservative_interactions_lt_nonincreasing

#print axioms StandingWave.interaction_hop_distance_triangle
#print axioms StandingWave.execute_history_difference_bounded_by_length
#print axioms StandingWave.execute_history_single_source_no_influence_beyond_length
#print axioms StandingWave.execute_history_difference_support_subset_ball

-- CausalDistance: structural connectivity
#print axioms StandingWave.interaction_reachable_equivalence
#print axioms StandingWave.execute_history_no_cross_component_influence

-- CausalDistance: extended metric
#print axioms StandingWave.interaction_hop_distance_eq_zero_iff
#print axioms StandingWave.interaction_hop_distance_eq_top_iff

-- CausalDistance: one-event propagation
#print axioms StandingWave.apply_admissible_event_difference_within_one_hop

-- JointModes: joint mode
#print axioms StandingWave.joint_mode_flip_sequence_odd_separable_iff
#print axioms StandingWave.joint_mode_mix_oscillators_det
