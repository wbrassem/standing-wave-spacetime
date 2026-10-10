/-
Copyright (c) 2026 Wayne Brassem. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Wayne Brassem
-/

import Mathlib

/-!

# Primitive Oscillatory States

This module introduces an abstract oscillatory state and a
rotation-based evolution operation.

No physical spacetime, characteristic length, time interval,
or propagation speed is assumed.

The initial objective is to establish mathematical invariants
that may later support a physical interpretation.

-/

namespace StandingWave

/--
A primitive oscillatory state with two real components.

The components are abstract state coordinates. They are not
yet identified with physical position or momentum.
-/
@[ext]
structure OscillatorState where
  q : ℝ
  p : ℝ

/--
A nonnegative quadratic quantity associated with an
oscillatory state.

Its interpretation as physical energy is deliberately
left open.
-/
def oscillator_energy (x : OscillatorState) : ℝ :=
  x.q ^ 2 + x.p ^ 2

/--
A single rotational update of the oscillator state.

The coefficients `c` and `s` are initially arbitrary.
The condition `c² + s² = 1` is imposed separately
when proving invariance.
-/
def oscillator_rotate
    (x : OscillatorState)
    (c s : ℝ) : OscillatorState :=
  ⟨c * x.q - s * x.p,
   s * x.q + c * x.p⟩

/--
The quadratic oscillator quantity is nonnegative.
-/
theorem oscillator_energy_nonneg
    (x : OscillatorState) :
    0 ≤ oscillator_energy x := by
  unfold oscillator_energy
  positivity

/--
A normalized rotational update preserves the quadratic
oscillator invariant.

This is an exact algebraic result. No assumptions about
physical spacetime or microscopic scale are required.
-/
theorem oscillator_energy_rotate
    (x : OscillatorState)
    (c s : ℝ)
    (h : c ^ 2 + s ^ 2 = 1) :
    oscillator_energy (oscillator_rotate x c s) =
      oscillator_energy x := by
  change
    (c * x.q - s * x.p) ^ 2 +
      (s * x.q + c * x.p) ^ 2 =
    x.q ^ 2 + x.p ^ 2
  calc
    (c * x.q - s * x.p) ^ 2 +
        (s * x.q + c * x.p) ^ 2 =
      (c ^ 2 + s ^ 2) * (x.q ^ 2 + x.p ^ 2) := by
        ring
    _ = x.q ^ 2 + x.p ^ 2 := by
      rw [h]
      ring

/--
A normalized rotation is specified by two coefficients
together with a proof of their normalization.

This structure represents an admissible transformation,
not an additional constraint on oscillator states.
-/
@[ext]
structure NormalizedRotation where
  c : ℝ
  s : ℝ
  normalized : c ^ 2 + s ^ 2 = 1

/--
Apply a normalized rotation to an oscillator state.
-/
def apply_rotation
    (r : NormalizedRotation)
    (x : OscillatorState) : OscillatorState :=
  oscillator_rotate x r.c r.s

/--
Every normalized rotation preserves the oscillator
invariant.
-/
theorem apply_rotation_energy
    (r : NormalizedRotation)
    (x : OscillatorState) :
    oscillator_energy (apply_rotation r x) =
      oscillator_energy x := by
  exact oscillator_energy_rotate x r.c r.s r.normalized

/--
Apply a finite sequence of normalized rotations.

The rotations are applied in list order.
-/
def rotate_sequence
    (x : OscillatorState)
    (rs : List NormalizedRotation) : OscillatorState :=
  rs.foldl (fun state r => apply_rotation r state) x

/--
An arbitrary finite sequence of normalized rotations
preserves the oscillator invariant.

The rotation coefficients may vary at every step.
No assumption of periodicity is required.
-/
theorem rotate_sequence_energy
    (x : OscillatorState)
    (rs : List NormalizedRotation) :
    oscillator_energy (rotate_sequence x rs) =
      oscillator_energy x := by
  unfold rotate_sequence
  induction rs generalizing x with
  | nil =>
      rfl
  | cons r rs ih =>
      simp only [List.foldl_cons]
      rw [ih]
      exact apply_rotation_energy r x


/-!
## Algebraic Structure of Normalized Rotations

We establish the identity, composition, and inverse
operations for normalized rotations.

These are the concrete algebraic properties of SO(2).
No physical interpretation is required.
-/

/--
The identity rotation leaves every oscillator state
unchanged.
-/
def identity_rotation : NormalizedRotation where
  c := 1
  s := 0
  normalized := by norm_num

/--
Composition of two normalized rotations.

`compose_rotation r₂ r₁` represents applying `r₁`
first, followed by `r₂`.
-/
def compose_rotation
    (r₂ r₁ : NormalizedRotation) :
    NormalizedRotation where
  c := r₂.c * r₁.c - r₂.s * r₁.s
  s := r₂.s * r₁.c + r₂.c * r₁.s
  normalized := by
    calc
      (r₂.c * r₁.c - r₂.s * r₁.s) ^ 2 +
          (r₂.s * r₁.c + r₂.c * r₁.s) ^ 2 =
        (r₂.c ^ 2 + r₂.s ^ 2) *
          (r₁.c ^ 2 + r₁.s ^ 2) := by ring
      _ = 1 := by
        rw [r₂.normalized, r₁.normalized]
        ring

/--
The inverse rotation reverses the orientation
of the rotational update.
-/
def inverse_rotation
    (r : NormalizedRotation) :
    NormalizedRotation where
  c := r.c
  s := -r.s
  normalized := by
    simpa using r.normalized

/--
The identity rotation acts trivially on states.
-/
theorem apply_identity_rotation
    (x : OscillatorState) :
    apply_rotation identity_rotation x = x := by
  ext
  · simp [apply_rotation, oscillator_rotate,
      identity_rotation]
  · simp [apply_rotation, oscillator_rotate,
      identity_rotation]

/--
Composition of rotations agrees with successive
application to an oscillator state.
-/
theorem apply_compose_rotation
    (r₂ r₁ : NormalizedRotation)
    (x : OscillatorState) :
    apply_rotation (compose_rotation r₂ r₁) x =
      apply_rotation r₂ (apply_rotation r₁ x) := by
  ext <;>
    simp only [apply_rotation, oscillator_rotate,
      compose_rotation] <;>
    ring

/--
Applying a normalized rotation and then its inverse
recovers the original oscillator state.
-/
theorem apply_inverse_rotation
    (r : NormalizedRotation)
    (x : OscillatorState) :
    apply_rotation (inverse_rotation r)
      (apply_rotation r x) = x := by
  ext
  · simp only [apply_rotation, oscillator_rotate,
      inverse_rotation]
    calc
      r.c * (r.c * x.q - r.s * x.p) -
          (-r.s) * (r.s * x.q + r.c * x.p) =
        (r.c ^ 2 + r.s ^ 2) * x.q := by ring
      _ = x.q := by rw [r.normalized]; ring
  · simp only [apply_rotation, oscillator_rotate,
      inverse_rotation]
    calc
      (-r.s) * (r.c * x.q - r.s * x.p) +
          r.c * (r.s * x.q + r.c * x.p) =
        (r.c ^ 2 + r.s ^ 2) * x.p := by ring
      _ = x.p := by rw [r.normalized]; ring


/--
Composition of normalized rotations is commutative.

This is a defining algebraic property of planar
rotations and distinguishes SO(2) from the generally
noncommutative rotation groups in higher dimensions.
-/
theorem compose_rotation_comm
    (r₁ r₂ : NormalizedRotation) :
    compose_rotation r₁ r₂ =
      compose_rotation r₂ r₁ := by
  ext
  · change
      r₁.c * r₂.c - r₁.s * r₂.s =
        r₂.c * r₁.c - r₂.s * r₁.s
    ring
  · change
      r₁.s * r₂.c + r₁.c * r₂.s =
        r₂.s * r₁.c + r₂.c * r₁.s
    ring

/--
Composition of normalized rotations is associative.

This establishes that grouping successive rotations
does not affect the resulting transformation.

Unlike commutativity, associativity is also satisfied
by rotations in higher dimensions.
-/
theorem compose_rotation_assoc
    (r₁ r₂ r₃ : NormalizedRotation) :
    compose_rotation (compose_rotation r₁ r₂) r₃ =
      compose_rotation r₁ (compose_rotation r₂ r₃) := by
  ext
  · change
      (r₁.c * r₂.c - r₁.s * r₂.s) * r₃.c -
        (r₁.s * r₂.c + r₁.c * r₂.s) * r₃.s =
      r₁.c * (r₂.c * r₃.c - r₂.s * r₃.s) -
        r₁.s * (r₂.s * r₃.c + r₂.c * r₃.s)
    ring
  · change
      (r₁.s * r₂.c + r₁.c * r₂.s) * r₃.c +
        (r₁.c * r₂.c - r₁.s * r₂.s) * r₃.s =
      r₁.s * (r₂.c * r₃.c - r₂.s * r₃.s) +
        r₁.c * (r₂.s * r₃.c + r₂.c * r₃.s)
    ring

/--
The identity rotation is a left identity under composition.
-/
theorem compose_rotation_identity_left
    (r : NormalizedRotation) :
    compose_rotation identity_rotation r = r := by
  ext <;>
    simp [compose_rotation, identity_rotation]

/--
The identity rotation is also a right identity.

This follows from commutativity rather than requiring
a separate algebraic calculation.
-/
theorem compose_rotation_identity_right
    (r : NormalizedRotation) :
    compose_rotation r identity_rotation = r := by
  rw [compose_rotation_comm]
  exact compose_rotation_identity_left r

/--
Composing a rotation with its inverse produces the
identity rotation.
-/
theorem compose_rotation_inverse_left
    (r : NormalizedRotation) :
    compose_rotation (inverse_rotation r) r =
      identity_rotation := by
  ext
  · change
      r.c * r.c - (-r.s) * r.s = 1
    nlinarith [r.normalized]
  · change
      (-r.s) * r.c + r.c * r.s = 0
    ring

/--
The right inverse law follows from commutativity.
-/
theorem compose_rotation_inverse_right
    (r : NormalizedRotation) :
    compose_rotation r (inverse_rotation r) =
      identity_rotation := by
  rw [compose_rotation_comm]
  exact compose_rotation_inverse_left r


/-!
## Commutative Group Structure

The normalized rotations form a commutative group
under composition.

The group operations use the definitions and proofs
already established in this module.
-/

/--
The commutative group of normalized planar rotations.

The group multiplication is rotation composition,
the identity is the trivial rotation, and the inverse
reverses the rotation.
-/
instance : CommGroup NormalizedRotation where
  mul := compose_rotation
  one := identity_rotation
  inv := inverse_rotation
  mul_assoc := compose_rotation_assoc
  one_mul := compose_rotation_identity_left
  mul_one := compose_rotation_identity_right
  inv_mul_cancel := compose_rotation_inverse_left
  mul_comm := compose_rotation_comm

/--
Multiplication of normalized rotations is defined
by their composition operation.
-/
theorem normalized_rotation_mul_def
    (r₁ r₂ : NormalizedRotation) :
    r₁ * r₂ = compose_rotation r₁ r₂ := by
  rfl

/--
Rotations commute using the general group operations.
-/
theorem normalized_rotation_mul_comm
    (r₁ r₂ : NormalizedRotation) :
    r₁ * r₂ = r₂ * r₁ := by
  exact mul_comm r₁ r₂


/-!
## Matrix Representation of SO(2)

Associate each normalized rotation with its
standard 2-by-2 real rotation matrix.
-/

/--
The matrix representation of a normalized rotation.
-/
def rotation_matrix
    (r : NormalizedRotation) :
    Matrix (Fin 2) (Fin 2) ℝ :=
  !![r.c, -r.s; r.s, r.c]

/--
Every normalized rotation is represented by a
matrix belonging to the special orthogonal group SO(2).
-/
theorem rotation_matrix_mem_so2
    (r : NormalizedRotation) :
    rotation_matrix r ∈
      Matrix.specialOrthogonalGroup (Fin 2) ℝ := by
  unfold rotation_matrix
  rw [Matrix.of_mem_specialOrthogonalGroup_fin_two_iff]
  refine ⟨rfl, rfl, ?_⟩
  simpa only [neg_sq] using r.normalized


/--
The matrix representation of normalized rotations
is injective.

Two normalized rotations with identical matrix
representations have identical coefficients.
-/
theorem rotation_matrix_injective :
    Function.Injective rotation_matrix := by
  intro r₁ r₂ h
  apply NormalizedRotation.ext
  · have h00 := congrArg
      (fun M : Matrix (Fin 2) (Fin 2) ℝ => M 0 0) h
    simpa [rotation_matrix] using h00
  · have h10 := congrArg
      (fun M : Matrix (Fin 2) (Fin 2) ℝ => M 1 0) h
    simpa [rotation_matrix] using h10

/--
Represent a normalized rotation as an element
of the matrix special orthogonal group SO(2).
-/
def rotation_to_so2
    (r : NormalizedRotation) :
    Matrix.specialOrthogonalGroup (Fin 2) ℝ :=
  ⟨rotation_matrix r, rotation_matrix_mem_so2 r⟩

/--
Construct a normalized rotation from an arbitrary
element of the two-dimensional special orthogonal group.

The rotation coefficients are recovered from the
first column of the matrix.
-/
def so2_to_rotation
    (M : Matrix.specialOrthogonalGroup (Fin 2) ℝ) :
    NormalizedRotation := by
  let A : Matrix (Fin 2) (Fin 2) ℝ := M.1
  have h := Matrix.mem_specialOrthogonalGroup_fin_two_iff.mp M.2
  refine ⟨A 0 0, A 1 0, ?_⟩
  rcases h with ⟨_, h01, hnorm⟩
  change A 0 1 = -A 1 0 at h01
  change A 0 0 ^ 2 + A 0 1 ^ 2 = 1 at hnorm
  have hs : A 0 1 ^ 2 = A 1 0 ^ 2 := by
    rw [h01]
    ring
  nlinarith

/--
Converting a normalized rotation to SO(2) and back
recovers the original rotation.
-/
theorem so2_to_rotation_rotation_to_so2
    (r : NormalizedRotation) :
    so2_to_rotation (rotation_to_so2 r) = r := by
  ext
  · rfl
  · rfl

/--
Converting an element of SO(2) to a normalized rotation
and back recovers the original matrix.

Together with the opposite conversion theorem,
this establishes a bijection between normalized
rotations and SO(2).
-/
theorem rotation_to_so2_so2_to_rotation
    (M : Matrix.specialOrthogonalGroup (Fin 2) ℝ) :
    rotation_to_so2 (so2_to_rotation M) = M := by
  apply Subtype.ext
  apply Matrix.ext
  intro i j
  have h := Matrix.mem_specialOrthogonalGroup_fin_two_iff.mp M.2
  fin_cases i <;> fin_cases j
  · rfl
  · change -(M.1 1 0) = M.1 0 1
    exact h.2.1.symm
  · rfl
  · change M.1 0 0 = M.1 1 1
    exact h.1

/--
The matrix representation preserves rotation composition.

The matrix associated with a product of normalized
rotations equals the product of their matrices.
-/
theorem rotation_to_so2_mul
    (r₁ r₂ : NormalizedRotation) :
    rotation_to_so2 (r₁ * r₂) =
      rotation_to_so2 r₁ * rotation_to_so2 r₂ := by
  rw [normalized_rotation_mul_def]
  apply Subtype.ext
  apply Matrix.ext
  intro i j
  fin_cases i <;> fin_cases j <;>
    simp [rotation_to_so2, rotation_matrix,
      compose_rotation] <;>
    ring

/--
The group of normalized rotations is isomorphic to
the two-dimensional special orthogonal group SO(2).
-/
def normalized_rotation_equiv_so2 :
    NormalizedRotation ≃*
      Matrix.specialOrthogonalGroup (Fin 2) ℝ where
  toFun := rotation_to_so2
  invFun := so2_to_rotation
  left_inv := so2_to_rotation_rotation_to_so2
  right_inv := rotation_to_so2_so2_to_rotation
  map_mul' := rotation_to_so2_mul

end StandingWave
