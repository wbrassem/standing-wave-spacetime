/-
Copyright (c) 2026 Wayne Brassem. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Wayne Brassem
-/

import StandingWave.Coupling

/-!
# Joint-Mode Amplitudes, Separability, and Conservative Transformations

This module investigates the algebraic structure of joint-mode
amplitudes represented by real two-by-two matrices.

The objective is to study separability, transformations of
joint amplitudes, and the conservation properties of those
transformations.

The development begins with a criterion for separability
and proceeds through independent transformations, conditional
phase flips, finite sequences of flips, and continuous
conditional rotations.

The module also establishes an algebraic connection to the
pairwise oscillator interactions defined in `StandingWave.Coupling`.

No physical interpretation of matrix nonseparability as
quantum entanglement is assumed.

No quantum measurement postulate, probability rule, physical
energy, or temporal evolution law is introduced.

All results concern the explicitly defined mathematical
structures and transformations.

## Foundational Principles

### Joint-Mode States

A joint-mode state is represented by a matrix

    M = [ a  b ]
        [ c  d ]

with real-valued entries.

The four entries represent abstract joint-mode amplitudes.
They do not, by definition, establish the existence of four
physical oscillators or a quantum mechanical state.

The representation permits the investigation of relationships
between amplitudes independently of their eventual physical
interpretation.

### Separability as an Algebraic Property

A joint-mode state is separable when its entries can be
expressed as products of two vectors:

    M(i,j) = u(i) * v(j).

The module proves the equivalence

    separable(M) <-> det(M) = 0.

For a two-by-two matrix, this is equivalent to

    a*d = b*c.

Nonseparability therefore describes a precise algebraic
property of the matrix.

Although this structure resembles the coefficient matrices
of bipartite quantum states, its physical identification
with quantum entanglement is not established here.

### Conservation Is Established, Not Assumed

The squared magnitude of a joint-mode state is

    Q(M) = a^2 + b^2 + c^2 + d^2.

Different transformations may preserve different quantities.

In particular, preserving total squared magnitude does not
necessarily preserve individual squared amplitudes or
separability.

No identification of quadratic magnitude with physical
energy is assumed.

Reversibility, conservation, and preservation of
separability are established separately.

## Mathematical Framework

### Independent Transformations

Transformations acting independently on the two matrix
indices are studied through independent scaling and
general linear mappings.

The module proves that independent linear transformations
preserve separability.

Invertible independent transformations therefore preserve
the separability classification in both directions.

Matrix transposition also preserves separability.

These results distinguish transformations acting
independently on the two factors from transformations
acting jointly on selected amplitudes.

### Entrywise Phase Flips

An entrywise phase flip negates one selected amplitude
while leaving the others unchanged.

Every individual flip preserves the square of every
matrix entry and consequently preserves total squared
magnitude.

The original lower-right phase flip is also shown to
be linear, involutive, and pairing-preserving.

A single entrywise flip need not preserve separability.

For a matrix

    M = [ a  b ]
        [ c  d ],

the separability condition after one flip is

    a*d = -(b*c).

For an initially separable state, a single flip creates
nonseparability exactly when

    a*d != 0.

This result holds regardless of which entry is flipped.

### Finite Sequences of Phase Flips

Finite sequences of arbitrary entrywise phase flips
preserve every individual squared amplitude.

Their effect on separability is characterized by
the parity of the sequence length.

For an even number of flips:

    separable(final state) <-> a*d = b*c.

For an odd number of flips:

    separable(final state) <-> a*d = -(b*c).

Thus even-length sequences always preserve the initial
separability classification.

Odd-length sequences may change that classification,
but do not necessarily do so.

The flip positions may repeat and are otherwise
unrestricted.

These results provide a complete algebraic
characterization of separability under finite
entrywise sign-flip sequences.

### Conditional Joint-Mode Rotation

A continuous family of conditional rotations is
introduced as a mathematical reference transformation.

The first row remains unchanged, while the second
row undergoes a planar rotation through angle theta:

    [ a  b ]       [ a                b               ]
    [ c  d ]  -->  [ c*cos - d*sin    c*sin + d*cos   ]

The module establishes that this transformation
preserves total squared magnitude for every state.

It is also reversible:

    R(-theta)(R(theta)(M)) = M.

For the reference state

    M = [ a  0 ]
        [ b  0 ],

the transformed determinant is

    det(R(theta)(M)) = a*b*sin(theta).

Consequently, continuous reversible transformations
can produce nonseparability from initially separable
states without requiring instantaneous sign inversion.

This is a mathematical construction.

The existence of a corresponding physical interaction
has not been established.

### Relationship to Pairwise Oscillator Coupling

Two primitive oscillator states can be represented
as the rows of a joint-mode coordinate matrix.

The normalized mixing operation defined in
`StandingWave.Coupling` preserves the determinant
of this matrix.

Thus the particular normalized pairwise mixing
operation preserves its separability classification.

This establishes a constraint on that concrete
coupling operation.

It does not prove that all conservative oscillator
interactions preserve separability.

Nor does the coordinate identification establish
a physical tensor-product interpretation.

## Principal Results

* `joint_mode_separable_iff_det_eq_zero`:
  Exact determinant criterion for separability.

* `joint_mode_independent_linear_preserves_separable`:
  Preservation of separability under independent
  linear transformations.

* `joint_mode_transpose_separable_iff`:
  Invariance of separability under transposition.

* `joint_mode_phase_flip_magnitude_sq`:
  Conservation of total squared magnitude under
  the original phase flip.

* `joint_mode_phase_flip_equiv`:
  Invertible linear structure of the original flip.

* `joint_mode_double_flip_separable_iff`:
  Preservation of separability classification under
  any two entrywise flips.

* `joint_mode_flip_at_of_separable_not_separable_iff`:
  Exact criterion for creation of nonseparability
  by an individual flip.

* `joint_mode_flip_sequence_entry_sq`:
  Preservation of individual squared amplitudes
  under arbitrary finite flip sequences.

* `joint_mode_flip_sequence_even_separable_iff`:
  Separability classification for even-length
  flip sequences.

* `joint_mode_flip_sequence_odd_separable_iff`:
  Separability classification for odd-length
  flip sequences.

* `joint_mode_conditional_rotate_seed_det`:
  Determinant evolution under conditional rotation
  of the reference state.

* `joint_mode_conditional_rotate_magnitude_sq`:
  Conservation of total squared magnitude under
  conditional rotation.

* `joint_mode_conditional_rotate_inverse`:
  Reversibility of conditional rotation.

* `joint_mode_mix_oscillators_det`:
  Determinant preservation under the normalized
  pairwise mixing operation from Coupling.

## Development Structure

The module follows four related investigations.

1. Joint-state algebra

    JointModeState
      |
      +-- Quadratic magnitude
      |
      +-- Separability
            |
            +-- Determinant criterion

2. Independent and entrywise transformations

    Independent transformations
      |
      +-- Separability preservation

    Entrywise phase flips
      |
      +-- Magnitude preservation
      |
      +-- Reversibility
      |
      +-- Finite sequences
            |
            +-- Even/odd separability criteria

3. Conditional rotations

    Conditional joint-mode rotation
      |
      +-- Reference-state determinant
      |
      +-- Magnitude preservation
      |
      +-- Inverse transformation

4. Relationship to primitive coupling

    OscillatorState pairs
      |
      +-- Joint-mode coordinate matrix
            |
            +-- Determinant preservation
                under normalized mixing

These diagrams summarize the mathematical organization
rather than the complete proof dependency graph.

## Scope and Interpretation

This module establishes algebraic properties of real
joint-mode amplitude matrices.

Its results demonstrate that reversibility and
conservation of total squared magnitude do not
necessarily imply preservation of separability.

The module does not establish that the modeled
nonseparability is physical quantum entanglement.

No complex-valued quantum state space, Born probability
rule, measurement projection, or collapse mechanism
is derived.

The parameter of a conditional rotation is a real angle.
Its use does not itself introduce physical time or an
equation of motion.

The algebraic relationship to `Coupling.lean` does not
establish that conditional rotation can arise from the
specific conservative mixing operation defined there.

Likewise, the module introduces no physical geometry,
spacetime metric, propagation law, or gravitational
interaction.

The constructions provide mathematical reference
structures against which subsequent dynamical models
may be tested.

Their physical significance remains an open question.
-/

namespace StandingWave

/-!
## Joint Two-Mode States

We begin with two oscillators, each admitting two
abstract modes.

A joint state assigns a real amplitude to each
of the four possible combinations of modes.

The mode labels are purely mathematical.
They do not yet represent physical frequencies,
spatial dimensions, or temporal evolution.

This representation introduces joint amplitudes
without assuming entanglement, normalization,
or any particular interaction law.
-/

/--
A joint amplitude state for two oscillators,
each possessing two abstract modes.

The matrix entry `M i j` gives the real amplitude
associated with mode `i` of the first oscillator
and mode `j` of the second oscillator.

Real amplitudes suffice for the initial experiment.
Complex amplitudes may be introduced later to
represent general phase relationships.
-/
abbrev JointModeState : Type :=
  Matrix (Fin 2) (Fin 2) ℝ


/--
The total squared magnitude of a joint two-mode state.

This is the sum of the squared real amplitudes
over all four joint mode combinations.

The definition introduces no normalization
requirement or conservation assumption.
-/
def joint_mode_magnitude_sq (M : JointModeState) : ℝ :=
  ∑ i : Fin 2, ∑ j : Fin 2, (M i j) ^ 2

/--
The total squared magnitude of a joint two-mode
state is nonnegative.

This follows from the nonnegativity of each
squared real amplitude and requires no additional
assumption about the joint state.
-/
theorem joint_mode_magnitude_sq_nonneg
    (M : JointModeState) :
    0 ≤ joint_mode_magnitude_sq M := by
  unfold joint_mode_magnitude_sq
  apply Finset.sum_nonneg
  intro i hi
  apply Finset.sum_nonneg
  intro j hj
  exact sq_nonneg (M i j)


/-!
## Separability of Joint States

We distinguish joint amplitude states that factor
into independent mode-amplitude vectors from
those that cannot be so factored.

Separability is defined by the existence of an
outer-product representation.

This definition does not assume normalization,
a measurement process, or a physical interpretation
of nonseparability as quantum entanglement.
-/

/--
A joint two-mode state is separable if its amplitudes
factor into a product of two independent mode vectors.

That is, there exist vectors `u` and `v` such that
`M i j = u i * v j` for every pair of mode indices.

This is a mathematical factorization property,
not an additional dynamical assumption.
-/
def joint_mode_separable (M : JointModeState) : Prop :=
  ∃ u v : Fin 2 → ℝ,
    ∀ i j, M i j = u i * v j

/--
A separable joint two-mode state has zero determinant.

If the joint amplitudes factor as `M i j = u i * v j`,
then the two products appearing in the determinant
are equal.

Consequently, the determinant vanishes.

This establishes a necessary algebraic condition
for separability without assuming normalization
or any particular interaction law.
-/
theorem joint_mode_separable_det_eq_zero
    (M : JointModeState)
    (hM : joint_mode_separable M) :
    Matrix.det M = 0 := by
  rcases hM with ⟨u, v, h⟩
  rw [Matrix.det_fin_two]
  rw [h 0 0, h 1 1, h 0 1, h 1 0]
  ring

/--
A joint two-mode state with zero determinant is separable.

The proof constructs an outer-product factorization
by considering whether the first row contains a
nonzero entry.

If the upper-left entry is nonzero, it provides
a pivot for the factorization.

If that entry vanishes but the upper-right entry
is nonzero, the determinant condition forces
the lower-left entry to vanish.

If both entries of the first row vanish, the
matrix factors directly through its second row.

The result requires neither normalization nor
any assumption about interactions.
-/
theorem joint_mode_det_eq_zero_implies_separable
    (M : JointModeState)
    (hdet : Matrix.det M = 0) :
    joint_mode_separable M := by
  have hcross :
      M 0 0 * M 1 1 = M 0 1 * M 1 0 := by
    rw [Matrix.det_fin_two] at hdet
    exact sub_eq_zero.mp hdet
  by_cases h00 : M 0 0 = 0
  · by_cases h01 : M 0 1 = 0
    · -- Both entries in the first row vanish.
      refine ⟨(fun i => if i = 0 then 0 else 1),
              (fun j => M 1 j), ?_⟩
      intro i j
      fin_cases i <;> fin_cases j <;>
        simp [h00, h01]
    · -- Use the nonzero upper-right entry as a pivot.
      have h10 : M 1 0 = 0 := by
        have hz : M 0 1 * M 1 0 = 0 := by
          simpa [h00] using hcross.symm
        exact (mul_eq_zero.mp hz).resolve_left h01
      refine ⟨(fun i => M i 1),
              (fun j => M 0 j / M 0 1), ?_⟩
      intro i j
      fin_cases i <;> fin_cases j <;>
        simp [h00, h10, h01]
  · -- Use the nonzero upper-left entry as a pivot.
    have h11 :
        M 1 1 = M 1 0 * (M 0 1 / M 0 0) := by
      calc
        M 1 1 = (M 0 1 * M 1 0) / M 0 0 := by
          apply (eq_div_iff h00).2
          calc
            M 1 1 * M 0 0 = M 0 0 * M 1 1 := by ring
            _ = M 0 1 * M 1 0 := hcross
        _ = M 1 0 * (M 0 1 / M 0 0) := by ring
    refine ⟨(fun i => M i 0),
            (fun j => M 0 j / M 0 0), ?_⟩
    intro i j
    fin_cases i <;> fin_cases j
    · simp [h00]
    · change M 0 1 = M 0 0 * (M 0 1 / M 0 0)
      field_simp [h00]
    · simp [h00]
    · simpa using h11


/--
A joint two-mode state is separable if and only if
its determinant vanishes.

This gives an exact algebraic characterization of
separability for real two-mode joint amplitude states.

The equivalence requires no normalization,
conservation law, or interaction assumption.
-/
theorem joint_mode_separable_iff_det_eq_zero
    (M : JointModeState) :
    joint_mode_separable M ↔ Matrix.det M = 0 := by
  constructor
  · intro hM
    exact joint_mode_separable_det_eq_zero M hM
  · intro hdet
    exact joint_mode_det_eq_zero_implies_separable M hdet


/-!
## Magnitude-Preserving Joint Transformations

We investigate transformations of joint mode states
that preserve individual squared amplitudes while
potentially changing their factorization properties.

Our first candidate negates the amplitude associated
with the joint mode pair `(1, 1)` while leaving
all other amplitudes unchanged.

This is a mathematically specified transformation.
We do not yet assume that it follows from the
existing oscillator interaction premises.
-/

/--
Negate the amplitude of joint mode `(1, 1)`,
leaving the other three amplitudes unchanged.

This transformation changes a relative sign
without changing any amplitude's squared magnitude.

It provides a candidate for investigating whether
magnitude preservation permits changes in separability.
-/
def joint_mode_phase_flip (M : JointModeState) :
    JointModeState :=
  fun i j =>
    if i = 1 ∧ j = 1 then -(M i j) else M i j

/--
The joint-mode phase flip preserves the squared
magnitude of every individual joint-mode amplitude.

This holds independently of normalization and
requires no assumption about the initial state.
-/
theorem joint_mode_phase_flip_entry_sq
    (M : JointModeState) (i j : Fin 2) :
    (joint_mode_phase_flip M i j) ^ 2 = (M i j) ^ 2 := by
  by_cases h : i = 1 ∧ j = 1
  · simp [joint_mode_phase_flip, h]
  · simp [joint_mode_phase_flip, h]

/--
The joint-mode phase flip preserves total squared
magnitude.

Since the squared magnitude of every joint-mode
amplitude is individually preserved, their sum
is also preserved.

This result follows from entrywise conservation
and requires no normalization assumption.

It does not establish conservation of physical
energy or independently defined oscillator energies.
-/
theorem joint_mode_phase_flip_magnitude_sq
    (M : JointModeState) :
    joint_mode_magnitude_sq (joint_mode_phase_flip M) =
      joint_mode_magnitude_sq M := by
  unfold joint_mode_magnitude_sq
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro j hj
  exact joint_mode_phase_flip_entry_sq M i j

/-!
## A Magnitude-Preserving Change of Separability

We construct an explicit normalized joint two-mode
state that is initially separable.

We will then apply the previously defined phase
flip and investigate whether separability changes
despite preservation of all squared amplitudes.

This experiment tests the mathematical consequences
of our definitions without introducing a new
physical interaction axiom.
-/

/--
The uniform joint two-mode state.

All four joint-mode amplitudes are equal to one half.

This state provides the initial condition for our
magnitude-preserving phase-flip experiment.
-/
noncomputable def joint_mode_uniform : JointModeState :=
  fun _ _ => (1 / 2 : ℝ)

/--
The uniform joint two-mode state is separable.

Its amplitudes factor into a constant vector
with entries one half and a constant vector
with entries one.
-/
theorem joint_mode_uniform_separable :
    joint_mode_separable joint_mode_uniform := by
  refine ⟨(fun _ => (1 / 2 : ℝ)),
          (fun _ => (1 : ℝ)), ?_⟩
  intro i j
  simp [joint_mode_uniform]

/--
Applying the joint-mode phase flip to the uniform
state produces a matrix with determinant negative
one half.

In particular, the determinant is nonzero even
though the transformation preserves every squared
joint-mode amplitude.

This establishes the algebraic fact needed to
demonstrate a change in separability.
-/
theorem joint_mode_phase_flip_uniform_det :
    Matrix.det (joint_mode_phase_flip joint_mode_uniform) =
      -(1 / 2 : ℝ) := by
  rw [Matrix.det_fin_two]
  norm_num [joint_mode_phase_flip, joint_mode_uniform]

/--
The phase flip of the uniform joint two-mode state
is nonseparable.

If the transformed state were separable, its
determinant would vanish. But its determinant
has already been proved equal to negative one half.

Thus the phase flip changes the separability of
the uniform state while preserving every squared
joint-mode amplitude.
-/
theorem joint_mode_phase_flip_uniform_not_separable :
    ¬ joint_mode_separable
        (joint_mode_phase_flip joint_mode_uniform) := by
  intro hsep
  have hzero :
      Matrix.det (joint_mode_phase_flip joint_mode_uniform) = 0 :=
    joint_mode_separable_det_eq_zero
      (joint_mode_phase_flip joint_mode_uniform) hsep
  rw [joint_mode_phase_flip_uniform_det] at hzero
  norm_num at hzero

/--
The joint-mode phase flip provides an explicit
example of a magnitude-preserving transformation
that changes separability.

The uniform initial state is separable, whereas
its phase-flipped state is nonseparable.

Nevertheless, every individual squared joint-mode
amplitude is preserved, as is their total squared
magnitude.

This is an algebraic existence result. It does
not establish that the transformation follows
from the oscillator interaction dynamics.
-/
theorem joint_mode_phase_flip_changes_separability :
    joint_mode_separable joint_mode_uniform ∧
    ¬ joint_mode_separable
        (joint_mode_phase_flip joint_mode_uniform) ∧
    (∀ i j : Fin 2,
      (joint_mode_phase_flip joint_mode_uniform i j) ^ 2 =
        (joint_mode_uniform i j) ^ 2) ∧
    joint_mode_magnitude_sq
        (joint_mode_phase_flip joint_mode_uniform) =
      joint_mode_magnitude_sq joint_mode_uniform := by
  refine ⟨joint_mode_uniform_separable,
          joint_mode_phase_flip_uniform_not_separable,
          ?_, ?_⟩
  · intro i j
    exact joint_mode_phase_flip_entry_sq
      joint_mode_uniform i j
  · exact joint_mode_phase_flip_magnitude_sq
      joint_mode_uniform

/-!
## Independent Mode Transformations

We first investigate transformations that act
independently on the two mode indices.

An independent rescaling multiplies each joint
amplitude by one factor associated with its first
index and another associated with its second index.

Such transformations preserve separability.
They need not preserve squared magnitude unless
additional conditions are imposed on the factors.
-/

/--
Independently rescale the two indices of a joint
mode state.

The functions `s` and `t` act on separate mode
indices, so the transformation introduces no
joint conditional dependence between them.
-/
noncomputable def joint_mode_independent_scale
    (s t : Fin 2 → ℝ) (M : JointModeState) :
    JointModeState :=
  fun i j => s i * M i j * t j

/--
Independent rescaling of the two mode indices
preserves separability.

If the original amplitudes factor as `u i * v j`,
then the transformed amplitudes factor as
`(s i * u i) * (v j * t j)`.

No normalization or magnitude-conservation
assumption is needed.
-/
theorem joint_mode_independent_scale_preserves_separable
    (M : JointModeState)
    (s t : Fin 2 → ℝ)
    (hM : joint_mode_separable M) :
    joint_mode_separable
      (joint_mode_independent_scale s t M) := by
  rcases hM with ⟨u, v, huv⟩
  refine ⟨(fun i => s i * u i),
          (fun j => v j * t j), ?_⟩
  intro i j
  change s i * M i j * t j =
    (s i * u i) * (v j * t j)
  rw [huv i j]
  ring

/--
The phase flip of the uniform joint-mode state
cannot be represented by independent scaling
of its two mode indices.

Any independent scaling of a separable state
remains separable.

Since the phase-flipped uniform state is
nonseparable, no pair of independent scaling
functions can produce it from the uniform state.

This result does not assume magnitude conservation
for the independent scaling functions.
-/
theorem joint_mode_phase_flip_uniform_not_independent_scale :
    ¬ ∃ s t : Fin 2 → ℝ,
      joint_mode_independent_scale s t joint_mode_uniform =
        joint_mode_phase_flip joint_mode_uniform := by
  rintro ⟨s, t, h⟩
  have hsep :
      joint_mode_separable
        (joint_mode_independent_scale s t joint_mode_uniform) :=
    joint_mode_independent_scale_preserves_separable
      joint_mode_uniform s t joint_mode_uniform_separable
  rw [h] at hsep
  exact joint_mode_phase_flip_uniform_not_separable hsep


/-!
## Independent Linear Mode Transformations

We extend independent mode scaling to arbitrary
linear transformations of the two mode spaces.

For a joint amplitude matrix M, independent
linear transformations A and B act as

    M' = A * M * B.transpose

Such transformations can mix modes within each
factor but cannot create nonseparability from
an initially separable state.

The transformations need not be invertible or
magnitude-preserving.
-/

/--
Apply independent linear transformations to the
two factors of a joint two-mode state.

The matrix A acts on the first mode index,
and B acts on the second mode index.
-/
noncomputable def joint_mode_independent_linear
    (A B M : JointModeState) : JointModeState :=
  A * M * B.transpose

/--
Independent linear transformations preserve
separability.

The proof uses the determinant characterization
of separability and multiplicativity of the
determinant.

Since det(M) = 0 for an initially separable state,
the determinant of A * M * B.transpose also
vanishes, regardless of A and B.
-/
theorem joint_mode_independent_linear_preserves_separable
    (A B M : JointModeState)
    (hM : joint_mode_separable M) :
    joint_mode_separable
      (joint_mode_independent_linear A B M) := by
  apply (joint_mode_separable_iff_det_eq_zero _).2
  have hdet : Matrix.det M = 0 :=
    (joint_mode_separable_iff_det_eq_zero M).1 hM
  simp [joint_mode_independent_linear, Matrix.det_mul, hdet]

/--
The phase flip of the uniform joint-mode state
cannot be produced by independent linear
transformations of the two mode spaces.

Any transformation of the form A * M * B.transpose
preserves separability when M is separable.

Since the phase-flipped uniform state is
nonseparable, no matrices A and B can reproduce
that result by independent linear action.

This strengthens the earlier impossibility
result for independent scaling.
-/
theorem joint_mode_phase_flip_uniform_not_independent_linear :
    ¬ ∃ A B : JointModeState,
      joint_mode_independent_linear A B joint_mode_uniform =
        joint_mode_phase_flip joint_mode_uniform := by
  rintro ⟨A, B, h⟩
  have hsep :
      joint_mode_separable
        (joint_mode_independent_linear A B joint_mode_uniform) :=
    joint_mode_independent_linear_preserves_separable
      A B joint_mode_uniform joint_mode_uniform_separable
  rw [h] at hsep
  exact joint_mode_phase_flip_uniform_not_separable hsep

/--
The joint-mode phase flip is an involution.

Applying the transformation twice restores
every joint-mode amplitude exactly.

Consequently, the transformation is reversible,
with itself as its inverse.

Together with squared-magnitude conservation,
this establishes that the phase flip can change
separability without losing amplitude information.
-/
theorem joint_mode_phase_flip_involutive
    (M : JointModeState) :
    joint_mode_phase_flip (joint_mode_phase_flip M) = M := by
  funext i j
  by_cases h : i = 1 ∧ j = 1
  · simp [joint_mode_phase_flip, h]
  · simp [joint_mode_phase_flip, h]

/--
The joint-mode phase flip is additive.

Applying the phase flip to the sum of two joint
states gives the same result as applying it to
each state separately and then adding.

This establishes the additive part of linearity
for the phase flip on the full joint-mode space.
-/
theorem joint_mode_phase_flip_add
    (M N : JointModeState) :
    joint_mode_phase_flip (M + N) =
      joint_mode_phase_flip M + joint_mode_phase_flip N := by
  funext i j
  by_cases h : i = 1 ∧ j = 1
  · simp [joint_mode_phase_flip, h, neg_add_rev, add_comm]
  · simp [joint_mode_phase_flip, h]

/--
The joint-mode phase flip commutes with real
scalar multiplication.

Scaling a joint state before applying the phase
flip produces the same result as applying the
phase flip first and then scaling.

Together with additivity, this establishes
that the phase flip is linear on the full
four-dimensional real joint-mode space.
-/
theorem joint_mode_phase_flip_smul
    (r : ℝ) (M : JointModeState) :
    joint_mode_phase_flip (r • M) =
      r • joint_mode_phase_flip M := by
  funext i j
  by_cases h : i = 1 ∧ j = 1
  · simp [joint_mode_phase_flip, h, smul_eq_mul, mul_neg]
  · simp [joint_mode_phase_flip, h]

/--
The joint-mode phase flip as a real linear map.

The additive and scalar-multiplication laws
have already been established.

This packages the phase flip as a linear operator
on the full joint two-mode amplitude space.

The operator acts jointly on the two mode indices.
It is not assumed to factor into independent
linear operations on the component mode spaces.
-/
def joint_mode_phase_flip_linear :
    JointModeState →ₗ[ℝ] JointModeState where
  toFun := joint_mode_phase_flip
  map_add' := joint_mode_phase_flip_add
  map_smul' := joint_mode_phase_flip_smul

/--
The joint-mode phase flip as a linear equivalence.

The transformation is linear and involutive:
applying it twice restores the original state.

Consequently, it is an invertible linear map
whose inverse is itself.

This packages the previously established
linearity and reversibility into a single
mathematical structure.

The equivalence does not imply preservation
of separability.
-/
def joint_mode_phase_flip_equiv :
    JointModeState ≃ₗ[ℝ] JointModeState where
  toFun := joint_mode_phase_flip
  invFun := joint_mode_phase_flip
  left_inv := joint_mode_phase_flip_involutive
  right_inv := joint_mode_phase_flip_involutive
  map_add' := joint_mode_phase_flip_add
  map_smul' := joint_mode_phase_flip_smul

/-!
## Inner Product Preservation

We investigate whether the joint-mode phase flip
preserves the Euclidean geometry of the full
four-dimensional joint amplitude space.

The natural real inner product is the sum of
products of corresponding joint-mode amplitudes.

Preservation of this inner product establishes
a stronger geometric property than conservation
of the squared magnitude of a single state.
-/

/--
The Euclidean inner product of two joint-mode
states, obtained by summing the products of
corresponding amplitudes.
-/
noncomputable def joint_mode_pairing
    (M N : JointModeState) : ℝ :=
  ∑ i : Fin 2, ∑ j : Fin 2, M i j * N i j

/--
The joint-mode phase flip preserves the
Euclidean inner product of any two joint states.

The distinguished amplitude changes sign
in both states, so its contribution to the
inner product remains unchanged.

All other contributions are unchanged directly.

Thus the phase flip preserves the full inner
product, not merely individual squared magnitudes.
-/
theorem joint_mode_phase_flip_pairing
    (M N : JointModeState) :
    joint_mode_pairing
        (joint_mode_phase_flip M)
        (joint_mode_phase_flip N) =
      joint_mode_pairing M N := by
  unfold joint_mode_pairing
  apply Finset.sum_congr rfl
  intro i hi
  apply Finset.sum_congr rfl
  intro j hj
  by_cases h : i = 1 ∧ j = 1
  · simp [joint_mode_phase_flip, h]
  · simp [joint_mode_phase_flip, h]

/-!
## Exchange of Joint Mode Factors

We investigate an operation on the combined
joint-mode state that exchanges its two indices.

Unlike independent transformations, this operation
exchanges the roles of the two mode spaces.

Nevertheless, it preserves separability because
the transpose of an outer product is another
outer product.

This provides a useful contrast with the
conditional phase flip, which can change
separability.
-/

/--
Transposing a separable joint-mode state
produces another separable state.

If the original amplitudes factor as
`M i j = u i * v j`, then the transposed
amplitudes factor as `v i * u j`.

Thus exchanging the two mode factors does
not create nonseparability.
-/
theorem joint_mode_transpose_preserves_separable
    (M : JointModeState)
    (hM : joint_mode_separable M) :
    joint_mode_separable M.transpose := by
  rcases hM with ⟨u, v, huv⟩
  refine ⟨v, u, ?_⟩
  intro i j
  change M j i = v i * u j
  rw [huv j i]
  ring

/--
Transposition preserves separability in both
directions.

A separable joint state remains separable
when its two mode factors are exchanged.

Conversely, if the transpose is separable,
transposing once more recovers the original
state, which must therefore also be separable.

Thus transposition can neither create nor
destroy nonseparability.
-/
theorem joint_mode_transpose_separable_iff
    (M : JointModeState) :
    joint_mode_separable M.transpose ↔
      joint_mode_separable M := by
  constructor
  · intro hT
    have hTT :=
      joint_mode_transpose_preserves_separable
        M.transpose hT
    simpa only [Matrix.transpose_transpose] using hTT
  · intro hM
    exact joint_mode_transpose_preserves_separable M hM

/--
Transposition preserves nonseparability in both
directions.

This follows immediately from the established
equivalence of separability under transposition.

Thus exchanging the two joint-mode factors
cannot create or destroy nonseparability.
-/
theorem joint_mode_transpose_nonseparable_iff
    (M : JointModeState) :
    ¬ joint_mode_separable M.transpose ↔
      ¬ joint_mode_separable M := by
  exact not_congr (joint_mode_transpose_separable_iff M)

/--
The determinant of a joint-mode state changes
under the conditional phase flip by twice the
negative product of its diagonal amplitudes.

Explicitly,

    det(phaseFlip M) = det(M) - 2 * M 0 0 * M 1 1.

This identity makes the relationship between
the joint phase structure and separability
explicit.

No assumption of separability or normalization
is needed.
-/
theorem joint_mode_phase_flip_det
    (M : JointModeState) :
    Matrix.det (joint_mode_phase_flip M) =
      Matrix.det M - 2 * (M 0 0 * M 1 1) := by
    simp [Matrix.det_fin_two, joint_mode_phase_flip]
    ring

/-!
## General Entrywise Phase Flips

We generalize the conditional phase flip so
that any of the four joint-mode amplitudes
may be negated.

The next goal is to characterize the effect
of two successive phase flips on the determinant.

The two flips need not target the same entry.
-/

/--
Negate the joint-mode amplitude at the specified
pair of indices, leaving the other amplitudes
unchanged.

The index pair `p` identifies one of the four
entries of the joint two-mode state.
-/
def joint_mode_flip_at
    (p : Fin 2 × Fin 2)
    (M : JointModeState) : JointModeState :=
  fun i j =>
    if (i, j) = p then -(M i j) else M i j

/--
Two arbitrary entrywise phase flips change the
determinant by at most a sign.

For any joint-mode state M and any two entry
positions p and q, the determinant after both
flips equals either det(M) or -det(M).

The positions may coincide or differ.

This holds without assuming separability,
normalization, or nonzero amplitudes.
-/
theorem joint_mode_double_flip_det_eq_or_neg
    (M : JointModeState)
    (p q : Fin 2 × Fin 2) :
    Matrix.det
        (joint_mode_flip_at q (joint_mode_flip_at p M)) =
          Matrix.det M ∨
    Matrix.det
        (joint_mode_flip_at q (joint_mode_flip_at p M)) =
          -Matrix.det M := by
  rcases p with ⟨pi, pj⟩
  rcases q with ⟨qi, qj⟩
  fin_cases pi <;> fin_cases pj <;>
    fin_cases qi <;> fin_cases qj <;>
    solve
      | left
        simp [joint_mode_flip_at, Matrix.det_fin_two]
        all_goals ring
      | right
        simp [joint_mode_flip_at, Matrix.det_fin_two]
        all_goals ring

/--
Two arbitrary entrywise phase flips preserve
the separability classification of a joint state.

The two flips may target the same entry or
different entries.

The determinant after both flips equals either
the original determinant or its negative.
Consequently, its zero/nonzero status is
unchanged.

Thus two phase flips can neither create nor
destroy nonseparability, even when the final
state differs from the initial state.

No normalization or nonzero-amplitude
assumptions are required.
-/
theorem joint_mode_double_flip_separable_iff
    (M : JointModeState)
    (p q : Fin 2 × Fin 2) :
    joint_mode_separable
        (joint_mode_flip_at q (joint_mode_flip_at p M)) ↔
      joint_mode_separable M := by
  rcases joint_mode_double_flip_det_eq_or_neg M p q with
    hsame | hneg
  · simp only [joint_mode_separable_iff_det_eq_zero, hsame]
  · simp only [joint_mode_separable_iff_det_eq_zero,
      hneg, neg_eq_zero]

/--
A single phase flip at the lower-right entry
produces a separable state exactly when the
two determinant products are negatives of
one another.

This condition does not assume that the
initial state is separable.

Consequently, a single phase flip need not
change the separability classification.
-/
theorem joint_mode_phase_flip_separable_iff
    (M : JointModeState) :
    joint_mode_separable (joint_mode_phase_flip M) ↔
      M 0 0 * M 1 1 = -(M 0 1 * M 1 0) := by
  rw [joint_mode_separable_iff_det_eq_zero]
  rw [joint_mode_phase_flip_det, Matrix.det_fin_two]
  constructor
  · intro h
    nlinarith
  · intro h
    nlinarith

/--
For an initially separable joint-mode state,
the conditional phase flip creates nonseparability
if and only if the product of the two diagonal
amplitudes is nonzero.

The initial separability hypothesis is essential.

Without it, a phase flip need not change the
separability classification.

This result characterizes exactly when our
original phase flip destroys separability.
-/
theorem joint_mode_phase_flip_of_separable_not_separable_iff
    (M : JointModeState)
    (hM : joint_mode_separable M) :
    ¬ joint_mode_separable (joint_mode_phase_flip M) ↔
      M 0 0 * M 1 1 ≠ 0 := by
  have hcross :
      M 0 0 * M 1 1 = M 0 1 * M 1 0 := by
    have hdet : Matrix.det M = 0 :=
      (joint_mode_separable_iff_det_eq_zero M).mp hM
    rw [Matrix.det_fin_two] at hdet
    exact sub_eq_zero.mp hdet
  rw [joint_mode_phase_flip_separable_iff, ← hcross]
  constructor
  · intro hnot hzero
    apply hnot
    simp [hzero]
  · intro hnonzero hneg
    apply hnonzero
    linarith

/--
Flipping any single joint-mode amplitude produces
the same separability classification as flipping
the lower-right amplitude.

The resulting matrices need not be equal.

The proof applies the established double-flip
separability theorem to the phase-flipped state,
using the fact that flipping the lower-right
entry again restores the original state.

No initial separability assumption is required.
-/
theorem joint_mode_flip_at_separable_iff_phase_flip
    (M : JointModeState)
    (p : Fin 2 × Fin 2) :
    joint_mode_separable (joint_mode_flip_at p M) ↔
      joint_mode_separable (joint_mode_phase_flip M) := by
  have hcancel :
      joint_mode_flip_at (1, 1) (joint_mode_phase_flip M) = M := by
    funext i j
    by_cases h : (i, j) = ((1 : Fin 2), (1 : Fin 2))
    · have hi : i = 1 := congrArg Prod.fst h
      have hj : j = 1 := congrArg Prod.snd h
      subst i
      subst j
      simp [joint_mode_flip_at, joint_mode_phase_flip]
    · have h' : ¬ (i = 1 ∧ j = 1) := by
        intro hij
        rcases hij with ⟨hi, hj⟩
        subst i
        subst j
        exact h rfl
      simp [joint_mode_flip_at, joint_mode_phase_flip, h, h']
  have hdouble :=
    joint_mode_double_flip_separable_iff
      (joint_mode_phase_flip M) (1, 1) p
  simpa only [hcancel] using hdouble

/--
For an initially separable joint-mode state,
flipping any one of its four amplitudes creates
nonseparability if and only if the product of
the two diagonal amplitudes is nonzero.

The result is independent of which amplitude
is selected.

The proof combines the established single-flip
criterion with the equivalence of separability
classifications for all individual entry flips.

No normalization assumption is required.
-/
theorem joint_mode_flip_at_of_separable_not_separable_iff
    (M : JointModeState)
    (p : Fin 2 × Fin 2)
    (hM : joint_mode_separable M) :
    ¬ joint_mode_separable (joint_mode_flip_at p M) ↔
      M 0 0 * M 1 1 ≠ 0 := by
  rw [joint_mode_flip_at_separable_iff_phase_flip M p]
  exact joint_mode_phase_flip_of_separable_not_separable_iff M hM

/-!
## Sequences of Paired Phase Flips

An even sequence of entrywise phase flips can
be grouped into consecutive pairs.

Each pair preserves the separability
classification, regardless of the entries
selected or the state on which it acts.

We extend the double-flip result to arbitrarily
long lists of such pairs.
-/

/--
Apply a sequence of pairs of entrywise phase flips.

Each pair contains two positions `p` and `q`.
The operation first flips `p`, then `q`, before
continuing with the remaining pairs.

The sequence may be empty, and positions may
repeat within or between pairs.
-/
def joint_mode_flip_pairs :
    List ((Fin 2 × Fin 2) × (Fin 2 × Fin 2)) →
      JointModeState → JointModeState
  | [], M => M
  | (p, q) :: rest, M =>
      joint_mode_flip_pairs rest
        (joint_mode_flip_at q (joint_mode_flip_at p M))

/--
Any finite sequence of paired phase flips
preserves the separability classification.

The proof proceeds by induction on the list
of flip pairs.

The empty sequence changes nothing.

For a nonempty sequence, the first two flips
preserve separability by the established
double-flip theorem. The induction hypothesis
then applies to the remaining pairs.

No initial separability assumption is required.
-/
theorem joint_mode_flip_pairs_separable_iff
    (pairs : List ((Fin 2 × Fin 2) × (Fin 2 × Fin 2)))
    (M : JointModeState) :
    joint_mode_separable (joint_mode_flip_pairs pairs M) ↔
      joint_mode_separable M := by
  induction pairs generalizing M with
  | nil =>
      rfl
  | cons pq rest ih =>
      rcases pq with ⟨p, q⟩
      change joint_mode_separable
        (joint_mode_flip_pairs rest
          (joint_mode_flip_at q (joint_mode_flip_at p M))) ↔
        joint_mode_separable M
      exact
        (ih (joint_mode_flip_at q (joint_mode_flip_at p M))).trans
          (joint_mode_double_flip_separable_iff M p q)

/--
Flipping any selected joint-mode amplitude
preserves the squared magnitude of every entry.

The result holds for all choices of the flip
position and all joint-mode states.

At the selected position, negation leaves
the square unchanged. Every other entry
is unchanged directly.

No normalization or separability assumption
is required.
-/
theorem joint_mode_flip_at_entry_sq
    (M : JointModeState)
    (p : Fin 2 × Fin 2)
    (i j : Fin 2) :
    (joint_mode_flip_at p M i j) ^ 2 =
      (M i j) ^ 2 := by
  by_cases h : (i, j) = p
  · simp [joint_mode_flip_at, h]
  · simp [joint_mode_flip_at, h]

/-!
## Arbitrary Sequences of Entrywise Phase Flips

We extend the single-entry phase flip to finite
sequences of arbitrary length.

Unlike a list of flip pairs, this representation
allows both even and odd numbers of operations.

We first establish preservation of the squared
magnitude of every joint-mode amplitude.

Subsequent results will characterize separability
using the parity of the sequence length.
-/

/--
Apply a finite sequence of entrywise phase flips.

The positions are processed in list order.
Each position identifies one joint-mode amplitude
to negate.

The empty sequence leaves the state unchanged.
Positions may appear multiple times.
-/
def joint_mode_flip_sequence :
    List (Fin 2 × Fin 2) →
      JointModeState → JointModeState
  | [], M => M
  | p :: rest, M =>
      joint_mode_flip_sequence rest
        (joint_mode_flip_at p M)

/--
Any finite sequence of entrywise phase flips
preserves every individual squared amplitude.

The proof proceeds by induction on the list
of selected positions.

The empty sequence changes nothing.

For a nonempty sequence, the first flip preserves
each squared amplitude, and the induction
hypothesis applies to the remaining flips.

This holds for arbitrary sequence length,
including both even and odd lengths.

No separability or normalization assumption
is required.
-/
theorem joint_mode_flip_sequence_entry_sq
    (positions : List (Fin 2 × Fin 2))
    (M : JointModeState)
    (i j : Fin 2) :
    (joint_mode_flip_sequence positions M i j) ^ 2 =
      (M i j) ^ 2 := by
  induction positions generalizing M with
  | nil =>
      rfl
  | cons p rest ih =>
      change
        (joint_mode_flip_sequence rest
          (joint_mode_flip_at p M) i j) ^ 2 =
          (M i j) ^ 2
      calc
        (joint_mode_flip_sequence rest
          (joint_mode_flip_at p M) i j) ^ 2
            = (joint_mode_flip_at p M i j) ^ 2 :=
              ih (joint_mode_flip_at p M)
        _ = (M i j) ^ 2 :=
              joint_mode_flip_at_entry_sq M p i j

/--
Any even-length sequence of entrywise phase flips
preserves the separability classification.

The flip positions may be arbitrary and may repeat.

The proof uses two-step induction on the sequence:
- An empty sequence leaves the state unchanged.
- A singleton cannot have even length.
- The first two flips preserve separability by
  the double-flip theorem, and the remaining
  sequence has even length.

No assumption about initial separability,
normalization, or nonzero amplitudes is required.
-/
theorem joint_mode_flip_sequence_even_separable_iff
    (positions : List (Fin 2 × Fin 2))
    (M : JointModeState)
    (heven : Even positions.length) :
    joint_mode_separable
        (joint_mode_flip_sequence positions M) ↔
      joint_mode_separable M := by
  revert heven
  induction positions using List.twoStepInduction
      generalizing M with
  | nil =>
      intro _
      rfl
  | singleton p =>
      intro heven
      exact False.elim (Nat.not_even_one heven)
  | cons_cons p q rest ih _ =>
      intro heven
      have hrest : Even rest.length := by
        simpa only [List.length_cons, Nat.succ_eq_add_one,
          Nat.even_add_one, not_not] using heven
      change joint_mode_separable
        (joint_mode_flip_sequence rest
          (joint_mode_flip_at q
            (joint_mode_flip_at p M))) ↔
          joint_mode_separable M
      exact
        (ih (joint_mode_flip_at q
          (joint_mode_flip_at p M)) hrest).trans
          (joint_mode_double_flip_separable_iff M p q)

/--
An odd-length sequence of entrywise phase flips
produces a separable state exactly when the two
determinant products of the initial state are
negatives of one another.

The sequence must contain at least one flip.
After the first flip, an even number of flips
remains, and those preserve separability.

The selected positions may be arbitrary and
may repeat.

No initial separability or normalization
assumption is required.
-/
theorem joint_mode_flip_sequence_odd_separable_iff
    (positions : List (Fin 2 × Fin 2))
    (M : JointModeState)
    (hodd : Odd positions.length) :
    joint_mode_separable
        (joint_mode_flip_sequence positions M) ↔
      M 0 0 * M 1 1 = -(M 0 1 * M 1 0) := by
  cases positions with
  | nil =>
      rcases hodd with ⟨k, hk⟩
      simp only [List.length_nil] at hk
      omega
  | cons p rest =>
      have hrest : Even rest.length := by
        rcases hodd with ⟨k, hk⟩
        refine ⟨k, ?_⟩
        simp only [List.length_cons] at hk
        omega
      change joint_mode_separable
        (joint_mode_flip_sequence rest
          (joint_mode_flip_at p M)) ↔
        M 0 0 * M 1 1 = -(M 0 1 * M 1 0)
      exact
        (joint_mode_flip_sequence_even_separable_iff
          rest (joint_mode_flip_at p M) hrest).trans
          ((joint_mode_flip_at_separable_iff_phase_flip M p).trans
            (joint_mode_phase_flip_separable_iff M))

/-!
## Conditional Joint-Mode Rotation

We investigate a candidate continuous transformation
that rotates the second row of a joint-mode matrix
while leaving the first row unchanged.

This is a mathematical reference operation.

Its physical realizability through the conservative
coupling rules of Standing Wave Spacetime has
not yet been established.
-/

/--
Rotate the second row of a joint-mode matrix
through an angle `θ`, leaving the first row
unchanged.

The transformation acts jointly on the two
mode indices and is not, in general, an
independent transformation of the two factors.
-/
noncomputable def joint_mode_conditional_rotate
    (θ : ℝ) (M : JointModeState) : JointModeState :=
  fun i j =>
    if i = 1 then
      if j = 0 then
        Real.cos θ * M 1 0 - Real.sin θ * M 1 1
      else
        Real.sin θ * M 1 0 + Real.cos θ * M 1 1
    else
      M i j

/--
A separable starting state with the apparatus
initially in its reference mode.
-/
noncomputable def joint_mode_pointer_seed
    (a b : ℝ) : JointModeState :=
  !![a, 0; b, 0]

/--
Under conditional rotation, the determinant
of the initially separable reference state
is the product of its two nonzero amplitudes
and the sine of the rotation angle.

Thus continuous rotation can generate
nonseparability without changing the
underlying state space or introducing
a projection operation.
-/
theorem joint_mode_conditional_rotate_seed_det
    (θ a b : ℝ) :
    Matrix.det
        (joint_mode_conditional_rotate θ
          (joint_mode_pointer_seed a b)) =
      a * b * Real.sin θ := by
  simp [Matrix.det_fin_two,
    joint_mode_conditional_rotate,
    joint_mode_pointer_seed]
  ring

/--
Conditional joint-mode rotation preserves the
total squared magnitude of every joint-mode state.

The transformation leaves the first row unchanged
and rotates the second row through angle `θ`.

The proof uses the ordinary two-dimensional
rotation identity together with
`cos² θ + sin² θ = 1`.

No separability or normalization assumption
is required.

This establishes mathematical conservation,
not yet physical realizability through the
coupling rules of Standing Wave Spacetime.
-/
theorem joint_mode_conditional_rotate_magnitude_sq
    (θ : ℝ) (M : JointModeState) :
    joint_mode_magnitude_sq
        (joint_mode_conditional_rotate θ M) =
      joint_mode_magnitude_sq M := by
  have hunit :
      Real.cos θ ^ 2 + Real.sin θ ^ 2 = 1 := by
    nlinarith [Real.sin_sq_add_cos_sq θ]
  have hrow (x y : ℝ) :
      (Real.cos θ * x - Real.sin θ * y) ^ 2 +
        (Real.sin θ * x + Real.cos θ * y) ^ 2 =
      x ^ 2 + y ^ 2 := by
    calc
      _ = (Real.cos θ ^ 2 + Real.sin θ ^ 2) *
            (x ^ 2 + y ^ 2) := by ring
      _ = x ^ 2 + y ^ 2 := by
        rw [hunit]
        ring
  have htotal :=
    congrArg
      (fun z : ℝ =>
        (M 0 0) ^ 2 + (M 0 1) ^ 2 + z)
      (hrow (M 1 0) (M 1 1))
  simpa [joint_mode_magnitude_sq, Fin.sum_univ_two,
    joint_mode_conditional_rotate, add_assoc] using htotal

/--
Conditional joint-mode rotation is reversible.

Applying a rotation through angle `θ` and
then applying the opposite rotation restores
the original joint-mode state exactly.

The proof uses the sine and cosine identities
for negative angles and the Pythagorean
trigonometric identity.

No separability or normalization assumption
is required.

This proves mathematical reversibility, not
physical realizability through the existing
conservative coupling rules.
-/
theorem joint_mode_conditional_rotate_inverse
    (θ : ℝ) (M : JointModeState) :
    joint_mode_conditional_rotate (-θ)
        (joint_mode_conditional_rotate θ M) = M := by
  have hunit :
      Real.cos θ ^ 2 + Real.sin θ ^ 2 = 1 := by
    nlinarith [Real.sin_sq_add_cos_sq θ]
  have hfirst (x y : ℝ) :
      Real.cos θ *
          (Real.cos θ * x - Real.sin θ * y) +
        Real.sin θ *
          (Real.sin θ * x + Real.cos θ * y) = x := by
    calc
      _ = (Real.cos θ ^ 2 + Real.sin θ ^ 2) * x := by
            ring
      _ = x := by
            rw [hunit]
            ring
  have hsecond (x y : ℝ) :
      -(Real.sin θ) *
          (Real.cos θ * x - Real.sin θ * y) +
        Real.cos θ *
          (Real.sin θ * x + Real.cos θ * y) = y := by
    calc
      _ = (Real.cos θ ^ 2 + Real.sin θ ^ 2) * y := by
            ring
      _ = y := by
            rw [hunit]
            ring
  funext i j
  fin_cases i <;> fin_cases j
  · simp [joint_mode_conditional_rotate]
  · simp [joint_mode_conditional_rotate]
  · simpa [joint_mode_conditional_rotate,
      Real.cos_neg, Real.sin_neg] using
        hfirst (M 1 0) (M 1 1)
  · simpa [joint_mode_conditional_rotate,
      Real.cos_neg, Real.sin_neg] using
        hsecond (M 1 0) (M 1 1)

/-!
## Connecting Pairwise Coupling to Joint-Mode Matrices

We represent the coordinates of two primitive
oscillator states as the rows of a real matrix.

This is an algebraic identification only.
It does not establish that the matrix is a
physical tensor-product state.

We test whether the normalized mixing
operation from Coupling.lean preserves
the determinant of this representation.
-/

/--
Represent two oscillator states as the rows
of a two-by-two real matrix.

The columns contain their q and p coordinates.
-/
noncomputable def joint_mode_of_oscillators
    (A B : OscillatorState) : JointModeState :=
  !![A.q, A.p; B.q, B.p]

/--
Normalized pairwise oscillator mixing preserves
the determinant of the corresponding coordinate
matrix.

This follows because the mixing matrix has
determinant one.

The result is an algebraic constraint on the
existing concrete coupling operation.

No physical interpretation of matrix
separability is assumed.
-/
theorem joint_mode_mix_oscillators_det
    (r : NormalizedRotation)
    (A B : OscillatorState) :
    Matrix.det
      (joint_mode_of_oscillators
        (mix_oscillators r (A, B)).1
        (mix_oscillators r (A, B)).2) =
      Matrix.det (joint_mode_of_oscillators A B) := by
  simp only [Matrix.det_fin_two]
  change
    (r.c * A.q - r.s * B.q) *
        (r.s * A.p + r.c * B.p) -
      (r.c * A.p - r.s * B.p) *
        (r.s * A.q + r.c * B.q) =
      A.q * B.p - A.p * B.q
  calc
    _ = (r.c ^ 2 + r.s ^ 2) *
          (A.q * B.p - A.p * B.q) := by ring
    _ = A.q * B.p - A.p * B.q := by
      rw [r.normalized]
      ring


end StandingWave
