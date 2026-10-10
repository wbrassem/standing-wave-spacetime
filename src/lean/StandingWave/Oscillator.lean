/-
Copyright (c) 2026 Wayne Brassem. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Wayne Brassem
-/

import Mathlib

/-!
# Primitive Oscillatory States

This module establishes an initial mathematical framework for an
abstract system of two-component states. Each state is represented
by two real-valued coordinates, and normalized planar rotations
provide a family of transformations acting on these states.

The construction begins without assuming an ambient physical
spacetime, a characteristic length scale, a time interval, or a
propagation speed. The state components are internal degrees of
freedom rather than coordinates in physical space.

The objective is to identify algebraic structures and invariants
independently of any proposed physical interpretation. Subsequent
modules may introduce interactions between states and investigate
whether collective behavior supports additional mathematical or
physical structure.

In particular, neither a standing-wave substrate nor a holographic
boundary is assumed. Such constructions remain possible future
interpretations rather than premises of the formal development.

## Mathematical Framework

An oscillator state is a pair of real numbers `(q, p)`, equipped
with the quadratic magnitude

    Q(q, p) = q² + p².

This quantity is nonnegative and positive definite:

    Q(q, p) ≥ 0,

    Q(q, p) = 0  ↔  q = 0 ∧ p = 0.

A normalized rotation is specified by real coefficients `(c, s)`
satisfying

    c² + s² = 1,

and acts on an oscillator state by

    q' = c*q - s*p,
    p' = s*q + c*p.

The module establishes that normalized rotations preserve the
quadratic magnitude, including under arbitrary finite sequences
of such transformations.

Normalized rotations are subsequently equipped with composition,
identity, and inverse operations. These operations form a
commutative group.

Finally, the group is identified with the standard matrix
representation of the two-dimensional special orthogonal group.
The correspondence is established as a group isomorphism:

    NormalizedRotation ≃* SO(2).

Thus, the internal rotation operations have a familiar algebraic
structure without requiring a physical interpretation of the
underlying state components.

## Principal Results

* `oscillator_magnitude_sq_nonneg`:
  Nonnegativity of the quadratic magnitude.

* `oscillator_magnitude_sq_eq_zero_iff`:
  The quadratic magnitude vanishes exactly at the zero state.

* `oscillator_magnitude_sq_rotate`:
  Preservation of the quadratic magnitude under a normalized
  planar rotation.

* `apply_rotation_magnitude_sq`:
  Preservation of the quadratic magnitude by a transformation
  represented as a `NormalizedRotation`.

* `rotate_sequence_magnitude_sq`:
  Preservation of the quadratic magnitude under arbitrary
  finite sequences of normalized rotations.

* `compose_rotation_assoc`:
  Associativity of normalized rotation composition.

* `compose_rotation_comm`:
  Commutativity of normalized rotation composition.

* `rotation_matrix_mem_so2`:
  Membership of the rotation matrix in the special
  orthogonal group.

* `rotation_to_so2_mul`:
  Compatibility of rotation composition with matrix
  multiplication.

* `normalized_rotation_equiv_so2`:
  Group isomorphism between normalized rotations and SO(2).

## Development Structure

The principal definitions and results are organized into
three related developments.

1. Primitive states and quadratic magnitude

    OscillatorState
      |
      +-- oscillator_magnitude_sq
      |     |
      |     +-- oscillator_magnitude_sq_nonneg
      |     |
      |     +-- oscillator_magnitude_sq_eq_zero_iff
      |
      +-- oscillator_rotate
            |
            +-- oscillator_magnitude_sq_rotate

2. Normalized transformations and finite compositions

    NormalizedRotation
      |
      +-- apply_rotation
      |     |
      |     +-- apply_rotation_magnitude_sq
      |     |
      |     +-- rotate_sequence
      |           |
      |           +-- rotate_sequence_magnitude_sq
      |
      +-- identity_rotation
      |
      +-- compose_rotation
      |
      +-- inverse_rotation

3. Algebraic structure and matrix representation

    NormalizedRotation
      |
      +-- Identity, composition, and inverse laws
      |     |
      |     +-- CommGroup NormalizedRotation
      |
      +-- rotation_matrix
            |
            +-- rotation_matrix_mem_so2
            |
            +-- rotation_to_so2
            |     |
            |     +-- rotation_to_so2_mul
            |
            +-- so2_to_rotation
                  |
                  +-- normalized_rotation_equiv_so2

The diagrams summarize the logical organization of the
development rather than the exact declaration dependencies
of every Lean proof.

## Scope and Interpretation

The quantity `oscillator_magnitude_sq` is the squared Euclidean
magnitude of an abstract two-component state.

Its nonnegativity, positive definiteness, and preservation
under normalized rotations are mathematical properties
established within this module.

No identification with physical energy, mass, probability,
or another observable quantity is assumed.

Similarly, a normalized rotation is an internal state
transformation rather than a prescribed evolution through
physical time.

The existence of the rotation group does not establish
propagation, coupling, spacetime geometry, or a preferred
direction of time.

Future developments may introduce collective invariants
for interacting states. Such invariants need not coincide
with the quadratic magnitude defined here.

The physical significance of these structures, if any,
remains a question for subsequent mathematical development
and empirical investigation.
-/

namespace StandingWave


/-!
## Quadratic Invariant and Rotational Transformations

We introduce a primitive two-component real state, define its
quadratic magnitude, and establish the basic properties of that
quantity.

The quadratic magnitude is nonnegative and positive definite:
it vanishes precisely when both state components are zero.

We then define planar transformations and restrict their
coefficients to the unit circle to obtain normalized rotations.
These rotations preserve the quadratic magnitude, individually
and under arbitrary finite compositions.

The results establish an invariant of the specified
transformations without introducing physical time, spatial
geometry, propagation, or interactions between distinct states.
-/

/--
A primitive state represented by two real-valued components.

The coordinates `q` and `p` are abstract internal degrees of
freedom. They are not identified with physical position, momentum,
or coordinates of an ambient spacetime.

No normalization, boundedness, or relationship between the
components is imposed. Every pair of real numbers therefore
defines a valid state.

The names `q` and `p` are conventional and do not prescribe
a physical interpretation.
-/
@[ext]
structure OscillatorState where
  q : ℝ
  p : ℝ

/--
The squared Euclidean magnitude of a primitive state.

For a state `x = (q, p)`, the quantity is defined by

    Q(x) = q² + p².

It is nonnegative and vanishes precisely when both state
components are zero.

The quadratic magnitude is preserved by normalized planar
rotations, as established by `oscillator_magnitude_sq_rotate`.

This quantity is defined algebraically. No identification
with physical energy or any other observable quantity
is assumed.
-/
def oscillator_magnitude_sq (x : OscillatorState) : ℝ :=
  x.q ^ 2 + x.p ^ 2

/--
Apply a planar linear transformation to a primitive state.

For a state `(q, p)` and real coefficients `c` and `s`,
the transformation is defined by

    q' = c*q - s*p,
    p' = s*q + c*p.

Equivalently, the transformation is represented by the matrix

    [ c  -s ]
    [ s   c ].

The coefficients are arbitrary real numbers. In general,
the transformation satisfies the scaling identity

    Q(q', p') = (c² + s²)*Q(q, p),

where `Q` denotes `oscillator_magnitude_sq`.

When `c² + s² = 1`, the transformation is a normalized
rotation and preserves the quadratic magnitude.

Without this normalization condition, the transformation
need not preserve the magnitude.

This definition introduces no time parameter, dynamical law,
or physical interpretation of the transformation.
-/
def oscillator_rotate
    (x : OscillatorState)
    (c s : ℝ) : OscillatorState :=
  ⟨c * x.q - s * x.p,
   s * x.q + c * x.p⟩

/--
The squared magnitude of every primitive state is nonnegative.

For any state `x = (q, p)`,

    Q(x) = q² + p² ≥ 0.

Both components are real, so their squares are nonnegative.
Their sum is therefore nonnegative.

This property holds for every oscillator state, independently
of any normalization condition or transformation.
-/
theorem oscillator_magnitude_sq_nonneg
    (x : OscillatorState) :
    0 ≤ oscillator_magnitude_sq x := by
  unfold oscillator_magnitude_sq
  positivity

/--
The quadratic magnitude vanishes exactly at the zero state.

For any primitive state `x = (q, p)`,

    Q(x) = q² + p²,

and therefore

    Q(x) = 0  ↔  q = 0 ∧ p = 0.

Both squared components are nonnegative, so their sum can
vanish only when each component vanishes. Conversely, if both
components are zero, their squared magnitudes sum to zero.

Together with `oscillator_magnitude_sq_nonneg`, this
establishes positive definiteness of the quadratic magnitude.

In particular, a state of zero quadratic magnitude has no
nonzero components. This property may provide useful bounds
in future models of interacting states, although no coupling
or exchange law is assumed here.
-/
theorem oscillator_magnitude_sq_eq_zero_iff
    (x : OscillatorState) :
    oscillator_magnitude_sq x = 0 ↔ x.q = 0 ∧ x.p = 0 := by
  unfold oscillator_magnitude_sq
  constructor
  · intro h
    constructor
    · nlinarith [sq_nonneg x.q, sq_nonneg x.p]
    · nlinarith [sq_nonneg x.q, sq_nonneg x.p]
  · rintro ⟨hq, hp⟩
    rw [hq, hp]
    norm_num

/--
A normalized planar rotation preserves the squared magnitude
of an oscillator state.

For an arbitrary state `(q, p)`, the transformation

    q' = c*q - s*p,
    p' = s*q + c*p

satisfies the general algebraic identity

    q'² + p'² = (c² + s²)*(q² + p²).

Equivalently, writing `Q` for `oscillator_magnitude_sq`,

    Q(oscillator_rotate x c s) = (c² + s²)*Q(x).

Consequently, when the coefficients satisfy the normalization
condition

    c² + s² = 1,

the squared magnitude is preserved exactly:

    oscillator_magnitude_sq (oscillator_rotate x c s)
      = oscillator_magnitude_sq x.

The proof first establishes the general scaling identity by
polynomial expansion and then applies the normalization condition.

This is an algebraic invariance result, independent of any
assumed physical spacetime, characteristic scale, or temporal
evolution law. It does not establish preservation of a collective
invariant under interactions between distinct states.
-/
theorem oscillator_magnitude_sq_rotate
    (x : OscillatorState)
    (c s : ℝ)
    (h : c ^ 2 + s ^ 2 = 1) :
    oscillator_magnitude_sq (oscillator_rotate x c s) =
      oscillator_magnitude_sq x := by
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
A normalized planar rotation represented by two real coefficients.

The coefficients `c` and `s` satisfy the normalization condition

    c² + s² = 1.

They may be interpreted mathematically as the cosine and sine
of a rotation angle, although no angle parameter is introduced
or required by this representation.

The normalization condition ensures that the corresponding
transformation preserves the quadratic magnitude of every
oscillator state.

This structure specifies a normalized transformation rather
than imposing any additional constraint on `OscillatorState`.

Normalized rotations subsequently acquire composition,
identity, and inverse operations, forming a commutative group
isomorphic to SO(2).
-/
@[ext]
structure NormalizedRotation where
  c : ℝ
  s : ℝ
  normalized : c ^ 2 + s ^ 2 = 1

/--
Apply a normalized rotation to an oscillator state.

This operation specializes `oscillator_rotate` to coefficients
provided by a `NormalizedRotation`.

Because the normalization condition is part of the rotation
structure, the transformation preserves the squared magnitude
`oscillator_magnitude_sq` of every oscillator state.

The corresponding invariance theorem is
`apply_rotation_magnitude_sq`.

The operation describes an internal transformation of the
state. It does not introduce physical time, spatial motion,
or a dynamical evolution law.
-/
def apply_rotation
    (r : NormalizedRotation)
    (x : OscillatorState) : OscillatorState :=
  oscillator_rotate x r.c r.s

/--
Every normalized rotation preserves the squared magnitude
of an oscillator state.

For any normalized rotation `r` and state `x`,

    oscillator_magnitude_sq (apply_rotation r x)
      = oscillator_magnitude_sq x.

The result follows directly from `oscillator_magnitude_sq_rotate`,
using the normalization condition `r.normalized` carried by
the rotation structure.

Unlike `oscillator_magnitude_sq_rotate`, this theorem requires
no separate normalization hypothesis: every value of type
`NormalizedRotation` satisfies that condition by construction.

This establishes invariance under individual normalized
rotations. Preservation under arbitrary finite sequences
of rotations is established by `rotate_sequence_magnitude_sq`.

The result concerns an internal state transformation. It
introduces no physical time parameter or interaction law,
and makes no claim about conservation of physical energy.
-/
theorem apply_rotation_magnitude_sq
    (r : NormalizedRotation)
    (x : OscillatorState) :
    oscillator_magnitude_sq (apply_rotation r x) =
      oscillator_magnitude_sq x := by
  exact oscillator_magnitude_sq_rotate x r.c r.s r.normalized

/--
Apply a finite sequence of normalized rotations to an oscillator state.

Starting from the initial state `x`, the rotations in `rs` are
applied successively in list order. Each rotation acts on the
state produced by the preceding rotation.

The implementation uses `List.foldl`, with `x` as the initial
accumulator and `apply_rotation` as the update operation.

For a list of rotations `[r₁, r₂, r₃]`, the resulting state is

    apply_rotation r₃
      (apply_rotation r₂
        (apply_rotation r₁ x)).

An empty list leaves the initial state unchanged.

Since every element of `rs` is a `NormalizedRotation`, each
individual transformation preserves the quadratic magnitude.
Consequently, the complete finite sequence also preserves it,
as established by `rotate_sequence_magnitude_sq`.

The sequence represents an ordered composition of internal
transformations. No time intervals, rotation rates, or
physical evolution law are assumed.
-/
def rotate_sequence
    (x : OscillatorState)
    (rs : List NormalizedRotation) : OscillatorState :=
  rs.foldl (fun state r => apply_rotation r state) x

/--
An arbitrary finite sequence of normalized rotations preserves
the squared magnitude of an oscillator state.

For every initial state `x` and finite list of normalized
rotations `rs`,

    oscillator_magnitude_sq (rotate_sequence x rs)
      = oscillator_magnitude_sq x.

The rotations may have different coefficients at every step.
No assumption of periodicity, a common rotation angle, or
commutativity is required for this invariance result.

The proof proceeds by induction on the list of rotations.

* The empty sequence leaves the initial state unchanged.
* For a nonempty sequence, the first rotation preserves the
  quadratic magnitude by `apply_rotation_magnitude_sq`, and the
  induction hypothesis establishes preservation under the
  remaining rotations.

This establishes invariance under arbitrary finite compositions
of normalized rotations. It does not establish preservation
under arbitrary state transformations or interactions between
distinct oscillator states.

No interpretation as physical time evolution or conservation
of a physical observable is assumed.
-/
theorem rotate_sequence_magnitude_sq
    (x : OscillatorState)
    (rs : List NormalizedRotation) :
    oscillator_magnitude_sq (rotate_sequence x rs) =
      oscillator_magnitude_sq x := by
  unfold rotate_sequence
  induction rs generalizing x with
  | nil =>
      rfl
  | cons r rs ih =>
      simp only [List.foldl_cons]
      rw [ih]
      exact apply_rotation_magnitude_sq r x


/-!
## Algebraic Structure of Normalized Rotations

We define identity, composition, and inverse operations for
normalized planar rotations and establish their algebraic
properties.

Each operation preserves the normalization condition

    c² + s² = 1,

so its result remains a member of `NormalizedRotation`.

The resulting operations satisfy the group laws, including
associativity, identity, and inverse properties. Composition
is also commutative, giving a commutative group.

The identification of this group with the matrix special
orthogonal group SO(2) is established in a subsequent section.

These are algebraic properties of internal transformations.
No temporal evolution law or physical interpretation is assumed.
-/

/--
The identity element of normalized planar rotations.

Its coefficients are

    c = 1,
    s = 0,

which satisfy the normalization condition

    1² + 0² = 1.

Applying this rotation leaves every oscillator state unchanged,
as established by `apply_identity_rotation`.

It also serves as the left and right identity for rotation
composition, as established by the corresponding group laws.
-/
def identity_rotation : NormalizedRotation where
  c := 1
  s := 0
  normalized := by norm_num

/--
Compose two normalized planar rotations.

The expression `compose_rotation r₂ r₁` represents the
composition in which `r₁` is applied first, followed by `r₂`.

For rotations with coefficients

    r₁ = (c₁, s₁),
    r₂ = (c₂, s₂),

the composed coefficients are

    c = c₂*c₁ - s₂*s₁,
    s = s₂*c₁ + c₂*s₁.

These are the algebraic composition formulas for planar
rotations.

The normalization proof establishes

    c² + s²
      = (c₂² + s₂²)*(c₁² + s₁²)
      = 1,

so composing normalized rotations produces another
normalized rotation.

Compatibility between this operation and successive
applications to an oscillator state is established by
`apply_compose_rotation`.

Associativity, commutativity, identity, and inverse laws
are proved separately.
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
          (r₁.c ^ 2 + r₁.s ^ 2) := by
            ring
      _ = 1 := by
        rw [r₂.normalized, r₁.normalized]
        ring

/--
Construct the inverse of a normalized planar rotation.

For a normalized rotation with coefficients `(c, s)`,
the inverse has coefficients

    (c, -s).

The normalization condition is preserved because

    c² + (-s)² = c² + s² = 1.

When interpreted in terms of an angle, this corresponds
to negating the rotation angle.

The inverse transformation undoes the original rotation,
as established by `apply_inverse_rotation`.

The corresponding left and right inverse laws for rotation
composition are established separately.
-/
def inverse_rotation
    (r : NormalizedRotation) :
    NormalizedRotation where
  c := r.c
  s := -r.s
  normalized := by
    simpa using r.normalized

/--
The identity rotation leaves every oscillator state unchanged.

For every state `x`,

    apply_rotation identity_rotation x = x.

The identity rotation has coefficients `(1, 0)`, so its
action reduces to

    q' = q,
    p' = p.

The proof establishes equality of both state components
using the extensionality principle for `OscillatorState`.

This identifies `identity_rotation` as the identity
transformation on the state space.
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
Composition of normalized rotations agrees with successive
application to an oscillator state.

For normalized rotations `r₁` and `r₂` and any state `x`,

    apply_rotation (compose_rotation r₂ r₁) x
      = apply_rotation r₂ (apply_rotation r₁ x).

Thus `compose_rotation r₂ r₁` represents applying `r₁`
first and then applying `r₂`.

The proof expands the component formulas for both sides,
reduces equality of states to equality of their components,
and verifies the resulting polynomial identities.

No normalization hypotheses are needed in the proof of this
compatibility identity beyond those already carried by
the rotation structures.

This establishes compatibility between rotation composition
and the transformations they induce on the state space.
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
Applying a normalized rotation followed by its inverse
recovers the original oscillator state.

For every normalized rotation `r` and state `x`,

    apply_rotation (inverse_rotation r)
      (apply_rotation r x) = x.

The inverse rotation has coefficients `(r.c, -r.s)`.
Expanding the successive transformations shows that
both state components are multiplied by

    r.c² + r.s².

The normalization property

    r.c² + r.s² = 1

therefore reduces the resulting state to the original
state `x`.

The proof verifies this identity separately for the two
components using polynomial identities and normalization.

This establishes that the inverse rotation undoes the
original transformation on every oscillator state.
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
Composition of normalized planar rotations is commutative.

For any normalized rotations `r₁` and `r₂`,

    compose_rotation r₁ r₂ = compose_rotation r₂ r₁.

The result follows from the commutativity of multiplication
in the real numbers. Expanding the coefficients of both
compositions produces equal polynomial expressions,
independently of the order of the rotations.

The proof uses extensionality for `NormalizedRotation`,
reducing equality of rotations to equality of their
coefficients. Each coefficient identity is then established
by elementary polynomial algebra.

The normalization hypotheses are not explicitly needed
for these polynomial identities; they are part of the
rotation structures.

Commutativity is characteristic of planar rotation
composition and distinguishes the resulting group from
the generally noncommutative rotation groups in higher
dimensions.

The identification with SO(2) is established separately.
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
The identity rotation is a left identity for rotation composition.

For every normalized rotation `r`,

    compose_rotation identity_rotation r = r.

Since `identity_rotation` has coefficients `(1, 0)`,
the composition formulas reduce to

    c = 1*r.c - 0*r.s = r.c,
    s = 0*r.c + 1*r.s = r.s.

The proof uses extensionality to reduce equality of rotations
to equality of their coefficients, followed by simplification.

This establishes the left identity law independently of
commutativity or the other group laws.
-/
theorem compose_rotation_identity_left
    (r : NormalizedRotation) :
    compose_rotation identity_rotation r = r := by
  ext <;>
    simp [compose_rotation, identity_rotation]

/--
The identity rotation is also a right identity for composition.

For every normalized rotation `r`,

    compose_rotation r identity_rotation = r.

The result follows from `compose_rotation_comm`, which
reverses the composition order, and
`compose_rotation_identity_left`, which establishes the
corresponding left identity law.

Thus no separate calculation of the composition
coefficients is required.

Together, the left and right identity theorems establish
that `identity_rotation` is a two-sided identity for
normalized rotation composition.
-/
theorem compose_rotation_identity_right
    (r : NormalizedRotation) :
    compose_rotation r identity_rotation = r := by
  rw [compose_rotation_comm]
  exact compose_rotation_identity_left r

/--
The inverse of a normalized rotation is a left inverse
under rotation composition.

For every normalized rotation `r`,

    compose_rotation (inverse_rotation r) r
      = identity_rotation.

The inverse rotation has coefficients `(r.c, -r.s)`.
Expanding the composition formulas gives

    c = r.c² + r.s²,
    s = 0.

The normalization condition

    r.c² + r.s² = 1

therefore reduces the composed coefficients to `(1, 0)`,
which are precisely those of `identity_rotation`.

The proof establishes equality of the two coefficients
separately. The first equality requires normalization,
while the second is a polynomial identity independent
of normalization.

This establishes the left inverse law directly from
the definitions.
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
The inverse of a normalized rotation is also a right inverse
under rotation composition.

For every normalized rotation `r`,

    compose_rotation r (inverse_rotation r)
      = identity_rotation.

The result follows from `compose_rotation_comm`, which
reverses the composition order, and
`compose_rotation_inverse_left`, which establishes the
corresponding left inverse law.

Thus no separate expansion of the rotation coefficients
is required.

Together, the left and right inverse theorems establish
that `inverse_rotation r` is a two-sided inverse of `r`
under normalized rotation composition.
-/
theorem compose_rotation_inverse_right
    (r : NormalizedRotation) :
    compose_rotation r (inverse_rotation r) =
      identity_rotation := by
  rw [compose_rotation_comm]
  exact compose_rotation_inverse_left r


/-!
## Commutative Group Structure

The normalized planar rotations form a commutative group
under the composition operation defined in the preceding
section.

The group structure is established from the previously
proved algebraic properties:

* Closure under composition is guaranteed by the return
  type of `compose_rotation`.
* Associativity is established by `compose_rotation_assoc`.
* The left and right identity laws are established by
  `compose_rotation_identity_left` and
  `compose_rotation_identity_right`.
* The inverse law is established by
  `compose_rotation_inverse_left`.
* Commutativity is established by `compose_rotation_comm`.

Registering these results as a `CommGroup` instance allows
normalized rotations to use Lean's standard group notation
and theorems.

This structure concerns the algebra of internal state
transformations. It does not introduce a physical evolution
law or imply that other transformations of the model must
commute.
-/

/--
Equip normalized planar rotations with a commutative group
structure under composition.

The group operations are identified with the concrete
operations defined previously:

* Multiplication is `compose_rotation`.
* The identity element is `identity_rotation`.
* Inversion is `inverse_rotation`.

The required group laws are supplied by the corresponding
theorems established in the preceding section.

In particular, multiplication is associative and
commutative, every rotation has a two-sided identity,
and every rotation has an inverse.

This instance makes the standard notation

    r₁ * r₂
    1
    r⁻¹

available for values of type `NormalizedRotation`.

The group structure is purely algebraic. Its subsequent
identification with the matrix group SO(2) is established
separately.
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
Group multiplication of normalized rotations is their
previously defined composition operation.

For any normalized rotations `r₁` and `r₂`,

    r₁ * r₂ = compose_rotation r₁ r₂.

This equality follows directly from the `CommGroup`
instance, whose multiplication operation is defined
to be `compose_rotation`.

The proof is by reflexivity because the two expressions
are definitionally equal.

This theorem provides an explicit bridge between the
standard group notation and the concrete composition
operation used in earlier results.
-/
theorem normalized_rotation_mul_def
    (r₁ r₂ : NormalizedRotation) :
    r₁ * r₂ = compose_rotation r₁ r₂ := by
  rfl

/--
Multiplication of normalized rotations is commutative.

For any normalized rotations `r₁` and `r₂`,

    r₁ * r₂ = r₂ * r₁.

This is the commutativity law of the `CommGroup`
instance for `NormalizedRotation`.

The proof applies Mathlib's general theorem `mul_comm`,
which is available because the group structure has
already been established.

Unlike `compose_rotation_comm`, this theorem uses
the abstract group interface rather than expanding
the concrete rotation coefficients.

It provides a named statement of commutativity in
standard group notation.
-/
theorem normalized_rotation_mul_comm
    (r₁ r₂ : NormalizedRotation) :
    r₁ * r₂ = r₂ * r₁ := by
  exact mul_comm r₁ r₂


/-!
## Matrix Representation of SO(2)

We associate each normalized planar rotation with its standard
two-by-two real rotation matrix.

For normalized coefficients `(c, s)` satisfying

    c² + s² = 1,

the corresponding matrix is

    [ c  -s ]
    [ s   c ].

This representation connects the abstract rotation operations
developed in the preceding sections with the matrix special
orthogonal group SO(2).

The development establishes that these matrices belong to SO(2),
that the representation is injective, and that every element of
SO(2) corresponds to a normalized rotation.

Compatibility with multiplication is established subsequently,
allowing the correspondence to be assembled into a group
isomorphism.

The matrices represent transformations of abstract internal
state coordinates. No identification with rotations in physical
space is assumed.
-/

/--
The standard two-by-two matrix representation of a
normalized planar rotation.

For a normalized rotation `r` with coefficients `(r.c, r.s)`,
the corresponding matrix is

    [ r.c  -r.s ]
    [ r.s   r.c ].

This matrix acts on the real coordinate pair `(q, p)` by

    q' = r.c*q - r.s*p,
    p' = r.s*q + r.c*p,

which agrees with the transformation defined by
`apply_rotation`.

The normalization condition

    r.c² + r.s² = 1

ensures that the matrix is orthogonal with determinant one.

Membership in the matrix special orthogonal group SO(2)
is established separately by `rotation_matrix_mem_so2`.

This definition introduces the matrix representation only;
it does not itself establish the group-theoretic properties
of the correspondence.
-/
def rotation_matrix
    (r : NormalizedRotation) :
    Matrix (Fin 2) (Fin 2) ℝ :=
  !![r.c, -r.s; r.s, r.c]

/--
The matrix representation of every normalized rotation
belongs to the special orthogonal group SO(2).

For a normalized rotation `r` with coefficients `(c, s)`,
the associated matrix is

    [ c  -s ]
    [ s   c ].

Membership in SO(2) requires the matrix to be orthogonal
and to have determinant one.

For this particular matrix, both properties follow from
the normalization condition

    c² + s² = 1.

The proof uses Mathlib's characterization of two-dimensional
special orthogonal matrices,
`Matrix.of_mem_specialOrthogonalGroup_fin_two_iff`.

The structural equalities between matrix entries hold by
definition. The remaining quadratic identity follows
directly from `r.normalized`, since `(-s)² = s²`.

This establishes that every normalized rotation has a
representation in SO(2). The converse correspondence and
the group isomorphism are established separately.

The result concerns an abstract matrix representation and
does not identify the state coordinates with physical space.
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
The matrix representation of normalized rotations is injective.

For any normalized rotations `r₁` and `r₂`,

    rotation_matrix r₁ = rotation_matrix r₂
      → r₁ = r₂.

The coefficients of a normalized rotation can be recovered
directly from its matrix representation:

    r.c = (rotation_matrix r) 0 0,
    r.s = (rotation_matrix r) 1 0.

Consequently, equal matrix representations imply equal
rotation coefficients and hence equal normalized rotations.

The proof extracts the relevant matrix-entry equalities
using `congrArg`, then applies the extensionality principle
for `NormalizedRotation`.

No normalization identities are required for this argument;
injectivity follows directly from the matrix representation.

Together with the subsequent construction of an inverse
correspondence, this result contributes to identifying
normalized rotations with SO(2).
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
Represent a normalized planar rotation as an element of
the matrix special orthogonal group SO(2).

For a normalized rotation `r`, this construction packages

    rotation_matrix r

together with the proof `rotation_matrix_mem_so2 r`
that the matrix belongs to SO(2).

The resulting value has type

    Matrix.specialOrthogonalGroup (Fin 2) ℝ,

rather than the unrestricted type of real two-by-two matrices.

This definition does not perform any additional matrix
calculation. It uses the membership theorem established
previously.

Injectivity follows from `rotation_matrix_injective`.
Compatibility with group multiplication is established
separately.
-/
def rotation_to_so2
    (r : NormalizedRotation) :
    Matrix.specialOrthogonalGroup (Fin 2) ℝ :=
  ⟨rotation_matrix r, rotation_matrix_mem_so2 r⟩

/--
Construct a normalized planar rotation from an arbitrary
element of the matrix special orthogonal group SO(2).

Given `M ∈ SO(2)`, let `A` denote its underlying real
two-by-two matrix.

The rotation coefficients are recovered from the first
column of `A`:

    c = A 0 0,
    s = A 1 0.

Mathlib's characterization of SO(2) supplies the
structural identities

    A 0 1 = -A 1 0,
    A 0 0 ^ 2 + A 0 1 ^ 2 = 1.

Substituting the first identity into the second gives

    (A 0 0)² + (A 1 0)² = 1,

which establishes the normalization condition required
by `NormalizedRotation`.

The proof extracts these identities from the membership
property of `M` and derives normalization by elementary
polynomial reasoning.

This constructs a normalized rotation from every element
of SO(2). The subsequent inverse-correspondence theorems
establish that this construction and `rotation_to_so2`
are mutually inverse.
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

For every normalized rotation `r`,

    so2_to_rotation (rotation_to_so2 r) = r.

The forward conversion constructs the matrix

    [ r.c  -r.s ]
    [ r.s   r.c ],

and the reverse conversion extracts its first-column
entries, recovering the original coefficients `r.c`
and `r.s`.

The proof uses extensionality for `NormalizedRotation`.
Both coefficient equalities hold by definitional
reduction, so each is proved by reflexivity.

This establishes that `so2_to_rotation` is a left
inverse of `rotation_to_so2`.
-/
theorem so2_to_rotation_rotation_to_so2
    (r : NormalizedRotation) :
    so2_to_rotation (rotation_to_so2 r) = r := by
  ext
  · rfl
  · rfl

/--
Converting an element of SO(2) to a normalized rotation
and back recovers the original element of SO(2).

For every `M : Matrix.specialOrthogonalGroup (Fin 2) ℝ`,

    rotation_to_so2 (so2_to_rotation M) = M.

The reverse conversion extracts the first-column entries
of the underlying matrix:

    c = M.1 0 0,
    s = M.1 1 0.

The forward conversion reconstructs the matrix

    [ c  -s ]
    [ s   c ].

Mathlib's characterization of SO(2) ensures that the
original matrix has this form. In particular,

    M.1 0 1 = -M.1 1 0,
    M.1 0 0 = M.1 1 1.

The proof first reduces equality of SO(2) elements to
equality of their underlying matrices, then verifies
all four entries.

The first-column entries agree definitionally.
The remaining entries agree by the structural identities
supplied by SO(2) membership.

This establishes that `rotation_to_so2` is a left
inverse of `so2_to_rotation`.

Together with `so2_to_rotation_rotation_to_so2`,
it establishes a bijection between normalized rotations
and SO(2).
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
The correspondence from normalized rotations to SO(2)
preserves group multiplication.

For any normalized rotations `r₁` and `r₂`,

    rotation_to_so2 (r₁ * r₂)
      = rotation_to_so2 r₁ * rotation_to_so2 r₂.

Thus composing rotations and then converting to SO(2)
gives the same result as converting each rotation to
SO(2) and multiplying the corresponding matrices.

The proof first unfolds the group multiplication on
`NormalizedRotation` using `normalized_rotation_mul_def`.

Equality in SO(2) is then reduced to equality of the
underlying matrices, and subsequently to equality of
their individual entries.

For each of the four matrix entries, expanding the
rotation composition and matrix multiplication formulas
reduces the required equality to a polynomial identity,
which is established by `ring`.

This establishes the multiplication-preservation property
needed to construct a group isomorphism between
`NormalizedRotation` and SO(2).
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
The group isomorphism between normalized planar rotations
and the two-dimensional special orthogonal group SO(2).

The forward map `rotation_to_so2` associates each normalized
rotation with its corresponding matrix in SO(2).

The inverse map `so2_to_rotation` reconstructs a normalized
rotation from an arbitrary element of SO(2).

The theorems `so2_to_rotation_rotation_to_so2` and
`rotation_to_so2_so2_to_rotation` establish that these
maps are mutually inverse.

The theorem `rotation_to_so2_mul` establishes that the
forward map preserves group multiplication.

Together, these results define a multiplicative equivalence
(`MulEquiv`) between `NormalizedRotation` and
`Matrix.specialOrthogonalGroup (Fin 2) ℝ`.

This is an isomorphism of groups, not merely a bijection
of their underlying sets.

The result concerns the algebraic structure of normalized
transformations and makes no identification of the abstract
state coordinates with physical spatial dimensions.
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
