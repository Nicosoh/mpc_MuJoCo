import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

def clamp(x, lo=0.0, hi=1.0):
    return ca.fmin(ca.fmax(x, lo), hi)

def closest_pt_segment_segment():
    # Inputs
    p1 = ca.SX.sym("p1", 3)                 # Point 1 of segment 1
    q1 = ca.SX.sym("q1", 3)                 # Point 2 of segment 1
    p2 = ca.SX.sym("p2", 3)                 # Point 1 of segment 2
    q2 = ca.SX.sym("q2", 3)                 # Point 2 of segment 2

    eps = 1e-8

    d1 = q1 - p1                            # Direction vector of segment 1
    d2 = q2 - p2                            # Direction vector of segment 2
    r  = p1 - p2

    a = ca.dot(d1, d1)                      # Squared length of segment 1
    e = ca.dot(d2, d2)                      # Squared length of segment 2
    f = ca.dot(d2, r)
    c = ca.dot(d1, r)
    b = ca.dot(d1, d2)

    denom = a * e - b * b

    # Initialize
    s = ca.SX(0)                            # Position along segment 1
    t = ca.SX(0)                            # Position along segment 2

    # --- Both segments degenerate into points ---
    both_degenerate = ca.logic_and(a <= eps, e <= eps)

    s = ca.if_else(both_degenerate, 0.0, s)             # Set s = 0 if true
    t = ca.if_else(both_degenerate, 0.0, t)             # Set t = 0 if true

    # --- First segment degenerate into a point only ---
    first_degenerate = ca.logic_and(a <= eps, e > eps)
    t_fd = clamp(f / e)
    s = ca.if_else(first_degenerate, 0.0, s)            # Set s = 0 if true
    t = ca.if_else(first_degenerate, t_fd, t)           # Set t as the clamped point

    # --- Second segment degenerate into a point only ---
    second_degenerate = ca.logic_and(a > eps, e <= eps)
    s_sd = clamp(-c / a)
    s = ca.if_else(second_degenerate, s_sd, s)          # Set s as the clamped point
    t = ca.if_else(second_degenerate, 0.0, t)           # Set t = 0 if true

    # --- General case ---
    general = ca.logic_and(a > eps, e > eps)            # True is both is not a point

    s_gc = ca.if_else(                                  # If lines are not parallel (denom > 0)
        denom > eps,                                    # Then clamp segment 1
        clamp((b * f - c * e) / denom),                 # If not set s to zero.
        0.0
    )

    t_gc = (b * s_gc + f) / e                           # Compute closest point on segment 2 w.r.t s_gc.

    # Clamp t and recompute s if needed
    t_gc_clamped = clamp(t_gc)                          # Premptive clamping, does not matter if it already within [0,1]

    s_gc = ca.if_else(                                  
        t_gc < 0.0,                                     # If t is less than 0.0, outside of the segment.
        clamp(-c / a),                                  # Clamp t to 0.0 and recalculate s
        ca.if_else(                                     
            t_gc > 1.0,                                 # If t is more than 1.0, outside of segment
            clamp((b - c) / a),                         # Clamp t to 1.0 and recalculate s
            s_gc                                        # If both not true, then s_gc remains and done.
        )
    )

    s = ca.if_else(general, s_gc, s)
    t = ca.if_else(general, t_gc_clamped, t)

    # Closest points
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t

    dist2 = ca.dot(c1 - c2, c1 - c2)

    return ca.Function(
        "closest_seg_seg",
        [p1, q1, p2, q2],
        [dist2, s, t, c1, c2],
        ["p1", "q1", "p2", "q2"],
        ["dist2", "s", "t", "c1", "c2"]
    )

# Evaluate CasADi function
f = closest_pt_segment_segment()

p1 = np.array([0.0, 0.0, 1.0])
q1 = np.array([0.0, 0.0, 1.5])
p2 = np.array([0.0, 0.0, 0.0])
q2 = np.array([0.8, 0.0, 0.2])

dist2, s, t, c1, c2 = f(p1, q1, p2, q2)

# Convert CasADi -> NumPy
c1 = np.array(c1).astype(float).flatten()
c2 = np.array(c2).astype(float).flatten()

# -------------------------------
# 2D Plot (X-Z plane)
# -------------------------------
plt.figure(figsize=(6, 6))

# Segment 1
plt.plot(
    [p1[0], q1[0]],
    [p1[2], q1[2]],
    "b-", linewidth=2, label="Segment 1"
)

# Segment 2
plt.plot(
    [p2[0], q2[0]],
    [p2[2], q2[2]],
    "r-", linewidth=2, label="Segment 2"
)

# Closest points
plt.scatter(c1[0], c1[2], color="blue", s=60)
plt.scatter(c2[0], c2[2], color="red", s=60)

# Shortest distance line
plt.plot(
    [c1[0], c2[0]],
    [c1[2], c2[2]],
    "k--", linewidth=1.5, label="Closest distance"
)

# Formatting
plt.xlabel("X")
plt.ylabel("Z")
plt.title(f"Closest distance (squared) = {float(dist2):.4f}")
plt.legend()
plt.axis("equal")
plt.grid(True)

plt.show()