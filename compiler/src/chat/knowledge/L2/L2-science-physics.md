# physics (L2)
science/physics — Physics simulation

## Functions
integrate_euler(state: array, dt: float, f: string) → array — Euler step
integrate_rk4(state: array, dt: float, f: string) → array — RK4 step
spring_damper(x: float, v: float, k: float, c: float, dt: float) → map — Spring step
projectile(x: float, y: float, vx: float, vy: float, dt: float) → map — Projectile step
kinetic_energy(m: float, v: float) → float — 0.5*m*v^2
potential_energy(m: float, h: float) → float — m*g*h
gravitational_force(m1: float, m2: float, r: float) → float — Newton gravity
wave_equation_1d(u: array, dx: float, dt: float, c: float) → array — 1D wave step
