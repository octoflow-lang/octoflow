# physics2d (L2)
game/physics2d — 2D physics (gravity, Euler integration, wall bounce, collision)

## Functions
phys_add_body(world: map, x: float, y: float, vx: float, vy: float, r: float) → int
  Add circular body, return body ID
phys_gravity(world: map, g: float) → float
  Apply gravity acceleration to all bodies
phys_integrate(world: map, dt: float) → float
  Euler integration step for all bodies
phys_bounce_walls(world: map, w: float, h: float) → float
  Bounce bodies off rectangular boundary
phys_collide_pairs(world: map) → float
  Resolve circle-circle collisions
phys_kinetic_energy(world: map) → float
  Sum kinetic energy of all bodies
