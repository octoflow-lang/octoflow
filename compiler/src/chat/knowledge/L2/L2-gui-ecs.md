# ecs (L2)
game/ecs — Entity Component System (SoA, position/velocity/hp)

## Functions
ecs_new(max: int) → map — create world
ecs_create(world: map) → int — spawn entity
ecs_add_pos(world: map, id: int, x, y: float) → float
ecs_add_vel(world: map, id: int, vx, vy: float) → float
ecs_add_hp(world: map, id: int, hp: float) → float
ecs_update_physics(world: map, dt: float) → float
ecs_damage(world: map, id: int, amount: float) → float
ecs_get_pos(world: map, id: int) → array — [x, y]
ecs_alive_count(world: map) → int — entities with hp > 0
