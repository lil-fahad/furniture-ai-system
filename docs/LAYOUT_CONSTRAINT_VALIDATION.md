# Layout Constraint Validation

FurnitureAI validates generated or user-supplied furniture layouts with deterministic geometry before a layout is treated as executable.

## Hard constraints

The validator reports an error when:

- a furniture footprint has zero width or depth;
- any part of a furniture footprint extends outside its room polygon;
- two furniture footprints overlap by a non-trivial area;
- furniture intersects a door opening;
- an explicitly requested minimum clearance is violated.

## Clearance policy

`minimum_clearance` defaults to `0`. FurnitureAI does not invent a universal walkway or furniture-spacing distance when the plan has no applicable rule or user requirement. When provided, clearance is interpreted in the coordinate units used by the supplied room and furniture geometry.

A future rules layer may supply code-, project-, jurisdiction-, or user-specific clearance requirements, but those values must carry their own source/provenance rather than being silently embedded in this geometry validator.

## Trust boundary

LLM or vision output may propose placements, but it cannot override constraint results. Geometry validation is deterministic and independent of model confidence. The report contains explicit issue codes and affected room/item/opening identifiers so callers can reject, repair, or regenerate an invalid layout.

## Current issue codes

- `invalid_footprint`
- `outside_room`
- `collision`
- `door_blocked`
- `clearance_violation`

The validator does not claim that a geometrically valid layout is automatically compliant with every building code, accessibility standard, manufacturer instruction, or human preference. Those require separately sourced rule sets and evaluation.
