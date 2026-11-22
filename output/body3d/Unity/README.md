# Unity setup (Built-in Render Pipeline, Unity 2020.3)

These files were written without a Unity editor available, so they have not
been compiled or run in Unity yet. The lighting maths is the same as
`tools/sprite_stack/body3d/lit.py`, which renders `Preview/stacked_preview.png`.

## Files
- `SpriteStackLit.shader` - layer shader with a light fixed on screen
- `SpriteStackCharacter.cs` - builds the stack from several parts on one layer grid and turns it

## Steps
1. Copy `Body/Albedo`, `Body/Normal`, `Leaves/Albedo`, `Leaves/Normal`, `SpriteStackLit.shader`
   and `SpriteStackCharacter.cs` into `Assets/`.
2. Albedo PNGs: Texture Type **Sprite (2D and UI)**, Sprite Mode Single, Pivot **Center**,
   Pixels Per Unit e.g. 100, Filter Bilinear, Compression None (or High Quality),
   **Alpha Is Transparency ON**. The transparent pixels are black (rgb 0); without this option
   bilinear filtering darkens every layer edge and the stack shows dark rings.
   Generate Mip Maps ON if the stack is drawn much smaller than 512 px.
3. Normal PNGs: Texture Type **Default**, **sRGB (Color Texture) off**, Compression None,
   Filter Bilinear. Do *not* import them as "Normal map" (that re-encodes the channels).
4. Create a Material with shader **SpriteStack/Lit**.
5. Add `SpriteStackCharacter` to an empty GameObject and add one Part per folder:
   - Body: 64 Albedo sprites + 64 Normal textures, layerIndex 0..63, rank 0, twoSided off, glossScale 1
   - Leaves: the sprites/normals of `Leaves/`, layerIndex from `Leaves/stack_config.json`
     (`layer_index`), rank 1, **twoSided on**, glossScale 0.35
   Set the material, `layerSpacingPx` (`character_config.json` → `layer_spacing_px`) and the
   same Pixels Per Unit. `character_config.json` → `draw` lists the resulting order.
6. Turn the character by changing `yaw` (degrees). The light stays at the screen's upper left.

## Notes (from the sprite-stacking pitfalls)
- At diagonal yaws (around 45 deg) the near-vertical lower body shows a faint brick-like
  pattern (about 1.6 brightness levels, std): every layer edge becomes a pixel staircase and
  the fill copies repeat it. It is inherent to sprite stacking with rotated layers; drawing
  the stack at 2x into a RenderTexture and scaling it down hides it.
- Draw the stack at (or near) its native size, e.g. into a RenderTexture, and scale the
  result down. Leaves are thin strips per layer; drawing every layer already shrunk makes
  them semi-transparent and stripy.
- `fillCopies` draws each layer several times between layers so no gaps show;
  about `layer_spacing_px * on-screen scale` copies is enough.
- Outline: render the stack to a RenderTexture and outline that once; do not outline layers.
- Round the stack's screen position to whole pixels if it looks jittery.
- Sort whole stacks against other objects by their ground point, not layer by layer.
- Without the custom shader, use the baked `Body/Layer_###.png` (light from straight above),
  with the same import settings as the Albedo PNGs (Alpha Is Transparency ON).
