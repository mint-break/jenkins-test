# Unity setup (Built-in Render Pipeline, Unity 2020.3)

These files were written without a Unity editor available, so they have not
been compiled or run in Unity yet. The lighting maths is the same as
`tools/sprite_stack/body3d/lit.py`, which renders `Preview/stacked_preview.png`.

## Files
- `SpriteStackLit.shader` - layer shader with a light fixed on screen
- `SpriteStackCharacter.cs` - builds the stack from several parts on one layer grid and turns it

## Steps
1. Copy the `Albedo` and `Normal` folders of Body, Face, Leaves, Limbs/Arms, Limbs/Feet, Tail, `SpriteStackLit.shader`
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
   - Face: sprites/normals of `Face/`, layerIndex from `Face/stack_config.json` (fractional),
     span 0.125, rank 2, twoSided off, glossScale 0
   - Arms / Feet: sprites/normals of `Limbs/Arms` and `Limbs/Feet`, layerIndex from their
     `stack_config.json` (Feet go negative: below the body's first layer), span 1, rank 3 / 4,
     twoSided off, glossScale 0
   - Tail: sprites/normals of `Tail/`, layerIndex from `Tail/stack_config.json`, span 1, rank 5,
     twoSided off, glossScale 0
   - Leaves: the sprites/normals of `Leaves/`, layerIndex from `Leaves/stack_config.json`
     (`layer_index`, currently 61..84), rank 1, **twoSided on**, glossScale 0.35
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
- `copiesPerLayer` (8) draws each body layer 8 times across its height so no gaps show.
  Keep it at 8: the Face layers (1/8 spacing) sit exactly on those copies' rims; with a
  different value the face turns stripy.
- Outline: render the stack to a RenderTexture and outline that once; do not outline layers.
- Round the stack's screen position to whole pixels if it looks jittery.
- Sort whole stacks against other objects by their ground point, not layer by layer.
- Without the custom shader, use the baked `Body/Layer_###.png` (light from straight above),
  with the same import settings as the Albedo PNGs (Alpha Is Transparency ON).

## Smooth body (optional, recommended)
The plain stack draws each body layer's single cross-section several times across its height,
so the outline and the shading step at every layer boundary (horizontal banding, most visible
from the back and at 45 deg). With `interpolate` on, every copy blends the two nearest layers:
the signed distance in `Body/SDF` (8-bit, square-root companded, +-48 px; see the shader) is interpolated and
thresholded, and the normals are interpolated, so outline and shading change continuously.
- Import `Body/SDF/*.png` as Default, **sRGB off**, Single Channel (R), Compression None,
  Filter Bilinear, and assign them to the Body part's `sdf` (000..063), tick `interpolate`.
- Use `Body/NormalExt/*.png` (normals extended over the whole canvas; import like the
  Normal maps) as the Body part's `normals` when `interpolate` is on: interpolation reads a
  layer's normal where the next layer covers the pixel.
- Albedo sprites of the Body: Mesh Type **Full Rect** (a tight mesh would clip the interpolated
  outline where the next layer is larger).
- Other parts keep `interpolate` off.
