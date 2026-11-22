# Unity setup (Built-in Render Pipeline, Unity 2020.3)

These files were written without a Unity editor available, so they have not
been compiled or run in Unity yet. The lighting maths is the same as
`tools/sprite_stack/body3d/lit.py`, which renders `Preview/stacked_preview.png`.

## Files
- `SpriteStackLit.shader` - layer shader with a light fixed on screen
- `SpriteStackBody.cs` - builds the stack from the 64 layers and turns it

## Steps
1. Copy `Body/Albedo`, `Body/Normal`, `SpriteStackLit.shader` and `SpriteStackBody.cs` into `Assets/`.
2. Albedo PNGs: Texture Type **Sprite (2D and UI)**, Sprite Mode Single, Pivot **Center**,
   Pixels Per Unit e.g. 100, Filter Bilinear, Compression None (or High Quality),
   **Alpha Is Transparency ON**. The transparent pixels are black (rgb 0); without this option
   bilinear filtering darkens every layer edge and the stack shows dark rings.
   Generate Mip Maps ON if the stack is drawn much smaller than 512 px.
3. Normal PNGs: Texture Type **Default**, **sRGB (Color Texture) off**, Compression None,
   Filter Bilinear. Do *not* import them as "Normal map" (that re-encodes the channels).
4. Create a Material with shader **SpriteStack/Lit**.
5. Add `SpriteStackBody` to an empty GameObject; assign the 64 Albedo sprites and the 64
   Normal textures in order (000 = bottom), the material, `layerSpacingPx` from
   `Body/stack_config.json` (`layer_spacing_px`) and the same Pixels Per Unit.
6. Turn the character by changing `yaw` (degrees). The light stays at the screen's upper left.

## Notes (from the sprite-stacking pitfalls)
- `fillCopies` draws each layer several times between layers so no gaps show;
  about `layer_spacing_px * on-screen scale` copies is enough.
- Outline: render the stack to a RenderTexture and outline that once; do not outline layers.
- Round the stack's screen position to whole pixels if it looks jittery.
- Sort whole stacks against other objects by their ground point, not layer by layer.
- Without the custom shader, use the baked `Body/Layer_###.png` (light from straight above),
  with the same import settings as the Albedo PNGs (Alpha Is Transparency ON).
