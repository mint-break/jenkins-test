// Builds a sprite stack from the exported layers and keeps every layer's
// lighting consistent while it turns (Built-in Render Pipeline, Unity 2020.3).
//
// Setup (see README.md):
//   1. Import Body/Albedo/*.png as Sprite (Single), pivot Center, Pixels Per Unit = ppu below.
//   2. Import Body/Normal/*.png as Default texture, sRGB OFF, compression None (NOT "Normal map").
//   3. Make a material with the "SpriteStack/Lit" shader.
//   4. Add this component to an empty GameObject, assign layers/normals in order 000..063.
//
// NOTE: written without access to a Unity editor in this environment, so it
// has not been compiled or run in Unity yet.
using UnityEngine;

public class SpriteStackBody : MonoBehaviour
{
    [Tooltip("Body/Albedo/Layer_000..063 as Sprites, bottom first")]
    public Sprite[] layers;
    [Tooltip("Body/Normal/Layer_000..063 textures, same order")]
    public Texture2D[] normals;
    public Material material;

    [Tooltip("layer_spacing_px from stack_config.json")]
    public float layerSpacingPx = 6.2297f;
    [Tooltip("Pixels Per Unit used when importing the sprites")]
    public float pixelsPerUnit = 100f;
    [Tooltip("Vertical squash = sin(camera pitch); stack_config.json rendering.recommended_squash")]
    [Range(0.05f, 1f)] public float squash = 0.75f;
    [Tooltip("Draw each layer this many times between layers so the stack has no gaps")]
    [Range(1, 12)] public int fillCopies = 7;
    [Tooltip("Stack yaw in degrees (counter-clockwise on screen)")]
    public float yaw;
    public int baseSortingOrder;

    Transform _root;
    SpriteRenderer[] _renderers;
    MaterialPropertyBlock _mpb;

    void Start() { Build(); }

    public void Build()
    {
        if (_root != null) Destroy(_root.gameObject);
        _root = new GameObject("Stack").transform;
        _root.SetParent(transform, false);
        _mpb = new MaterialPropertyBlock();

        int n = layers.Length;
        _renderers = new SpriteRenderer[n * fillCopies];
        float step = layerSpacingPx / pixelsPerUnit;           // world units between layers
        int k = 0;
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < fillCopies; c++)
            {
                var go = new GameObject($"L{i:000}_{c}");
                // the squash parent: rotate first (child), then squash (parent) - like the preview
                var squashT = new GameObject("squash").transform;
                squashT.SetParent(_root, false);
                squashT.localPosition = new Vector3(0f, (i + (float)c / fillCopies) * step, 0f);
                squashT.localScale = new Vector3(1f, squash, 1f);
                go.transform.SetParent(squashT, false);
                var sr = go.AddComponent<SpriteRenderer>();
                sr.sprite = layers[i];
                sr.sharedMaterial = material;
                sr.sortingOrder = baseSortingOrder + k;
                _renderers[k++] = sr;
            }
        }
        Apply();
    }

    void LateUpdate() { Apply(); }

    void Apply()
    {
        if (_renderers == null) return;
        int n = layers.Length;
        for (int k = 0; k < _renderers.Length; k++)
        {
            int i = k / fillCopies;
            var sr = _renderers[k];
            sr.transform.localRotation = Quaternion.Euler(0f, 0f, yaw);
            sr.GetPropertyBlock(_mpb);
            _mpb.SetTexture("_NormalTex", normals[i]);
            _mpb.SetFloat("_Yaw", yaw);
            _mpb.SetFloat("_Squash", squash);
            _mpb.SetFloat("_Height01", (i + 0.5f) / n);
            sr.SetPropertyBlock(_mpb);
        }
    }
}
