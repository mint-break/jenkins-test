// Builds the character's sprite stack from several parts (Body, Leaves, ...) that
// share one layer grid, and keeps their lighting consistent while it turns.
// Built-in Render Pipeline, Unity 2020.3. Data: character_config.json ("draw" list).
//
// NOTE: written without access to a Unity editor in this environment, so it
// has not been compiled or run in Unity yet.
using System;
using UnityEngine;

public class SpriteStackCharacter : MonoBehaviour
{
    [Serializable]
    public class Part
    {
        public string name = "Body";
        [Tooltip("Albedo sprites of this part, in the order of layerIndex")]
        public Sprite[] layers;
        [Tooltip("Normal textures, same order")]
        public Texture2D[] normals;
        [Tooltip("Grid index of each layer, may be fractional (Body: 0..63, Leaves/Face: layer_index in their stack_config.json)")]
        public float[] layerIndex;
        [Tooltip("Grid units each layer covers: 1 for Body/Leaves, layer_span (1/8) for Face")]
        public float span = 1f;
        [Tooltip("Thin surfaces (leaves): light the side that faces the viewer")]
        public bool twoSided;
        [Range(0f, 1f)] public float glossScale = 1f;
        [Tooltip("Parts at the same grid index are drawn in ascending rank (Body 0, Leaves 1)")]
        public int rank;
        [Tooltip("Body only: interpolate between neighbouring layers (needs sdf, Body/SDF/Layer_###.png, "
               + "imported as Default, sRGB off, Single Channel R). Removes the horizontal banding.")]
        public bool interpolate;
        public Texture2D[] sdf;
    }

    public Part[] parts;
    public Material material;

    [Tooltip("layer_spacing_px from character_config.json")]
    public float layerSpacingPx = 6.2297f;
    [Tooltip("Pixels Per Unit used when importing the sprites")]
    public float pixelsPerUnit = 100f;
    [Tooltip("Body layer count (height01_divisor in character_config.json)")]
    public int height01Divisor = 64;
    [Range(0.05f, 1f)] public float squash = 0.75f;
    [Tooltip("copies_per_layer in character_config.json: a span-1 layer is drawn this many times. "
           + "Keep 8: the Face layers are aligned with these copies.")]
    public int copiesPerLayer = 8;
    [Tooltip("Yaw in degrees (counter-clockwise on screen)")]
    public float yaw;
    public int baseSortingOrder;

    struct Item { public SpriteRenderer sr; public Part part; public int idx; public float grid; }
    Item[] _items;
    Transform _root;
    MaterialPropertyBlock _mpb;

    void Start() { Build(); }

    public void Build()
    {
        if (_root != null) Destroy(_root.gameObject);
        _root = new GameObject("Stack").transform;
        _root.SetParent(transform, false);
        _mpb = new MaterialPropertyBlock();

        // every drawn copy of every part, sorted by height (ties: rank, Body first)
        var list = new System.Collections.Generic.List<(float h, int rank, Part part, int i)>();
        foreach (var p in parts)
        {
            int n = Mathf.Max(1, Mathf.RoundToInt(p.span * copiesPerLayer));
            for (int i = 0; i < p.layers.Length; i++)
                for (int c = 0; c < n; c++)
                    list.Add((Mathf.Round((p.layerIndex[i] + p.span * c / n) * 1e5f) / 1e5f, p.rank, p, i));
        }
        list.Sort((a, b) => a.h != b.h ? a.h.CompareTo(b.h) : a.rank.CompareTo(b.rank));

        float step = layerSpacingPx / pixelsPerUnit;
        var items = new System.Collections.Generic.List<Item>();
        int order = 0;
        foreach (var e in list)
        {
            var squashT = new GameObject("squash").transform;   // rotate (child), then squash (parent)
            squashT.SetParent(_root, false);
            squashT.localPosition = new Vector3(0f, e.h * step, 0f);
            squashT.localScale = new Vector3(1f, squash, 1f);
            var go = new GameObject($"{e.part.name}_{e.h:000.000}");
            go.transform.SetParent(squashT, false);
            var sr = go.AddComponent<SpriteRenderer>();
            sr.sprite = e.part.layers[e.i];
            sr.sharedMaterial = material;
            sr.sortingOrder = baseSortingOrder + order++;
            items.Add(new Item { sr = sr, part = e.part, idx = e.i, grid = e.h });
        }
        _items = items.ToArray();
        Apply();
    }

    void LateUpdate() { Apply(); }

    void Apply()
    {
        if (_items == null) return;
        foreach (var it in _items)
        {
            it.sr.transform.localRotation = Quaternion.Euler(0f, 0f, yaw);
            it.sr.GetPropertyBlock(_mpb);
            _mpb.SetTexture("_NormalTex", it.part.normals[it.idx]);
            _mpb.SetFloat("_Yaw", yaw);
            _mpb.SetFloat("_Squash", squash);
            _mpb.SetFloat("_Height01", (Mathf.Floor(it.grid) + 0.5f) / height01Divisor);
            _mpb.SetFloat("_TwoSided", it.part.twoSided ? 1f : 0f);
            if (it.part.interpolate && it.part.sdf != null && it.part.sdf.Length == it.part.layers.Length)
            {
                // copy bottom at grid height h; centres of layer j at j + 0.5 -> blend j and j + 1
                int n = it.part.layers.Length;
                float x = it.grid - it.part.layerIndex[0] + 0.5f / copiesPerLayer - 0.5f;
                int j = Mathf.FloorToInt(x);
                float t = x - j;
                if (j < 0) { j = 0; t = 0f; }
                if (j >= n - 1) { j = n - 1; t = 0f; }
                int j1 = Mathf.Min(j + 1, n - 1);
                it.sr.sprite = it.part.layers[j];
                _mpb.SetTexture("_NormalTex", it.part.normals[j]);
                _mpb.SetTexture("_MainTex1", it.part.layers[j1].texture);
                _mpb.SetTexture("_NormalTex1", it.part.normals[j1]);
                _mpb.SetTexture("_SdfTex", it.part.sdf[j]);
                _mpb.SetTexture("_SdfTex1", it.part.sdf[j1]);
                _mpb.SetFloat("_T", t);
                _mpb.SetFloat("_Interp", 1f);
            }
            else _mpb.SetFloat("_Interp", 0f);
            _mpb.SetFloat("_GlossScale", it.part.glossScale);
            it.sr.SetPropertyBlock(_mpb);
        }
    }
}
