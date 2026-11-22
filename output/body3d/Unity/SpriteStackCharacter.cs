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
        [Tooltip("Grid index of each layer (Body: 0..63, Leaves: see Leaves/stack_config.json layer_index)")]
        public int[] layerIndex;
        [Tooltip("Thin surfaces (leaves): light the side that faces the viewer")]
        public bool twoSided;
        [Range(0f, 1f)] public float glossScale = 1f;
        [Tooltip("Parts at the same grid index are drawn in ascending rank (Body 0, Leaves 1)")]
        public int rank;
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
    [Tooltip("Draw each layer this many times between grid steps so the stack has no gaps")]
    [Range(1, 12)] public int fillCopies = 7;
    [Tooltip("Yaw in degrees (counter-clockwise on screen)")]
    public float yaw;
    public int baseSortingOrder;

    struct Item { public SpriteRenderer sr; public Part part; public int idx; public int grid; }
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

        // collect (grid, rank, part, i) and sort into one draw order
        var list = new System.Collections.Generic.List<(int grid, int rank, Part part, int i)>();
        foreach (var p in parts)
            for (int i = 0; i < p.layers.Length; i++)
                list.Add((p.layerIndex[i], p.rank, p, i));
        list.Sort((a, b) => a.grid != b.grid ? a.grid.CompareTo(b.grid) : a.rank.CompareTo(b.rank));

        float step = layerSpacingPx / pixelsPerUnit;
        var items = new System.Collections.Generic.List<Item>();
        int order = 0;
        foreach (var e in list)
        {
            for (int c = 0; c < fillCopies; c++)
            {
                var squashT = new GameObject("squash").transform;   // rotate (child), then squash (parent)
                squashT.SetParent(_root, false);
                squashT.localPosition = new Vector3(0f, (e.grid + (float)c / fillCopies) * step, 0f);
                squashT.localScale = new Vector3(1f, squash, 1f);
                var go = new GameObject($"{e.part.name}_{e.grid:000}_{c}");
                go.transform.SetParent(squashT, false);
                var sr = go.AddComponent<SpriteRenderer>();
                sr.sprite = e.part.layers[e.i];
                sr.sharedMaterial = material;
                sr.sortingOrder = baseSortingOrder + order++;
                items.Add(new Item { sr = sr, part = e.part, idx = e.i, grid = e.grid });
            }
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
            _mpb.SetFloat("_Height01", (it.grid + 0.5f) / height01Divisor);
            _mpb.SetFloat("_TwoSided", it.part.twoSided ? 1f : 0f);
            _mpb.SetFloat("_GlossScale", it.part.glossScale);
            it.sr.SetPropertyBlock(_mpb);
        }
    }
}
