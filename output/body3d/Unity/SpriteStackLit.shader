// Sprite-stack layer shader with a screen-fixed light (Built-in Render Pipeline, Unity 2020.3).
//
// Each layer is a SpriteRenderer using its Albedo PNG as the sprite and its
// Normal PNG as _NormalTex. The normal map is in LAYER space:
//   x = image right, y = image up (character's back), z = out of the layer (world up)
// The layer is rotated on screen by _Yaw (degrees, counter-clockwise), so the
// normal's xy is rotated by the same angle; the light stays fixed on screen.
// The maths mirrors tools/sprite_stack/body3d/lit.py (Python preview).
//
// NOTE: written without access to a Unity editor in this environment, so it
// has not been compiled or run in Unity yet.
Shader "SpriteStack/Lit"
{
    Properties
    {
        [PerRendererData] _MainTex ("Albedo (Sprite)", 2D) = "white" {}
        _NormalTex ("Normal (layer space)", 2D) = "bump" {}
        _ShadowColor ("Shadow", Color) = (0.631, 0.863, 0.706, 1)
        _BaseColor ("Base", Color) = (0.776, 0.988, 0.878, 1)
        _LightColor ("Light", Color) = (0.792, 0.992, 0.890, 1)
        _GlossColor ("Gloss", Color) = (0.980, 0.996, 0.988, 1)
        _LightDir ("Light dir (screen x, screen-plane y, up)", Vector) = (-0.45, -0.55, 0.70, 0)
        _Squash ("Camera squash (sin pitch)", Range(0.05, 1)) = 0.75
        _Yaw ("Stack yaw (deg, CCW)", Float) = 0
        _Height01 ("Layer height 0..1", Range(0, 1)) = 0.5
        _BaseFrom ("Shadow->Base from", Float) = 0.05
        _BaseTo ("Shadow->Base to", Float) = 0.22
        _LightFrom ("Base->Light from", Float) = 0.72
        _LightTo ("Base->Light to", Float) = 0.84
        _GlossFrom ("Gloss from", Float) = 0.955
        _GlossTo ("Gloss to", Float) = 0.99
        _GlossStrength ("Gloss strength", Range(0, 1)) = 0.6
        _GlossMinHeight ("Gloss min height", Range(0, 1)) = 0.55
        _GlossFullHeight ("Gloss full height", Range(0, 1)) = 0.65
        _RimStrength ("Rim darkening", Range(0, 1)) = 0.10
    }

    SubShader
    {
        Tags { "Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent"
               "PreviewType"="Plane" "CanUseSpriteAtlas"="True" }
        Cull Off
        Lighting Off
        ZWrite Off
        Blend One OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _NormalTex;
            fixed4 _ShadowColor, _BaseColor, _LightColor, _GlossColor;
            float4 _LightDir;
            float _Squash, _Yaw, _Height01;
            float _BaseFrom, _BaseTo, _LightFrom, _LightTo;
            float _GlossFrom, _GlossTo, _GlossStrength, _GlossMinHeight, _GlossFullHeight, _RimStrength;

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; fixed4 color : COLOR; };
            struct v2f { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; fixed4 color : COLOR; };

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                o.color = v.color;
                return o;
            }

            float3 tone(float3 n)
            {
                float3 L = normalize(_LightDir.xyz);
                float pitch = asin(saturate(_Squash));
                float3 V = float3(0, -cos(pitch), sin(pitch));
                float d = dot(n, L);
                float3 col = lerp(_ShadowColor.rgb, _BaseColor.rgb, smoothstep(_BaseFrom, _BaseTo, d));
                col = lerp(col, _LightColor.rgb, smoothstep(_LightFrom, _LightTo, d));
                float3 H = normalize(L + V);
                float g = smoothstep(_GlossFrom, _GlossTo, dot(n, H))
                        * smoothstep(_GlossMinHeight, _GlossFullHeight, _Height01);
                col = lerp(col, _GlossColor.rgb, _GlossStrength * g);
                float rim = smoothstep(0.35, 0.0, saturate(dot(n, V)));
                col *= 1.0 - _RimStrength * rim;
                return col / max(_BaseColor.rgb, 1e-3);      // multiplier relative to the base colour
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 albedo = tex2D(_MainTex, i.uv) * i.color;
                float3 n = tex2D(_NormalTex, i.uv).rgb * 2.0 - 1.0;   // import the normal PNG as a plain
                                                                       // texture (sRGB off), NOT as "Normal map"
                float a = radians(_Yaw);
                float c = cos(a), s = sin(a);
                n.xy = float2(n.x * c - n.y * s, n.x * s + n.y * c);
                n = normalize(n);
                fixed3 rgb = albedo.rgb * tone(n);
                return fixed4(rgb * albedo.a, albedo.a);              // premultiplied output
            }
            ENDCG
        }
    }
}
