// Sprite-stack layer shader with a screen-fixed light (URP, Unity 6000.3).
//
// Each layer is a SpriteRenderer using its Albedo PNG as the sprite and its
// Normal PNG as _NormalTex. The normal map is in LAYER space:
//   x = image right, y = image up (character's back), z = out of the layer (world up)
// The layer is rotated on screen by _Yaw (degrees, counter-clockwise), so the
// normal's xy is rotated by the same angle; the light stays fixed on screen.
// The maths mirrors tools/sprite_stack/body3d/lit.py (Python preview).
//
// Ported from the original Built-in (CGPROGRAM) version:
//  - HLSLPROGRAM + URP Core.hlsl, SubShader tagged for the UniversalPipeline
//  - the pass has no LightMode tag (= SRPDefaultUnlit), so both the Universal
//    and the 2D renderer draw it in the transparent queue
//  - fix: the interpolated body path divided the (straight-alpha) albedo by
//    its alpha, which blew layer edges up to white (white rings / washed-out
//    lower body). Albedo PNGs are straight alpha; their rgb is used as is.
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
        [Toggle] _TwoSided ("Two-sided (thin parts: leaves)", Float) = 0
        _GlossScale ("Gloss scale (per part)", Range(0, 1)) = 1
        [Toggle] _Interp ("Interpolate with the next layer (Body: SDF + normals)", Float) = 0
        _MainTex1 ("Next layer albedo", 2D) = "black" {}
        _NormalTex1 ("Next layer normal", 2D) = "black" {}
        _SdfTex ("This layer SDF", 2D) = "black" {}
        _SdfTex1 ("Next layer SDF", 2D) = "black" {}
        _T ("Blend towards the next layer 0..1", Range(0, 1)) = 0
    }

    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" "Queue"="Transparent" "IgnoreProjector"="True"
               "RenderType"="Transparent" "PreviewType"="Plane" "CanUseSpriteAtlas"="True" }
        Cull Off
        ZWrite Off
        Blend One OneMinusSrcAlpha

        Pass
        {
            Name "SpriteStackLit"

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_MainTex);    SAMPLER(sampler_MainTex);
            TEXTURE2D(_NormalTex);  SAMPLER(sampler_NormalTex);
            TEXTURE2D(_MainTex1);   SAMPLER(sampler_MainTex1);
            TEXTURE2D(_NormalTex1); SAMPLER(sampler_NormalTex1);
            TEXTURE2D(_SdfTex);     SAMPLER(sampler_SdfTex);
            TEXTURE2D(_SdfTex1);    SAMPLER(sampler_SdfTex1);

            CBUFFER_START(UnityPerMaterial)
                float4 _MainTex_ST;
                half4 _ShadowColor, _BaseColor, _LightColor, _GlossColor;
                float4 _LightDir;
                float _Squash, _Yaw, _Height01;
                float _BaseFrom, _BaseTo, _LightFrom, _LightTo;
                float _GlossFrom, _GlossTo, _GlossStrength, _GlossMinHeight, _GlossFullHeight, _RimStrength;
                float _TwoSided, _GlossScale;
                float _Interp, _T;
            CBUFFER_END

            struct Attributes { float4 positionOS : POSITION; float2 uv : TEXCOORD0; half4 color : COLOR; };
            struct Varyings   { float4 positionCS : SV_POSITION; float2 uv : TEXCOORD0; half4 color : COLOR; };

            Varyings vert (Attributes v)
            {
                Varyings o;
                o.positionCS = TransformObjectToHClip(v.positionOS.xyz);
                o.uv = v.uv;
                o.color = v.color;
                return o;
            }

            float3 tone(float3 n)
            {
                float3 L = normalize(_LightDir.xyz);
                float pitch = asin(saturate(_Squash));
                float3 V = float3(0, -cos(pitch), sin(pitch));
                if (_TwoSided > 0.5 && dot(n, V) < 0) n = -n;     // light thin parts from the visible side
                float d = dot(n, L);
                float3 col = lerp(_ShadowColor.rgb, _BaseColor.rgb, smoothstep(_BaseFrom, _BaseTo, d));
                col = lerp(col, _LightColor.rgb, smoothstep(_LightFrom, _LightTo, d));
                float3 H = normalize(L + V);
                float g = smoothstep(_GlossFrom, _GlossTo, dot(n, H))
                        * smoothstep(_GlossMinHeight, _GlossFullHeight, _Height01);
                col = lerp(col, _GlossColor.rgb, _GlossStrength * _GlossScale * g);
                float rim = smoothstep(0.35, 0.0, saturate(dot(n, V)));
                col *= 1.0 - _RimStrength * rim;
                return col / max(_BaseColor.rgb, 1e-3);      // multiplier relative to the base colour
            }

            half4 frag (Varyings i) : SV_Target
            {
                half4 albedo = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv) * i.color;
                // import the normal PNG as a plain texture (sRGB off), NOT as "Normal map"
                float3 n = SAMPLE_TEXTURE2D(_NormalTex, sampler_NormalTex, i.uv).rgb * 2.0 - 1.0;
                if (_Interp > 0.5)
                {
                    // continuous stack: blend this layer with the next one. SDF (8-bit,
                    // sqrt-companded: d = sign(q) * 48 * q^2, q = (v - 128) / 127, px inside
                    // the edge) is interpolated, then thresholded, so the outline is the
                    // real cross-section between the two layers.
                    float q0 = (SAMPLE_TEXTURE2D(_SdfTex,  sampler_SdfTex,  i.uv).r * 255.0 - 128.0) / 127.0;
                    float q1 = (SAMPLE_TEXTURE2D(_SdfTex1, sampler_SdfTex1, i.uv).r * 255.0 - 128.0) / 127.0;
                    float d0 = sign(q0) * 48.0 * q0 * q0;
                    float d1 = sign(q1) * 48.0 * q1 * q1;
                    float cov = saturate(0.5 + lerp(d0, d1, _T));
                    clip(cov - 1e-4);
                    float3 n1 = SAMPLE_TEXTURE2D(_NormalTex1, sampler_NormalTex1, i.uv).rgb * 2.0 - 1.0;
                    bool h0 = dot(n, n) > 0.25, h1 = dot(n1, n1) > 0.25;
                    if (!h0) n = n1;
                    if (!h1) n1 = n;
                    n = lerp(n, n1, _T);
                    half4 albedo1 = SAMPLE_TEXTURE2D(_MainTex1, sampler_MainTex1, i.uv);
                    // straight alpha: take the colour of whichever layer covers the pixel more
                    half3 c0 = albedo.a >= albedo1.a ? albedo.rgb : albedo1.rgb * i.color.rgb;
                    albedo = half4(c0, cov * i.color.a);
                }
                float a = radians(_Yaw);
                float c = cos(a), s = sin(a);
                n.xy = float2(n.x * c - n.y * s, n.x * s + n.y * c);
                n = normalize(n);
                half3 rgb = albedo.rgb * tone(n);
                return half4(rgb * albedo.a, albedo.a);       // premultiplied output
            }
            ENDHLSL
        }
    }
}
