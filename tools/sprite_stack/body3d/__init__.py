"""Body-only 3D reconstruction from a 4-view sheet (front, side, back, top).

analyze.py : STEP 1  measure the body outline in every view (limbs, feet, tail,
             leaves removed) and report what was measured vs. estimated
model.py   : STEP 2  smooth 3D body (spline profiles + superellipse sections),
             implicit surface + triangle mesh (OBJ)
preview.py : STEP 3  orthographic front / side / back / top renders + overlays
"""
