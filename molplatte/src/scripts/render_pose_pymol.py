#!/usr/bin/env python
"""Ray-traced docked-pose image, rendered by PyMOL.

Runs in its OWN interpreter, not the training environment. PyMOL pulls Qt5,
X11 and pulseaudio, and conda would put its numpy 2.5.2 underneath a
pip-installed torch built against 2.4.4 -- so it lives in a separate env and is
invoked as a subprocess. `lead_report` falls back to its matplotlib renderer
when this is unavailable, which keeps PyMOL an optional dependency.

    python render_pose_pymol.py <receptor.pdb> <pose.sdf> <out.png> [caption]

Receptor is drawn as thin lines limited to residues near the ligand; the whole
protein would bury it. Ligand is sticks coloured by element.
"""
import sys

import pymol
from pymol import cmd

def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    receptor, pose, out = sys.argv[1:4]
    caption = sys.argv[4] if len(sys.argv) > 4 else ""

    pymol.finish_launching(["pymol", "-qc"])   # quiet, no GUI
    cmd.load(receptor, "rec")
    cmd.load(pose, "lig")

    cmd.hide("everything")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)

    # Pocket: residues with any atom within 5 A of the ligand.
    cmd.select("pocket", "byres (rec within 5 of lig)")
    cmd.show("lines", "pocket")
    cmd.color("grey70", "pocket")
    cmd.set("line_width", 1.4)
    # A translucent surface reads the shape of the cavity without hiding the pose.
    cmd.show("surface", "pocket")
    cmd.set("transparency", 0.72, "pocket")
    cmd.set("surface_quality", 1)

    cmd.show("sticks", "lig")
    cmd.color("grey25", "lig and elem C")
    cmd.color("firebrick", "lig and elem O")
    cmd.color("marine", "lig and elem N")
    cmd.color("yellow", "lig and elem S")
    cmd.set("stick_radius", 0.16, "lig")

    # Polar contacts are the reason a pose scores well; show them.
    cmd.distance("hbonds", "lig", "pocket", mode=2)
    cmd.color("black", "hbonds")
    cmd.set("dash_gap", 0.28)
    cmd.hide("labels", "hbonds")

    cmd.orient("lig")
    # A tight zoom clips the pocket surface at the frame edge and the pose
    # loses its context, which is the only reason to draw the surface at all.
    cmd.zoom("lig", 6.5)
    # Default clipping planes slice through the surface. Widen the slab so the
    # cavity is drawn whole rather than cut open.
    cmd.clip("slab", 90)
    cmd.set("field_of_view", 20)
    cmd.set("ray_trace_mode", 0)
    cmd.set("antialias", 2)
    cmd.set("depth_cue", 0)
    cmd.set("ray_shadows", 0)
    cmd.png(out, width=1300, height=1000, dpi=150, ray=1)
    cmd.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
