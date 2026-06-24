#!/usr/bin/env python3
"""
visualize_geometry.py
======================
Visualizes the CURRENT (post-fix) SuperK-style flush-PMT geometry:

  - Acrylic vessel: 5mm PMMA shell, 6 through-holes (one per axis)
  - Detection medium: sphere with 6 sagitta-matched flat notches
  - PMT glass: flush with medium at r = R_INNER (zero gap, zero overlap)
  - Photocathode + body: extend outward through the acrylic hole

All dimensions match DetectorConstruction.cc exactly, including the
sagitta correction that fixed the original zero-detection bug.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Wedge
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings, os
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "figure.dpi":150,"savefig.dpi":200,"savefig.bbox":"tight"})

# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY — must match DetectorConstruction.cc exactly
# https://www.hamamatsu.com/us/en/product/optical-sensors/pmt/pmt_tube-alone/head-on-type/R1307.html
# ══════════════════════════════════════════════════════════════════════════════
R_INNER   = 150.0   # mm, medium sphere / vessel inner radius
WALL      = 5.0     # mm, acrylic wall thickness
R_OUTER   = R_INNER + WALL   # 155.0 mm
TUBE_R    = 38.0    # mm, R1307 tube OD/2 (also the hole radius)
PC_R      = 25.75   # mm, R1307 active photocathode radius
T_GLASS   = 2.0     # mm, borosilicate window thickness
T_PC      = 0.5     # mm, photocathode disc thickness
L_BODY    = 147.5   # mm, PMT body length
L_TOTAL   = 150.0   # mm, total PMT length

# Sagitta — how far a flat disc of radius TUBE_R is recessed from a sphere
# of radius R_INNER. This is the quantity that caused the original bug,
# and the notch depth below is sized specifically to remove it.
SAGITTA     = R_INNER - np.sqrt(R_INNER**2 - TUBE_R**2)   # ≈ 4.893 mm
NOTCH_DEPTH = SAGITTA + 5.0                                # ≈ 9.893 mm, as coded

# Key axial positions (all flush at R_INNER by construction)
GLASS_FRONT  = R_INNER                          # exact flush face
GLASS_CENTRE = R_INNER + T_GLASS/2
PC_CENTRE    = R_INNER + T_GLASS + T_PC/2
BODY_CENTRE  = R_INNER + T_GLASS + T_PC + L_BODY/2
HOLE_CENTRE  = R_OUTER                          # acrylic hole centred on shell
HOLE_LEN     = WALL + 15.0                       # through-hole with margin

AXIS_LABELS = ["+x","−x","+y","−y","+z","−z"]
PMT_FACE_COLORS = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261","#a8dadc"]

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
#  Track / photon generators (unchanged physics, just for illustration)
# ─────────────────────────────────────────────────────────────────────────────
def electron_track(E_MeV=5.0, n_steps=120):
    pos=np.zeros(3); pts=[pos.copy()]
    direction=np.array([0.55,0.25,0.80]); direction/=np.linalg.norm(direction)
    step=0.20; scat=0.12/max(E_MeV,0.5); rng=0.412*E_MeV**1.27; cum=0.
    for _ in range(n_steps):
        if cum>rng: break
        p1=np.cross(direction,[0,1,0])
        p1=p1/np.linalg.norm(p1) if np.linalg.norm(p1)>1e-6 else np.cross(direction,[1,0,0])
        p1/=np.linalg.norm(p1); p2=np.cross(direction,p1)
        direction+=scat*(np.random.randn()*p1+np.random.randn()*p2)
        direction/=np.linalg.norm(direction)
        pos=pos+step*direction; pts.append(pos.copy()); cum+=step
        if np.linalg.norm(pos)>R_INNER*0.1-0.5: break
    return np.array(pts)

def cherenkov_photons(track,n=22,theta_c=41.2):
    photons=[]; tc=np.radians(theta_c); step=max(1,len(track)//n)
    for i in range(0,len(track)-step,step):
        d=track[i+1]-track[i]; d/=(np.linalg.norm(d)+1e-12)
        phi=np.random.uniform(0,2*np.pi)
        p=np.cross(d,[0,0,1])
        p=(p/np.linalg.norm(p)) if np.linalg.norm(p)>1e-6 else np.cross(d,[1,0,0]); p/=np.linalg.norm(p)
        v=np.cross(d,p)
        pd=np.cos(tc)*d+np.sin(tc)*(np.cos(phi)*p+np.sin(phi)*v); pd/=np.linalg.norm(pd)
        o=track[i]; a=1.; b=2*np.dot(o,pd); c=np.dot(o,o)-R_INNER**2
        disc=b*b-4*a*c
        if disc<0: continue
        t=(-b+np.sqrt(disc))/(2*a)
        if t<0: continue
        photons.append((o,o+t*pd))
    return photons

def scint_photons(track,n=20):
    photons=[]; idxs=np.random.choice(len(track),min(n,len(track)),replace=False)
    for i in idxs:
        o=track[i]; d=np.random.randn(3); d/=np.linalg.norm(d)
        a=1.; b=2*np.dot(o,d); c=np.dot(o,o)-R_INNER**2
        disc=b*b-4*a*c
        if disc<0: continue
        t=(-b+np.sqrt(disc))/(2*a)
        if t<0: continue
        photons.append((o,o+t*d))
    return photons

def wls_photons(sci,n=7):
    wls=[]; ch=np.random.choice(len(sci),min(n,len(sci)),replace=False)
    for i in ch:
        o,e=sci[i]; f=np.random.uniform(0.3,0.6); ab=o+f*(e-o)
        d=np.random.randn(3); d/=np.linalg.norm(d)
        a=1.; b=2*np.dot(ab,d); c=np.dot(ab,ab)-R_INNER**2
        disc=b*b-4*a*c
        if disc<0: continue
        t=(-b+np.sqrt(disc))/(2*a)
        if t<0: continue
        wls.append((o,ab,ab+t*d))
    return wls

# ─────────────────────────────────────────────────────────────────────────────
#  3D geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def disc_verts(centre,normal,radius,n=40):
    n_vec=np.array(normal)/np.linalg.norm(normal)
    u=np.cross(n_vec,[1,0,0])
    u=(u/np.linalg.norm(u)) if np.linalg.norm(u)>1e-6 else np.cross(n_vec,[0,1,0])
    u/=np.linalg.norm(u); v=np.cross(n_vec,u)
    theta=np.linspace(0,2*np.pi,n)
    pts=np.array(centre)+radius*(np.outer(np.cos(theta),u)+np.outer(np.sin(theta),v))
    return [list(map(tuple,pts))]

def cylinder_surf(centre,axis,radius,length,n=20):
    ax=np.array(axis)/np.linalg.norm(axis)
    u=np.cross(ax,[1,0,0])
    u=(u/np.linalg.norm(u)) if np.linalg.norm(u)>1e-6 else np.cross(ax,[0,1,0])
    u/=np.linalg.norm(u); v=np.cross(ax,u)
    theta=np.linspace(0,2*np.pi,n,endpoint=True)
    c=np.array(centre); h=ax*length/2; faces=[]
    for i in range(n-1):
        p0=c-h+radius*(np.cos(theta[i])*u+np.sin(theta[i])*v)
        p1=c-h+radius*(np.cos(theta[i+1])*u+np.sin(theta[i+1])*v)
        p2=c+h+radius*(np.cos(theta[i+1])*u+np.sin(theta[i+1])*v)
        p3=c+h+radius*(np.cos(theta[i])*u+np.sin(theta[i])*v)
        faces.append([tuple(p0),tuple(p1),tuple(p2),tuple(p3)])
    return faces

def sphere_with_holes_wire(ax,R,hole_dirs,hole_r,n=24,color="steelblue",alpha=0.15,lw=0.5):
    """Wireframe sphere with circular gaps where holes are cut."""
    u=np.linspace(0,np.pi,80)
    for phi in np.linspace(0,2*np.pi,n,endpoint=False):
        x=R*np.sin(u)*np.cos(phi); y=R*np.sin(u)*np.sin(phi); z=R*np.cos(u)
        pts=np.stack([x,y,z],axis=1)
        mask=np.ones(len(pts),dtype=bool)
        for hd in hole_dirs:
            hd=np.array(hd)
            ang=np.arccos(np.clip(pts@hd/R,-1,1))
            local_r=R*np.sin(ang)
            close_to_axis = (ang<np.pi/2) & (local_r<hole_r*1.3)
            mask &= ~close_to_axis
        xm=np.where(mask,x,np.nan)
        ax.plot(xm,np.where(mask,y,np.nan),np.where(mask,z,np.nan),
                color=color,alpha=alpha,lw=lw)

def hole_rim_circle(ax,centre_dir,R,hole_r,color="gold",lw=1.2,alpha=0.8):
    """Draw the rim circle where a hole meets the sphere surface."""
    d=np.array(centre_dir,dtype=float); d/=np.linalg.norm(d)
    u=np.cross(d,[1,0,0]); u=(u/np.linalg.norm(u)) if np.linalg.norm(u)>1e-6 else np.cross(d,[0,1,0])
    u/=np.linalg.norm(u); v=np.cross(d,u)
    theta=np.linspace(0,2*np.pi,60)
    centre=d*R
    pts=centre+hole_r*(np.outer(np.cos(theta),u)+np.outer(np.sin(theta),v))
    ax.plot(pts[:,0],pts[:,1],pts[:,2],color=color,lw=lw,alpha=alpha)

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — 3D view of the corrected SuperK-style geometry
# ══════════════════════════════════════════════════════════════════════════════
def plot_3d_geometry(out_dir):
    fig=plt.figure(figsize=(18,14)); fig.patch.set_facecolor("#080c10")
    ax=fig.add_axes([0.01,0.12,0.60,0.85],projection='3d')
    ax.set_facecolor("#080c10")
    ax.set_xlim(-280,280); ax.set_ylim(-280,280); ax.set_zlim(-280,280)
    ax.set_xlabel("x [mm]",color="white",labelpad=6)
    ax.set_ylabel("y [mm]",color="white",labelpad=6)
    ax.set_zlabel("z [mm]",color="white",labelpad=6)
    ax.tick_params(colors="white")
    for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        pane.fill=False; pane.set_edgecolor("#1a1a2e")
    ax.grid(True,color="#111122",alpha=0.4)

    axis_dirs=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    # Acrylic outer shell, semi-transparent, with holes implied by gaps
    sphere_with_holes_wire(ax,R_OUTER,axis_dirs,TUBE_R,n=28,color="#88ccff",alpha=0.14,lw=0.5)
    # Medium sphere (now with sagitta-matched flat notches, gaps shown the same way)
    sphere_with_holes_wire(ax,R_INNER,axis_dirs,TUBE_R,n=28,color="#4488ff",alpha=0.22,lw=0.6)

    # Faint solid acrylic shell surface
    u=np.linspace(0,2*np.pi,40); v=np.linspace(0,np.pi,20)
    xs=R_OUTER*np.outer(np.cos(u),np.sin(v)); ys=R_OUTER*np.outer(np.sin(u),np.sin(v))
    zs=R_OUTER*np.outer(np.ones_like(u),np.cos(v))
    ax.plot_surface(xs,ys,zs,alpha=0.035,color="#88ccff",shade=False,linewidth=0)

    for idx,outward in enumerate(axis_dirs):
        outward=np.array(outward,dtype=float)
        inward=-outward
        col=PMT_FACE_COLORS[idx]

        # Rim circle at R_INNER — shows where the flat notch face meets the medium
        hole_rim_circle(ax,outward,R_INNER,TUBE_R,color="#66ddff",lw=1.0,alpha=0.55)
        # Rim circle at R_OUTER — shows the acrylic through-hole
        hole_rim_circle(ax,outward,R_OUTER,TUBE_R,color="#aaccee",lw=1.0,alpha=0.4)

        glass_c=outward*GLASS_CENTRE
        pc_c   =outward*PC_CENTRE
        body_c =outward*BODY_CENTRE

        # Glass window — front face exactly at R_INNER (flush, zero-gap)
        gv=disc_verts(glass_c,outward,TUBE_R)
        ax.add_collection3d(Poly3DCollection(gv,alpha=0.55,facecolor="#aaddff",
                                              edgecolor="#88bbdd",lw=0.8))

        # Photocathode — active area, gold
        pv=disc_verts(pc_c,outward,PC_R)
        ax.add_collection3d(Poly3DCollection(pv,alpha=0.92,facecolor=col,
                                              edgecolor="gold",lw=1.2))

        # PMT body cylinder
        faces=cylinder_surf(body_c,outward,TUBE_R,L_BODY,n=16)
        ax.add_collection3d(Poly3DCollection(faces,alpha=0.45,facecolor="#333344",
                                              edgecolor="#222233",lw=0.3))

        # Inward arrow showing photon direction at the flush boundary
        tip=outward*(R_INNER-18)
        ax.quiver(*tip,*(inward*22),color="gold",alpha=0.85,lw=1.6,
                  arrow_length_ratio=0.35,length=1)

        lbl=outward*(R_INNER+L_TOTAL+18)
        ax.text(*lbl,f"R1307\n{AXIS_LABELS[idx]}",color=col,fontsize=7.5,
                ha='center',va='center',fontweight='bold')

    # Electron track + photons
    track=electron_track(5.0,120)
    cer=cherenkov_photons(track,22); sci=scint_photons(track,25); wls=wls_photons(sci,8)
    ax.plot(track[:,0],track[:,1],track[:,2],color="#00ff88",lw=2.5,alpha=0.95,zorder=10,label="e⁻ 5 MeV")
    ax.scatter(0,0,0,color="lime",s=90,zorder=15,edgecolors="white",lw=1.2)
    for i,(o,e) in enumerate(cer):
        ax.plot([o[0],e[0]],[o[1],e[1]],[o[2],e[2]],color="#60c0ff",alpha=0.40,lw=0.8,
                label="Cherenkov" if i==0 else "")
    for i,(o,e) in enumerate(sci):
        ax.plot([o[0],e[0]],[o[1],e[1]],[o[2],e[2]],color="#ffaa00",alpha=0.30,lw=0.7,ls="--",
                label="PPO scint." if i==0 else "")
    for i,(_,ab,ne) in enumerate(wls):
        ax.plot([ab[0],ne[0]],[ab[1],ne[1]],[ab[2],ne[2]],color="#da70d6",alpha=0.60,lw=1.0,
                label="bis-MSB WLS" if i==0 else "")
        ax.scatter(*ab,color="violet",s=15,alpha=0.8,zorder=8)

    for vec,col,lbl in [([1,0,0],"#ff4444","X"),([0,1,0],"#44ff44","Y"),([0,0,1],"#4499ff","Z")]:
        L=R_INNER*1.25
        ax.quiver(0,0,0,vec[0]*L,vec[1]*L,vec[2]*L,color=col,lw=2,arrow_length_ratio=0.07)
        ax.text(vec[0]*(L+15),vec[1]*(L+15),vec[2]*(L+15),lbl,color=col,fontsize=11,fontweight='bold')

    ax.set_title("NeutrinoDetSim — SuperK-Style Flush PMT Geometry (post-fix)\n"
                 "6 holes through acrylic · medium notched to match glass (sagitta-corrected) · 5 MeV e⁻",
                 color="white",fontsize=11,pad=8)

    handles=[Line2D([0],[0],color="#00ff88",lw=2,label="e⁻ track (5 MeV)"),
             Line2D([0],[0],color="#60c0ff",lw=1,label="Cherenkov photons"),
             Line2D([0],[0],color="#ffaa00",lw=1,ls="--",label="PPO scintillation"),
             Line2D([0],[0],color="#da70d6",lw=1,label="bis-MSB WLS re-emit"),
             Line2D([0],[0],color="#aaddff",lw=4,label="R1307 borosilicate glass (flush @ R_in)"),
             Line2D([0],[0],color="gold",lw=4,label="R1307 photocathode (Ø51.5mm)"),
             Line2D([0],[0],color="#66ddff",lw=1.5,label="Notch rim @ R_in (sagitta-matched)"),
             Line2D([0],[0],color="#88ccff",lw=2,alpha=0.5,label="Acrylic hole rim @ R_out")]
    ax.legend(handles=handles,loc="lower left",facecolor="#111111",edgecolor="#444444",
              labelcolor="white",fontsize=7.3,framealpha=0.85)
    ax.view_init(elev=20,azim=-50)

    # ── Right-side inset: zoomed single-PMT junction (the fix, illustrated) ────
    ax2=fig.add_axes([0.62,0.52,0.36,0.44])
    ax2.set_facecolor("#080c10"); ax2.set_aspect("equal")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#333344")
    ax2.set_title("Zoom: PMT Junction at +x  (the sagitta fix)",color="white",fontsize=10)
    ax2.set_xlabel("radial position [mm]",color="white",fontsize=9)
    ax2.set_ylabel("transverse [mm]",color="white",fontsize=9)
    ax2.grid(True,color="#111122",alpha=0.4)

    xr=np.linspace(R_INNER-25,R_OUTER+25,400)
    # True sphere profile (for reference, dashed)
    yr_sphere=np.sqrt(np.maximum(R_INNER**2-xr**2,0)) if False else None

    # Draw acrylic block with hole
    ax2.add_patch(Rectangle((R_INNER,-TUBE_R-15),WALL,2*(TUBE_R+15),
                             facecolor="#1a2535",edgecolor="#334455",lw=0.8,zorder=0))
    ax2.add_patch(Rectangle((R_INNER,-TUBE_R),WALL,2*TUBE_R,
                             facecolor="none",edgecolor="#88ccff",lw=1.5,ls="--",zorder=1))
    ax2.text(R_INNER+WALL/2,TUBE_R+18,"acrylic\n(hole cut through)",
             ha='center',color="#88ccff",fontsize=7.5)

    # Sagitta illustration: the spherical cap that WAS overlapping, now removed
    theta=np.linspace(-np.arcsin(TUBE_R/R_INNER),np.arcsin(TUBE_R/R_INNER),100)
    cap_x=R_INNER*np.cos(theta); cap_y=R_INNER*np.sin(theta)
    ax2.plot(cap_x,cap_y,color="#ff5566",lw=1.8,ls=":",alpha=0.85,
             label=f"old spherical surface\n(removed cap, sagitta={SAGITTA:.2f}mm)")
    ax2.fill_betweenx(cap_y,cap_x,R_INNER,where=(cap_x>=R_INNER-0.01),
                       color="#ff5566",alpha=0.0)
    # Shade the removed cap region
    notch_x = np.full_like(theta, R_INNER - NOTCH_DEPTH)
    ax2.fill_betweenx(np.clip(cap_y,-TUBE_R,TUBE_R), notch_x, cap_x,
                       color="#ff5566",alpha=0.12,label="notch removed (old overlap zone)")

    # New flat medium face at R_INNER
    ax2.plot([R_INNER,R_INNER],[-TUBE_R,TUBE_R],color="#4488ff",lw=2.5,zorder=5,
              label="medium flat face (new, r=R_in exactly)")

    # Glass disc, flush
    ax2.add_patch(Rectangle((R_INNER,-TUBE_R),T_GLASS,2*TUBE_R,
                             facecolor="#aaddff",edgecolor="#88bbdd",lw=1.2,alpha=0.85,zorder=6))
    ax2.text(R_INNER+T_GLASS/2,-TUBE_R-10,"glass\n(flush, zero gap)",
             ha='center',va='top',color="#aaddff",fontsize=7.5)

    # Photocathode strip
    ax2.add_patch(Rectangle((R_INNER+T_GLASS,-PC_R),T_PC,2*PC_R,
                             facecolor="gold",edgecolor="orange",lw=1.2,zorder=7))

    ax2.axvline(R_INNER,color="white",lw=0.8,ls=":",alpha=0.5)
    ax2.text(R_INNER,TUBE_R+28,f"r = R_in = {R_INNER:.0f} mm",ha='center',color="white",fontsize=8)

    ax2.set_xlim(R_INNER-22,R_OUTER+12)
    ax2.set_ylim(-(TUBE_R+25),(TUBE_R+35))
    ax2.legend(fontsize=7,loc="lower right",facecolor="#111111",edgecolor="#444444",labelcolor="white")

    # ── Bottom-right: dimension/parameter summary ──────────────────────────────
    ax3=fig.add_axes([0.62,0.06,0.36,0.40])
    ax3.axis("off"); ax3.set_facecolor("#080c10")
    txt = (
        "SuperK-Style Flush PMT — Fix Summary\n"
        "──────────────────────────────────────\n"
        f"R_inner (medium/vessel)   = {R_INNER:.1f} mm\n"
        f"R_outer (acrylic)         = {R_OUTER:.1f} mm\n"
        f"Wall thickness            = {WALL:.1f} mm\n"
        f"PMT tube radius (hole r)  = {TUBE_R:.1f} mm\n"
        f"Photocathode radius       = {PC_R:.2f} mm\n\n"
        "Bug (before fix):\n"
        f"  Medium = perfect sphere, R={R_INNER:.0f}mm\n"
        f"  Glass  = flat disc, flush at R={R_INNER:.0f}mm\n"
        f"  -> flat disc recessed from sphere by sagitta:\n"
        f"     s = R - sqrt(R²-r²) = {SAGITTA:.3f} mm\n"
        f"  -> medium overlapped glass by up to {SAGITTA:.2f}mm\n"
        f"     at the disc rim -> ambiguous boundary ->\n"
        f"     zero photon detections\n\n"
        "Fix (this geometry):\n"
        f"  Notch depth = sagitta + margin = {NOTCH_DEPTH:.3f} mm\n"
        f"  Notch cutting face lands at EXACTLY r={R_INNER:.0f}mm\n"
        f"  Glass front face also at EXACTLY r={R_INNER:.0f}mm\n"
        f"  -> two coincident flat planes, zero gap,\n"
        f"     zero overlap, single clean boundary\n\n"
        f"Geometric coverage  = "
        f"{6*np.pi*PC_R**2/(4*np.pi*R_INNER**2)*100:.2f} %\n"
        f"Optical path: medium -> glass (Fresnel ~0.4%)\n"
        f"              -> photocathode -> SD\n"
    )
    ax3.text(0.02,0.98,txt,transform=ax3.transAxes,color="lightgrey",fontsize=8.3,
             va='top',ha='left',family='monospace',
             bbox=dict(facecolor="#0d1118",edgecolor="#333344",boxstyle="round,pad=0.6",alpha=0.92))

    os.makedirs(out_dir,exist_ok=True)
    fname=os.path.join(out_dir,"geo_3d_detector_R1307_fixed.png")
    fig.savefig(fname,facecolor=fig.get_facecolor(),dpi=200)
    plt.close(fig)
    print(f"  [✓] 3D geometry (fixed) → {fname}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Cross-sections showing holes correctly
# ══════════════════════════════════════════════════════════════════════════════
def plot_cross_sections(out_dir):
    fig,axes=plt.subplots(1,3,figsize=(16,6)); fig.patch.set_facecolor("#080c10")
    titles=["XY Plane (top view)","XZ Plane (side view)","YZ Plane (front view)"]
    plane_idx=[(0,1),(0,2),(1,2)]
    plane_pmts=[
        [(1,0,0,"#e63946","+x"),(-1,0,0,"#457b9d","−x"),(0,1,0,"#2a9d8f","+y"),(0,-1,0,"#e9c46a","−y")],
        [(1,0,0,"#e63946","+x"),(-1,0,0,"#457b9d","−x"),(0,0,1,"#a8dadc","+z"),(0,0,-1,"#f4a261","−z")],
        [(0,1,0,"#2a9d8f","+y"),(0,-1,0,"#e9c46a","−y"),(0,0,1,"#a8dadc","+z"),(0,0,-1,"#f4a261","−z")],
    ]
    al=["x [mm]","y [mm]","z [mm]"]
    track=electron_track(5.0,120); cer=cherenkov_photons(track,20)
    sci=scint_photons(track,18); wls=wls_photons(sci,6)

    for ci,(ax,title,(d0,d1),pmts) in enumerate(zip(axes,titles,plane_idx,plane_pmts)):
        ax.set_facecolor("#080c10"); ax.set_aspect("equal")
        ax.set_xlim(-290,290); ax.set_ylim(-290,290)
        ax.tick_params(colors="white"); ax.spines[:].set_color("#333344")
        ax.set_title(title,color="white",fontsize=10)
        ax.set_xlabel(al[d0],color="white"); ax.set_ylabel(al[d1],color="white")
        ax.grid(True,color="#111122",alpha=0.4)
        th=np.linspace(0,2*np.pi,300)

        # Acrylic ring (outer/inner) — will show gaps at hole positions
        ax.plot(R_OUTER*np.cos(th),R_OUTER*np.sin(th),color="#88ccff",lw=2.0,alpha=0.7)
        ax.fill_between(R_OUTER*np.cos(th),R_OUTER*np.sin(th),
                        R_INNER*np.sin(th),alpha=0.10,color="#88ccff")
        ax.plot(R_INNER*np.cos(th),R_INNER*np.sin(th),color="#4488ff",lw=1.3,alpha=0.5,ls=":")
        ax.fill(R_INNER*np.cos(th),R_INNER*np.sin(th),alpha=0.05,color="#4488ff")

        for px,py,pz,col,lbl in pmts:
            full=np.array([px,py,pz],dtype=float)
            nd2=full[[d0,d1]]; nm=np.linalg.norm(nd2)
            nd2=nd2/nm if nm>0 else nd2
            perp=np.array([-nd2[1],nd2[0]])

            # Acrylic hole gap (erase the ring locally)
            gap_centre=nd2*R_OUTER
            for frac in np.linspace(-1,1,2):
                pass  # visual gap handled by overdrawing background colour below
            gap_pts=np.array([nd2*(R_INNER-2)+TUBE_R*perp, nd2*(R_OUTER+2)+TUBE_R*perp,
                              nd2*(R_OUTER+2)-TUBE_R*perp, nd2*(R_INNER-2)-TUBE_R*perp])
            ax.fill(gap_pts[:,0],gap_pts[:,1],color="#080c10",zorder=2)

            # Glass — flush exactly at R_INNER
            gc=nd2*(R_INNER+T_GLASS/2)
            gpts=np.array([gc+TUBE_R*perp,gc-TUBE_R*perp,
                           gc-TUBE_R*perp+nd2*T_GLASS,gc+TUBE_R*perp+nd2*T_GLASS])
            ax.fill(gpts[:,0],gpts[:,1],color="#aaddff",alpha=0.55,zorder=3)
            ax.plot([nd2[0]*R_INNER-perp[0]*TUBE_R, nd2[0]*R_INNER+perp[0]*TUBE_R],
                    [nd2[1]*R_INNER-perp[1]*TUBE_R, nd2[1]*R_INNER+perp[1]*TUBE_R],
                    color="#66ddff",lw=1.3,alpha=0.8,zorder=4)  # flush boundary line

            # Photocathode
            pcc=nd2*(R_INNER+T_GLASS+T_PC/2)
            p1=pcc+PC_R*perp; p2=pcc-PC_R*perp
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=col,lw=5,alpha=0.92,
                    solid_capstyle='round',zorder=5)

            # Body
            bc=nd2*(R_INNER+T_GLASS+T_PC+L_BODY/2)
            bpts=np.array([bc+TUBE_R*perp,bc-TUBE_R*perp,
                           bc-TUBE_R*perp+nd2*L_BODY/2,bc+TUBE_R*perp+nd2*L_BODY/2])
            ax.fill(bpts[:,0],bpts[:,1],color="#222233",alpha=0.65,zorder=3)

            ax.text(nd2[0]*(R_INNER+L_TOTAL+22),nd2[1]*(R_INNER+L_TOTAL+22),
                    f"{lbl}",color=col,ha='center',va='center',fontsize=8,fontweight='bold')
            ax.annotate("",xy=nd2*(R_INNER-15),xytext=nd2*(R_INNER+T_GLASS),
                       arrowprops=dict(arrowstyle="->",color="gold",lw=1.5))

        ax.plot(track[:,d0],track[:,d1],color="#00ff88",lw=2.0,alpha=0.9)
        ax.scatter(0,0,color="lime",s=50,zorder=10)
        for o,e in cer: ax.plot([o[d0],e[d0]],[o[d1],e[d1]],color="#60c0ff",alpha=0.35,lw=0.8)
        for o,e in sci[:10]: ax.plot([o[d0],e[d0]],[o[d1],e[d1]],color="#ffaa00",alpha=0.22,lw=0.6,ls="--")
        for _,ab,ne in wls: ax.plot([ab[d0],ne[d0]],[ab[d1],ne[d1]],color="#da70d6",alpha=0.55,lw=0.9)

        ax.annotate("",xy=(R_OUTER,-R_OUTER*1.20),xytext=(-R_OUTER,-R_OUTER*1.20),
                   arrowprops=dict(arrowstyle="<->",color="white",lw=1.2))
        ax.text(0,-R_OUTER*1.28,"flush glass @ R_in=150mm (hole through acrylic)",
                ha='center',color="white",fontsize=7.2,va='top')

        if ci==0:
            handles=[Line2D([0],[0],color="#00ff88",lw=2,label="e⁻ track"),
                     Line2D([0],[0],color="#60c0ff",lw=1,label="Cherenkov"),
                     Line2D([0],[0],color="#ffaa00",lw=1,ls="--",label="Scintillation"),
                     Line2D([0],[0],color="#da70d6",lw=1,label="WLS (bis-MSB)"),
                     Line2D([0],[0],color="#88ccff",lw=2,label="Acrylic (holes through)"),
                     Line2D([0],[0],color="#66ddff",lw=1.3,label="Flush boundary (medium=glass)"),
                     Line2D([0],[0],color="gold",lw=3,label="R1307 photocathode")]
            ax.legend(handles=handles,facecolor="#111111",edgecolor="#444444",
                      labelcolor="white",fontsize=7.3,loc="upper right")

    fig.suptitle("NeutrinoDetSim — Cross-Sections of SuperK-Style Through-Hole Geometry (post-fix)",
                 color="white",fontsize=11,y=1.02)
    fig.tight_layout()
    fname=os.path.join(out_dir,"geo_cross_sections_R1307_fixed.png")
    fig.savefig(fname,facecolor=fig.get_facecolor(),dpi=200)
    plt.close(fig)
    print(f"  [✓] Cross-sections (fixed) → {fname}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Sagitta fix explainer (the bug, isolated and explained)
# ══════════════════════════════════════════════════════════════════════════════
def plot_sagitta_explainer(out_dir):
    fig,axes=plt.subplots(1,2,figsize=(13,6)); fig.patch.set_facecolor("#080c10")

    for ax in axes:
        ax.set_facecolor("#0d1420"); ax.tick_params(colors="white")
        ax.spines[:].set_color("#333344"); ax.set_aspect("equal")
        ax.grid(True,color="#111122",alpha=0.3)

    # ── LEFT: the bug ──────────────────────────────────────────────────────────
    ax=axes[0]
    ax.set_title("BEFORE — perfect sphere vs flat glass disc\n(caused zero detections)",
                 color="#ff6666",fontsize=10.5)
    theta_full=np.linspace(-1.1,1.1,300)
    x_sphere=R_INNER*np.cos(theta_full); y_sphere=R_INNER*np.sin(theta_full)
    ax.plot(x_sphere,y_sphere,color="#4488ff",lw=2.2,label="medium = perfect sphere")
    half_ang=np.arcsin(TUBE_R/R_INNER)
    ax.plot([R_INNER,R_INNER],[-TUBE_R,TUBE_R],color="#aaddff",lw=3,label="glass = flat disc, flush @ R_in")
    # shade overlap
    th_ov=np.linspace(-half_ang,half_ang,100)
    xov=R_INNER*np.cos(th_ov); yov=R_INNER*np.sin(th_ov)
    ax.fill_betweenx(yov,R_INNER,xov,color="#ff4444",alpha=0.35)
    ax.annotate(f"OVERLAP\n(sagitta = {SAGITTA:.2f} mm)",
                xy=(R_INNER-SAGITTA/2,TUBE_R*0.6),color="#ff8888",fontsize=9,
                ha='center',fontweight='bold')
    ax.set_xlim(R_INNER-15,R_INNER+10); ax.set_ylim(-TUBE_R-10,TUBE_R+10)
    ax.set_xlabel("radial [mm]",color="white"); ax.set_ylabel("transverse [mm]",color="white")
    ax.legend(fontsize=8.5,facecolor="#111111",edgecolor="#444444",labelcolor="white",loc="lower left")
    ax.text(0.97,0.04,"Two volumes occupy the\nsame space → ambiguous\noptical boundary → photons\nnever register a clean\nmedium→glass transition",
            transform=ax.transAxes,ha='right',va='bottom',color="#ffaaaa",fontsize=8,
            bbox=dict(facecolor="#1a0d0d",edgecolor="#553333",boxstyle="round,pad=0.4",alpha=0.85))

    # ── RIGHT: the fix ──────────────────────────────────────────────────────────
    ax=axes[1]
    ax.set_title("AFTER — sagitta-matched notch\n(clean single boundary, detections restored)",
                 color="#66ff99",fontsize=10.5)
    ax.plot(x_sphere,y_sphere,color="#4488ff",lw=1.3,ls=":",alpha=0.4,
            label="original sphere (reference)")
    # Notched medium face
    ax.plot([R_INNER,R_INNER],[-TUBE_R,TUBE_R],color="#4488ff",lw=3,
            label="medium = flat notch face @ R_in")
    # Glass, exactly coincident
    ax.plot([R_INNER+0.15,R_INNER+0.15],[-TUBE_R,TUBE_R],color="#aaddff",lw=3,ls="--",
            label="glass front face @ R_in (offset for visibility)")
    # Removed cap shading
    th_cap=np.linspace(-half_ang,half_ang,100)
    x_cap=R_INNER*np.cos(th_cap); y_cap=R_INNER*np.sin(th_cap)
    x_notch=np.full_like(th_cap,R_INNER-NOTCH_DEPTH)
    ax.fill_betweenx(np.clip(y_cap,-TUBE_R,TUBE_R),x_notch,x_cap,
                      color="#66ff99",alpha=0.15,label=f"notch removed (depth={NOTCH_DEPTH:.2f}mm)")
    ax.axvline(R_INNER-NOTCH_DEPTH,color="#66ff99",lw=0.8,ls=":",alpha=0.6)
    ax.set_xlim(R_INNER-15,R_INNER+10); ax.set_ylim(-TUBE_R-10,TUBE_R+10)
    ax.set_xlabel("radial [mm]",color="white"); ax.set_ylabel("transverse [mm]",color="white")
    ax.legend(fontsize=8,facecolor="#111111",edgecolor="#444444",labelcolor="white",loc="lower left")
    ax.text(0.97,0.04,"Two coincident flat planes\nat r=R_in → exact match,\nzero gap, zero overlap →\nsingle well-defined optical\nboundary → photons detected",
            transform=ax.transAxes,ha='right',va='bottom',color="#aaffcc",fontsize=8,
            bbox=dict(facecolor="#0d1a10",edgecolor="#335533",boxstyle="round,pad=0.4",alpha=0.85))

    fig.suptitle("Why the original geometry gave zero detections, and how the notch fix resolves it",
                 color="white",fontsize=12,y=1.03)
    fig.tight_layout()
    fname=os.path.join(out_dir,"geo_sagitta_fix_explainer.png")
    fig.savefig(fname,facecolor=fig.get_facecolor(),dpi=200)
    plt.close(fig)
    print(f"  [✓] Sagitta fix explainer → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",default="./plots/output")
    args=parser.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    print(f"\n NeutrinoDetSim — SuperK-Style Geometry Visualisation (post-fix)")
    print(f" R_inner={R_INNER}mm  R_outer={R_OUTER}mm  sagitta={SAGITTA:.3f}mm  notch={NOTCH_DEPTH:.3f}mm")
    print(f" Output: {args.out_dir}\n")
    plot_3d_geometry(args.out_dir)
    plot_cross_sections(args.out_dir)
    plot_sagitta_explainer(args.out_dir)
    print("\n Done.\n")


"""
#!/usr/bin/env python3

visualize_geometry.py  —  Realistic Hamamatsu R1307 + Acrylic vessel geometry

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Wedge, Arc, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings, os
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "figure.dpi":150,"savefig.dpi":200,"savefig.bbox":"tight"})

# ── Realistic geometry (mm) ────────────────────────────────────────────────────
R_INNER   = 150.0   # acrylic inner radius
R_OUTER   = 155.0   # acrylic outer radius (5mm wall)
R_TUBE    = 38.0    # R1307 tube radius (76mm OD / 2)
R_PC      = 25.75   # R1307 photocathode active radius (51.5mm / 2)
T_GLASS   = 2.0     # borosilicate window thickness
T_PC      = 0.5     # photocathode disc thickness
L_BODY    = 147.5   # PMT body length
L_TOTAL   = 150.0   # total PMT length

# PMT face sits at rOuter = 155mm
FACE_POS  = R_OUTER
PC_POS    = FACE_POS + T_GLASS + T_PC/2
BODY_POS  = FACE_POS + T_GLASS + T_PC + L_BODY/2

PMT_COLORS = {"glass":"#aaddff","pc":"#ffd700","body":"#555555","body_edge":"#333333"}
AXIS_LABELS = ["+x","−x","+y","−y","+z","−z"]
PMT_FACE_COLORS = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261","#a8dadc"]

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
#  Track generators
# ─────────────────────────────────────────────────────────────────────────────
def electron_track(E_MeV=5.0, n_steps=100):
    pos = np.zeros(3); pts = [pos.copy()]
    direction = np.array([0.55,0.25,0.80]); direction /= np.linalg.norm(direction)
    step=0.20; scat=0.12/max(E_MeV,0.5); rng=0.412*E_MeV**1.27; cum=0.
    for _ in range(n_steps):
        if cum>rng: break
        p1=np.cross(direction,[0,1,0]); p1=p1/np.linalg.norm(p1) if np.linalg.norm(p1)>1e-6 else np.cross(direction,[1,0,0])
        p1/=np.linalg.norm(p1); p2=np.cross(direction,p1)
        direction+= scat*(np.random.randn()*p1+np.random.randn()*p2)
        direction/=np.linalg.norm(direction)
        pos=pos+step*direction; pts.append(pos.copy()); cum+=step
        if np.linalg.norm(pos)>R_INNER*0.1-0.5: break
    return np.array(pts)

def cherenkov_photons(track, n=22, theta_c=41.2):
    photons=[]; tc=np.radians(theta_c); step=max(1,len(track)//n)
    for i in range(0,len(track)-step,step):
        d=track[i+1]-track[i];d/=(np.linalg.norm(d)+1e-12)
        phi=np.random.uniform(0,2*np.pi)
        p=np.cross(d,[0,0,1]); p=(p/np.linalg.norm(p)) if np.linalg.norm(p)>1e-6 else np.cross(d,[1,0,0]); p/=np.linalg.norm(p)
        v=np.cross(d,p); pd=np.cos(tc)*d+np.sin(tc)*(np.cos(phi)*p+np.sin(phi)*v); pd/=np.linalg.norm(pd)
        o=track[i]; a=1.; b=2*np.dot(o,pd); c=np.dot(o,o)-R_INNER**2
        disc=b*b-4*a*c
        if disc<0: continue
        t=(-b+np.sqrt(disc))/(2*a)
        if t<0: continue
        photons.append((o,o+t*pd))
    return photons

def scint_photons(track, n=20):
    photons=[]
    idxs=np.random.choice(len(track),min(n,len(track)),replace=False)
    for i in idxs:
        o=track[i]; d=np.random.randn(3); d/=np.linalg.norm(d)
        a=1.; b=2*np.dot(o,d); c=np.dot(o,o)-R_INNER**2
        disc=b*b-4*a*c
        if disc<0: continue
        t=(-b+np.sqrt(disc))/(2*a)
        if t<0: continue
        photons.append((o,o+t*d))
    return photons

def wls_photons(sci, n=7):
    wls=[]
    ch=np.random.choice(len(sci),min(n,len(sci)),replace=False)
    for i in ch:
        o,e=sci[i]; f=np.random.uniform(0.3,0.6); ab=o+f*(e-o)
        d=np.random.randn(3); d/=np.linalg.norm(d)
        a=1.; b=2*np.dot(ab,d); c=np.dot(ab,ab)-R_INNER**2
        disc=b*b-4*a*c
        if disc<0: continue
        t=(-b+np.sqrt(disc))/(2*a)
        if t<0: continue
        wls.append((o,ab,ab+t*d))
    return wls

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: draw a 3D disc
# ─────────────────────────────────────────────────────────────────────────────
def disc_verts(centre, normal, radius, n=40):
    n_vec=np.array(normal)/np.linalg.norm(normal)
    u=np.cross(n_vec,[1,0,0]); u=(u/np.linalg.norm(u)) if np.linalg.norm(u)>1e-6 else np.cross(n_vec,[0,1,0])
    u/=np.linalg.norm(u); v=np.cross(n_vec,u)
    theta=np.linspace(0,2*np.pi,n)
    pts=np.array(centre)+radius*(np.outer(np.cos(theta),u)+np.outer(np.sin(theta),v))
    return [list(map(tuple,pts))]

def cylinder_surf(centre, axis, radius, length, n=20):
   # Return list of Poly3D faces for a cylinder.
    ax=np.array(axis)/np.linalg.norm(axis)
    u=np.cross(ax,[1,0,0]); u=(u/np.linalg.norm(u)) if np.linalg.norm(u)>1e-6 else np.cross(ax,[0,1,0])
    u/=np.linalg.norm(u); v=np.cross(ax,u)
    theta=np.linspace(0,2*np.pi,n,endpoint=True)
    c=np.array(centre); h=ax*length/2
    faces=[]
    for i in range(n-1):
        p0=c-h+radius*(np.cos(theta[i])*u+np.sin(theta[i])*v)
        p1=c-h+radius*(np.cos(theta[i+1])*u+np.sin(theta[i+1])*v)
        p2=c+h+radius*(np.cos(theta[i+1])*u+np.sin(theta[i+1])*v)
        p3=c+h+radius*(np.cos(theta[i])*u+np.sin(theta[i])*v)
        faces.append([tuple(p0),tuple(p1),tuple(p2),tuple(p3)])
    return faces

# ─────────────────────────────────────────────────────────────────────────────
#  sphere wireframe
# ─────────────────────────────────────────────────────────────────────────────
def sphere_wire(ax, R, n=18, color="steelblue", alpha=0.15, lw=0.5):
    u=np.linspace(0,np.pi,60)
    for phi in np.linspace(0,np.pi,n//2):
        x=R*np.sin(u)*np.cos(phi); y=R*np.sin(u)*np.sin(phi); z=R*np.cos(u)
        ax.plot(x,y,z,color=color,alpha=alpha,lw=lw)
    for th in np.linspace(0,np.pi,n//2+1)[1:-1]:
        uu=np.linspace(0,2*np.pi,60)
        x=R*np.sin(th)*np.cos(uu); y=R*np.sin(th)*np.sin(uu); z=R*np.cos(th)*np.ones_like(uu)
        ax.plot(x,y,z,color=color,alpha=alpha,lw=lw)

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1 — 3D Realistic Detector View
# ─────────────────────────────────────────────────────────────────────────────
def plot_3d_geometry(out_dir):
    fig = plt.figure(figsize=(18,14))
    fig.patch.set_facecolor("#080c10")

    ax = fig.add_axes([0.01,0.12,0.60,0.85], projection='3d')
    ax.set_facecolor("#080c10")
    ax.set_xlim(-280,280); ax.set_ylim(-280,280); ax.set_zlim(-280,280)
    ax.set_xlabel("x [mm]",color="white",labelpad=6)
    ax.set_ylabel("y [mm]",color="white",labelpad=6)
    ax.set_zlabel("z [mm]",color="white",labelpad=6)
    ax.tick_params(colors="white")
    for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        pane.fill=False; pane.set_edgecolor("#1a1a2e")
    ax.grid(True,color="#111122",alpha=0.4)

    # Acrylic shell wireframe (outer)
    sphere_wire(ax, R_OUTER, n=24, color="#88ccff", alpha=0.12, lw=0.5)
    # Detection medium (inner)
    sphere_wire(ax, R_INNER, n=24, color="#4488ff", alpha=0.20, lw=0.6)

    # Acrylic shell semi-transparent (partial sphere for cutaway)
    u=np.linspace(0,2*np.pi,40); v=np.linspace(0,np.pi,20)
    xs=R_OUTER*np.outer(np.cos(u),np.sin(v))
    ys=R_OUTER*np.outer(np.sin(u),np.sin(v))
    zs=R_OUTER*np.outer(np.ones_like(u),np.cos(v))
    ax.plot_surface(xs,ys,zs,alpha=0.04,color="#88ccff",shade=False,linewidth=0)

    # ── 6 PMTs ─────────────────────────────────────────────────────────────────
    axes_dirs = [
        (np.array([ 1, 0, 0]),  np.array([-1, 0, 0])),
        (np.array([-1, 0, 0]),  np.array([ 1, 0, 0])),
        (np.array([ 0, 1, 0]),  np.array([ 0,-1, 0])),
        (np.array([ 0,-1, 0]),  np.array([ 0, 1, 0])),
        (np.array([ 0, 0, 1]),  np.array([ 0, 0,-1])),
        (np.array([ 0, 0,-1]),  np.array([ 0, 0, 1])),
    ]

    for idx,(outward,inward) in enumerate(axes_dirs):
        col = PMT_FACE_COLORS[idx]
        glass_c = outward*(FACE_POS + T_GLASS/2)
        pc_c    = outward*(FACE_POS + T_GLASS + T_PC/2)
        body_c  = outward*(FACE_POS + T_GLASS + T_PC + L_BODY/2)

        # Glass window (full tube radius)
        gv = disc_verts(glass_c, outward, R_TUBE)
        poly = Poly3DCollection(gv, alpha=0.55, facecolor="#aaddff", edgecolor="#88bbdd", lw=0.8)
        ax.add_collection3d(poly)

        # Photocathode (smaller active area, gold)
        pv = disc_verts(pc_c, outward, R_PC)
        poly = Poly3DCollection(pv, alpha=0.92, facecolor=col, edgecolor="gold", lw=1.2)
        ax.add_collection3d(poly)

        # PMT body cylinder
        faces = cylinder_surf(body_c, outward, R_TUBE, L_BODY, n=16)
        poly = Poly3DCollection(faces, alpha=0.45, facecolor="#333344", edgecolor="#222233", lw=0.3)
        ax.add_collection3d(poly)

        # Inward arrow (PMT pointing inward)
        tip = inward * 20 + outward*(FACE_POS + T_GLASS)
        ax.quiver(*tip, *(inward*25), color="gold", alpha=0.8,
                  lw=1.5, arrow_length_ratio=0.4, length=1)

        # Label
        lbl = outward*(FACE_POS + L_TOTAL + 18)
        ax.text(*lbl, f"R1307\n{AXIS_LABELS[idx]}", color=col,
                fontsize=7.5, ha='center', va='center', fontweight='bold')

    # ── Tracks ─────────────────────────────────────────────────────────────────
    track  = electron_track(5.0, 120)
    cer    = cherenkov_photons(track, 22)
    sci    = scint_photons(track, 25)
    wls    = wls_photons(sci, 8)

    ax.plot(track[:,0],track[:,1],track[:,2],
            color="#00ff88",lw=2.5,alpha=0.95,zorder=10,label="e⁻ 5 MeV")
    ax.scatter(0,0,0,color="lime",s=90,zorder=15,edgecolors="white",lw=1.2)

    for i,(o,e) in enumerate(cer):
        ax.plot([o[0],e[0]],[o[1],e[1]],[o[2],e[2]],
                color="#60c0ff",alpha=0.40,lw=0.8,
                label="Cherenkov" if i==0 else "")

    for i,(o,e) in enumerate(sci):
        ax.plot([o[0],e[0]],[o[1],e[1]],[o[2],e[2]],
                color="#ffaa00",alpha=0.30,lw=0.7,ls="--",
                label="PPO scint." if i==0 else "")

    for i,(_,ab,ne) in enumerate(wls):
        ax.plot([ab[0],ne[0]],[ab[1],ne[1]],[ab[2],ne[2]],
                color="#da70d6",alpha=0.60,lw=1.0,
                label="bis-MSB WLS" if i==0 else "")
        ax.scatter(*ab,color="violet",s=15,alpha=0.8,zorder=8)

    # Coordinate axes
    for vec,col,lbl in [([1,0,0],"#ff4444","X"),([0,1,0],"#44ff44","Y"),([0,0,1],"#4499ff","Z")]:
        L=R_INNER*1.25
        ax.quiver(0,0,0,vec[0]*L,vec[1]*L,vec[2]*L,color=col,lw=2,arrow_length_ratio=0.07)
        ax.text(vec[0]*(L+15),vec[1]*(L+15),vec[2]*(L+15),lbl,color=col,fontsize=11,fontweight='bold')

    ax.set_title("NeutrinoDetSim — Realistic Geometry\n"
                 "Acrylic (PMMA) vessel · 6× Hamamatsu R1307 · PPO+bis-MSB · 5 MeV e⁻",
                 color="white",fontsize=11,pad=8)

    handles=[mpatches.Patch(color="#00ff88",label="e⁻ track (5 MeV)"),
             mpatches.Patch(color="#60c0ff",label="Cherenkov photons"),
             mpatches.Patch(color="#ffaa00",label="PPO scintillation"),
             mpatches.Patch(color="#da70d6",label="bis-MSB WLS re-emit"),
             mpatches.Patch(color="#aaddff",label="R1307 borosilicate glass"),
             mpatches.Patch(color="gold",   label="R1307 photocathode (Ø51.5mm)"),
             mpatches.Patch(color="#88ccff",alpha=0.4,label="Acrylic vessel (PMMA, 5mm)"),]
    ax.legend(handles=handles,loc="lower left",facecolor="#111111",edgecolor="#444444",
              labelcolor="white",fontsize=7.5,framealpha=0.85)
    ax.view_init(elev=20,azim=-50)

    # ── Right panels ────────────────────────────────────────────────────────────
    # Cross-section (XZ)
    ax2 = fig.add_axes([0.62,0.52,0.36,0.44])
    ax2.set_facecolor("#080c10"); ax2.set_aspect("equal")
    ax2.set_xlim(-290,290); ax2.set_ylim(-290,290)
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#333344")
    ax2.set_title("XZ Cross-Section",color="white",fontsize=10)
    ax2.set_xlabel("x [mm]",color="white",fontsize=9)
    ax2.set_ylabel("z [mm]",color="white",fontsize=9)
    ax2.grid(True,color="#111122",alpha=0.4)

    th=np.linspace(0,2*np.pi,300)
    # Acrylic outer
    ax2.plot(R_OUTER*np.cos(th),R_OUTER*np.sin(th),color="#88ccff",lw=2.0,alpha=0.8)
    ax2.fill_between(R_OUTER*np.cos(th),R_OUTER*np.sin(th),
                     R_INNER*np.sin(th),alpha=0.15,color="#88ccff")
    # Inner medium
    ax2.plot(R_INNER*np.cos(th),R_INNER*np.sin(th),color="#4488ff",lw=1.5,alpha=0.7)
    ax2.fill(R_INNER*np.cos(th),R_INNER*np.sin(th),alpha=0.08,color="#4488ff")

    # PMTs in XZ: ±x and ±z
    for (px,pz,col,lbl) in [(FACE_POS,0,"#e63946","+x"),(-FACE_POS,0,"#457b9d","−x"),
                              (0,FACE_POS,"#a8dadc","+z"),(0,-FACE_POS,"#f4a261","−z")]:
        nd=np.array([px,pz]); nm=np.linalg.norm(nd)
        if nm == 0: continue
        ndu = nd/nm
        perp=np.array([-ndu[1],ndu[0]])
        # glass rect
        gc=ndu*(FACE_POS+T_GLASS/2)
        gpts=np.array([gc+R_TUBE*perp, gc-R_TUBE*perp,
                       gc-R_TUBE*perp+ndu*T_GLASS, gc+R_TUBE*perp+ndu*T_GLASS])
        ax2.fill(gpts[:,0],gpts[:,1],color="#aaddff",alpha=0.55)
        ax2.plot(np.append(gpts[:,0],gpts[0,0]),np.append(gpts[:,1],gpts[0,1]),
                 color="#88bbdd",lw=0.8)
        # photocathode line (gold)
        pc_centre=ndu*(FACE_POS+T_GLASS+T_PC/2)
        p1=pc_centre+R_PC*perp; p2=pc_centre-R_PC*perp
        ax2.plot([p1[0],p2[0]],[p1[1],p2[1]],color=col,lw=5,alpha=0.90,solid_capstyle='round')
        # body rect
        bc=ndu*(FACE_POS+T_GLASS+T_PC+L_BODY/2)
        bpts=np.array([bc+R_TUBE*perp, bc-R_TUBE*perp,
                       bc-R_TUBE*perp+ndu*L_BODY/2, bc+R_TUBE*perp+ndu*L_BODY/2])
        ax2.fill(bpts[:,0],bpts[:,1],color="#333344",alpha=0.70)
        ax2.plot(np.append(bpts[:,0],bpts[0,0]),np.append(bpts[:,1],bpts[0,1]),
                 color="#444455",lw=0.6)
        ax2.text(ndu[0]*(FACE_POS+L_TOTAL+22),ndu[1]*(FACE_POS+L_TOTAL+22),
                 f"R1307\n{lbl}",color=col,fontsize=7.5,ha='center',va='center',fontweight='bold')
        # Inward arrow
        arr_s=ndu*(FACE_POS+T_GLASS)
        arr_e=ndu*(FACE_POS-15)
        ax2.annotate("",xy=arr_e,xytext=arr_s,
                    arrowprops=dict(arrowstyle="->",color="gold",lw=1.5))

    # Tracks
    ax2.plot(track[:,0],track[:,2],color="#00ff88",lw=2.0,alpha=0.9)
    ax2.scatter(0,0,color="lime",s=50,zorder=10)
    for o,e in cer: ax2.plot([o[0],e[0]],[o[2],e[2]],color="#60c0ff",alpha=0.35,lw=0.8)
    for o,e in sci[:10]: ax2.plot([o[0],e[0]],[o[2],e[2]],color="#ffaa00",alpha=0.25,lw=0.6,ls="--")

    # Dimension arrows
    ax2.annotate("",xy=(R_OUTER,-R_OUTER*1.22),xytext=(-R_OUTER,-R_OUTER*1.22),
               arrowprops=dict(arrowstyle="<->",color="white",lw=1.2))
    ax2.text(0,-R_OUTER*1.31,"2×155 = 310 mm",ha='center',color="white",fontsize=8,va='top')
    ax2.annotate("",xy=(R_INNER,-R_INNER*1.12),xytext=(-R_INNER,-R_INNER*1.12),
               arrowprops=dict(arrowstyle="<->",color="#88aaff",lw=1.0))
    ax2.text(0,-R_INNER*1.20,"Ø300 mm inner",ha='center',color="#88aaff",fontsize=8,va='top')
    ax2.text(R_INNER*1.22,0,"5mm\nPMMA",ha='left',color="#88ccff",fontsize=7.5,va='center')

    # ── R1307 detail panel ─────────────────────────────────────────────────────
    ax3 = fig.add_axes([0.62,0.06,0.36,0.40])
    ax3.set_facecolor("#080c10"); ax3.set_aspect("equal")
    ax3.set_xlim(-10,175); ax3.set_ylim(-50,50)
    ax3.tick_params(colors="white"); ax3.spines[:].set_color("#333344")
    ax3.set_title("R1307 Cross-Section (1 PMT)",color="white",fontsize=10)
    ax3.set_xlabel("Along PMT axis [mm]",color="white",fontsize=8)
    ax3.grid(True,color="#111122",alpha=0.3)
    ax3.axvline(0,color="#88ccff",lw=1,ls="--",alpha=0.5)

    # Glass window
    from matplotlib.patches import FancyBboxPatch
    ax3.add_patch(Rectangle((0,-R_TUBE),T_GLASS,2*R_TUBE,
                             facecolor="#aaddff",edgecolor="#88bbdd",lw=1.0,alpha=0.7))
    ax3.text(T_GLASS/2,R_TUBE+4,"BS glass\nn=1.50",ha='center',color="#aaddff",fontsize=7.5)

    # Photocathode
    ax3.add_patch(Rectangle((T_GLASS,-R_PC),T_PC,2*R_PC,
                             facecolor="#ffd700",edgecolor="orange",lw=1.2,alpha=0.95))
    ax3.text(T_GLASS+T_PC/2,-R_PC-7,"Photocathode\nØ51.5mm\nK₂CsSb",
             ha='center',color="#ffd700",fontsize=7,va='top')

    # PMT body
    ax3.add_patch(Rectangle((T_GLASS+T_PC,-R_TUBE),L_BODY,2*R_TUBE,
                             facecolor="#222233",edgecolor="#444466",lw=0.8,alpha=0.8))
    ax3.text(T_GLASS+T_PC+L_BODY/2,0,"Vacuum tube\n(dynode region)",
             ha='center',color="#aaaacc",fontsize=7.5)

    # Dimension arrows
    ax3.annotate("",xy=(0+T_GLASS,-42),xytext=(0,-42),
               arrowprops=dict(arrowstyle="<->",color="white",lw=1.0))
    ax3.text(T_GLASS/2,-46,"2mm",ha='center',color="white",fontsize=7.5)
    ax3.annotate("",xy=(T_GLASS+T_PC+L_BODY,-42),xytext=(0,-42),
               arrowprops=dict(arrowstyle="<->",color="#aaaacc",lw=1.0))
    ax3.text((T_GLASS+T_PC+L_BODY)/2,-46,"150mm total",ha='center',color="#aaaacc",fontsize=7.5)
    ax3.text(-8,0,"Ø76mm",ha='right',color="#aaaacc",fontsize=7.5,va='center',rotation=90)
    ax3.text(T_GLASS+T_PC/2,R_PC+4,"Ø51.5mm",ha='center',color="#ffd700",fontsize=7)

    # QE annotation
    ax3.text(T_GLASS+1,38,"QE=30%\n@420nm\nR1307",color="gold",fontsize=7.5,
             bbox=dict(facecolor="#111111",edgecolor="gold",boxstyle="round,pad=0.3",alpha=0.85))

    # Info box bottom left
    fig.text(0.01,0.02,
             "Hamamatsu R1307  ·  3-inch bialkali PMT\n"
             "Photocathode: K₂CsSb,  Ø51.5 mm active area\n"
             "Window: borosilicate glass  n=1.500 @ 420 nm\n"
             "QE peak: 30% @ 420 nm  |  Range: 300–650 nm\n"
             "Rise time: 3 ns  |  Transit time: 8 ns  |  Gain: 2.7×10⁵\n"
             "Vessel: PMMA acrylic  5 mm thick  |  Inner Ø300 mm",
             color="lightgrey",fontsize=8,va='bottom',
             bbox=dict(facecolor="#111111",edgecolor="#333344",
                       boxstyle="round,pad=0.5",alpha=0.88))

    os.makedirs(out_dir,exist_ok=True)
    fname=os.path.join(out_dir,"geo_3d_detector_R1307.png")
    fig.savefig(fname,facecolor=fig.get_facecolor(),dpi=200)
    plt.close(fig)
    print(f"  [✓] 3D R1307 geometry → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2 — R1307 QE + physics detail
# ─────────────────────────────────────────────────────────────────────────────
def plot_r1307_qe_and_physics(out_dir):
    fig, axes = plt.subplots(2,2,figsize=(13,10))
    fig.patch.set_facecolor("#080c10")
    for ax in axes.flatten():
        ax.set_facecolor("#0d1420"); ax.tick_params(colors="white")
        ax.spines[:].set_color("#333344"); ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white"); ax.title.set_color("white")
        ax.grid(True,color="#111122",alpha=0.4)

    # ── (A) R1307 QE spectrum ─────────────────────────────────────────────────
    ax=axes[0,0]
    wl_qe=np.array([270,290,310,330,350,370,390,410,420,430,
                    440,460,480,500,520,540,560,580,600,640])
    qe   =np.array([1, 4,10,18,24,27,29,30,30,30,
                    29,26,22,17,13, 9, 6, 3, 1, 0])/100.
    wl_f=np.linspace(260,660,400)
    qe_f=np.interp(wl_f,wl_qe,qe,left=0,right=0)

    ax.fill_between(wl_f,qe_f*100,alpha=0.35,color="#ffd700")
    ax.plot(wl_f,qe_f*100,color="#ffd700",lw=2.5,label="R1307 QE (digitised from Fig.1)")
    ax.axvline(420,color="white",lw=1,ls="--",alpha=0.6)
    ax.text(422,31,"420 nm\nPeak 30%",color="white",fontsize=8.5)
    ax.axvspan(300,650,alpha=0.05,color="white",label="Spectral range 300–650 nm")

    # PPO+bis-MSB emission overlay (normalised to QE scale)
    wl_sp=np.linspace(260,620,400)
    ppo_e=np.exp(-0.5*((wl_sp-360)/20)**2)+0.4*np.exp(-0.5*((wl_sp-380)/14)**2)
    bms_e=np.exp(-0.5*((wl_sp-420)/22)**2)+0.28*np.exp(-0.5*((wl_sp-448)/16)**2)
    ppo_e/=ppo_e.max(); bms_e/=bms_e.max()
    ax.plot(wl_sp,ppo_e*22,color="#44aaff",lw=1.5,ls="--",alpha=0.8,label="PPO emit (360nm, norm.)")
    ax.plot(wl_sp,bms_e*22,color="#ffaa00",lw=1.5,ls=":",alpha=0.8,label="bis-MSB emit (420nm, norm.)")

    ax.set_xlabel("Wavelength [nm]"); ax.set_ylabel("Quantum Efficiency [%]")
    ax.set_title("A — Hamamatsu R1307 QE + Fluor Emission"); ax.set_xlim(260,660)
    ax.legend(fontsize=8,facecolor="#111111",edgecolor="#333344",labelcolor="white")

    # ── (B) Cherenkov threshold per medium ────────────────────────────────────
    ax=axes[0,1]
    media_n={"H₂O":1.340,"D₂O":1.333,"WbLS":1.345,"Gd-WbLS":1.346,
             "Te-WbLS":1.353,"D₂O+LAB":1.338,"D₂O+Gd-LAB":1.339,"D₂O+Te-LAB":1.345}
    cols=["#0077BB","#33BBEE","#EE7733","#CC3311","#EE3377","#009988","#117733","#AA3377"]
    for i,(med,n) in enumerate(media_n.items()):
        E_thr=(1/np.sqrt(1-1/n**2)-1)*0.511
        ax.barh(i,E_thr,color=cols[i],alpha=0.85,edgecolor="#333344",lw=0.6)
        ax.text(E_thr+0.005,i,f"{E_thr:.3f} MeV",va='center',color="white",fontsize=8.5)
    ax.set_yticks(range(len(media_n))); ax.set_yticklabels(list(media_n.keys()),color="white",fontsize=9)
    ax.set_xlabel("Cherenkov threshold [MeV]")
    ax.set_title("B — Cherenkov Threshold per Medium")
    ax.axvline(0.511,color="grey",lw=1,ls=":",alpha=0.5); ax.text(0.513,7.5,"mₑc²",color="grey",fontsize=8)

    # ── (C) Refractive index dispersion + R1307 glass ─────────────────────────
    ax=axes[1,0]
    wl_n=np.linspace(270,620,300)
    ri_data = {
        "H₂O":       [1.332,1.333,1.334,1.335,1.337,1.339,1.341,1.343,1.346,1.350,1.354,1.358],
        "D₂O":       [1.325,1.326,1.327,1.328,1.330,1.332,1.334,1.336,1.339,1.343,1.347,1.351],
        "WbLS":      [1.334,1.335,1.336,1.337,1.340,1.341,1.344,1.346,1.349,1.353,1.357,1.362],
        "Acrylic":   [1.504,1.502,1.500,1.498,1.494,1.492,1.490,1.488,1.486,1.484,1.482,1.480],
        "BS Glass":  [1.530,1.527,1.524,1.521,1.517,1.515,1.513,1.511,1.509,1.506,1.503,1.500],
    }
    ri_wl=[270,290,310,330,370,390,420,440,480,540,590,620]
    ri_cols=["#0077BB","#33BBEE","#EE7733","#88ccff","#ffd700"]
    ri_ls=["-","-","-","--","-."]
    for (med,vals),col,ls in zip(ri_data.items(),ri_cols,ri_ls):
        n_i=np.interp(wl_n,ri_wl,vals)
        ax.plot(wl_n,n_i,color=col,lw=2.0,ls=ls,label=med)
    ax.axvline(420,color="white",lw=0.8,ls=":",alpha=0.4)
    ax.set_xlabel("Wavelength [nm]"); ax.set_ylabel("Refractive index n")
    ax.set_title("C — Refractive Index Dispersion (medium → glass)")
    ax.legend(fontsize=8.5,facecolor="#111111",edgecolor="#333344",labelcolor="white")
    ax.set_xlim(270,620)

    # ── (D) PMT solid angle + geometric coverage ──────────────────────────────
    ax=axes[1,1]
    r_inner_range=np.linspace(50,300,200)
    for n_pmt,col,lbl in [(6,"#ffd700","6 PMTs (this design)"),
                           (12,"#ff7733","12 PMTs"),(20,"#44ff88","20 PMTs")]:
        cov=n_pmt*np.pi*R_PC**2/(4*np.pi*r_inner_range**2)*100
        ax.plot(r_inner_range,cov,color=col,lw=2.0,label=lbl)
    ax.axvline(R_INNER,color="white",lw=1.2,ls="--",alpha=0.7)
    ax.text(R_INNER+3,ax.get_ylim()[1]*0.9 if ax.get_ylim()[1]>0 else 5,
            f"R={R_INNER:.0f}mm\nthis design",color="white",fontsize=8.5)
    # Mark the current coverage
    cov_now=6*np.pi*R_PC**2/(4*np.pi*R_INNER**2)*100
    ax.scatter([R_INNER],[cov_now],color="#ffd700",s=80,zorder=5)
    ax.text(R_INNER+5,cov_now,f" {cov_now:.1f}%",color="#ffd700",fontsize=9,va='center')
    ax.set_xlabel("Inner sphere radius [mm]")
    ax.set_ylabel("Geometric coverage [%]")
    ax.set_title("D — R1307 Photocathode Coverage vs Vessel Radius")
    ax.legend(fontsize=9,facecolor="#111111",edgecolor="#333344",labelcolor="white")

    fig.suptitle("Hamamatsu R1307 PMT — Realistic Simulation Parameters",
                 color="white",fontsize=12,y=1.01)
    fig.tight_layout()
    fname=os.path.join(out_dir,"geo_R1307_parameters.png")
    fig.savefig(fname,facecolor=fig.get_facecolor(),dpi=200)
    plt.close(fig)
    print(f"  [✓] R1307 parameters  → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3 — Three cross-sections (unchanged but with correct dimensions)
# ─────────────────────────────────────────────────────────────────────────────
def plot_cross_sections(out_dir):
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    fig.patch.set_facecolor("#080c10")
    titles=["XY Plane (top view)","XZ Plane (side view)","YZ Plane (front view)"]
    plane_idx=[(0,1),(0,2),(1,2)]
    plane_pmts=[
        [(1,0,0,"#e63946","+x"),(-1,0,0,"#457b9d","−x"),(0,1,0,"#2a9d8f","+y"),(0,-1,0,"#e9c46a","−y")],
        [(1,0,0,"#e63946","+x"),(-1,0,0,"#457b9d","−x"),(0,0,1,"#a8dadc","+z"),(0,0,-1,"#f4a261","−z")],
        [(0,1,0,"#2a9d8f","+y"),(0,-1,0,"#e9c46a","−y"),(0,0,1,"#a8dadc","+z"),(0,0,-1,"#f4a261","−z")],
    ]
    al=["x [mm]","y [mm]","z [mm]"]
    track=electron_track(5.0,120)
    cer=cherenkov_photons(track,20)
    sci=scint_photons(track,18)
    wls=wls_photons(sci,6)

    for ci,(ax,title,(d0,d1),pmts) in enumerate(zip(axes,titles,plane_idx,plane_pmts)):
        ax.set_facecolor("#080c10"); ax.set_aspect("equal")
        ax.set_xlim(-290,290); ax.set_ylim(-290,290)
        ax.tick_params(colors="white"); ax.spines[:].set_color("#333344")
        ax.set_title(title,color="white",fontsize=10)
        ax.set_xlabel(al[d0],color="white"); ax.set_ylabel(al[d1],color="white")
        ax.grid(True,color="#111122",alpha=0.4)
        th=np.linspace(0,2*np.pi,300)
        # Acrylic shell
        ax.plot(R_OUTER*np.cos(th),R_OUTER*np.sin(th),color="#88ccff",lw=2.0,alpha=0.7)
        ax.fill_between(R_OUTER*np.cos(th),R_OUTER*np.sin(th),
                        R_INNER*np.sin(th),alpha=0.12,color="#88ccff")
        # Inner medium
        ax.plot(R_INNER*np.cos(th),R_INNER*np.sin(th),color="#4488ff",lw=1.5,alpha=0.6)
        ax.fill(R_INNER*np.cos(th),R_INNER*np.sin(th),alpha=0.06,color="#4488ff")

        for px,py,pz,col,lbl in pmts:
            pos2=np.array([px,py,pz],dtype=float)[[d0,d1]]*FACE_POS
            nd=np.array([px,py,pz],dtype=float)[[d0,d1]]
            nm=np.linalg.norm(nd); nd=nd/nm if nm>0 else nd
            perp=np.array([-nd[1],nd[0]])
            # glass
            gc=nd*(FACE_POS+T_GLASS/2)
            gpts=np.array([gc+R_TUBE*perp,gc-R_TUBE*perp,
                           gc-R_TUBE*perp+nd*T_GLASS,gc+R_TUBE*perp+nd*T_GLASS])
            ax.fill(gpts[:,0],gpts[:,1],color="#aaddff",alpha=0.45)
            # photocathode
            pcc=nd*(FACE_POS+T_GLASS+T_PC/2)
            p1=pcc+R_PC*perp; p2=pcc-R_PC*perp
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=col,lw=5,alpha=0.90,solid_capstyle='round')
            # body
            bc=nd*(FACE_POS+T_GLASS+T_PC+L_BODY/2)
            bpts=np.array([bc+R_TUBE*perp,bc-R_TUBE*perp,
                           bc-R_TUBE*perp+nd*L_BODY/2,bc+R_TUBE*perp+nd*L_BODY/2])
            ax.fill(bpts[:,0],bpts[:,1],color="#222233",alpha=0.65)
            ax.text(nd[0]*(FACE_POS+L_TOTAL+22),nd[1]*(FACE_POS+L_TOTAL+22),
                    f"{lbl}",color=col,ha='center',va='center',fontsize=8,fontweight='bold')
            # inward arrow
            ax.annotate("",xy=nd*(FACE_POS-15),xytext=nd*(FACE_POS+T_GLASS),
                       arrowprops=dict(arrowstyle="->",color="gold",lw=1.5))

        ax.plot(track[:,d0],track[:,d1],color="#00ff88",lw=2.0,alpha=0.9)
        ax.scatter(0,0,color="lime",s=50,zorder=10)
        for o,e in cer: ax.plot([o[d0],e[d0]],[o[d1],e[d1]],color="#60c0ff",alpha=0.35,lw=0.8)
        for o,e in sci[:10]: ax.plot([o[d0],e[d0]],[o[d1],e[d1]],color="#ffaa00",alpha=0.22,lw=0.6,ls="--")
        for _,ab,ne in wls: ax.plot([ab[d0],ne[d0]],[ab[d1],ne[d1]],color="#da70d6",alpha=0.55,lw=0.9)

        ax.annotate("",xy=(R_OUTER,-R_OUTER*1.20),xytext=(-R_OUTER,-R_OUTER*1.20),
                   arrowprops=dict(arrowstyle="<->",color="white",lw=1.2))
        ax.text(0,-R_OUTER*1.28,"2×155=310 mm (incl. acrylic)",ha='center',color="white",fontsize=7.5,va='top')
        if ci==0:
            from matplotlib.lines import Line2D
            handles=[Line2D([0],[0],color="#00ff88",lw=2,label="e⁻ track"),
                     Line2D([0],[0],color="#60c0ff",lw=1,label="Cherenkov"),
                     Line2D([0],[0],color="#ffaa00",lw=1,ls="--",label="Scintillation"),
                     Line2D([0],[0],color="#da70d6",lw=1,label="WLS (bis-MSB)"),
                     Line2D([0],[0],color="#88ccff",lw=2,label="Acrylic PMMA"),
                     Line2D([0],[0],color="gold",lw=3,label="R1307 photocathode")]
            ax.legend(handles=handles,facecolor="#111111",edgecolor="#444444",
                      labelcolor="white",fontsize=7.5,loc="upper right")

    fig.suptitle("NeutrinoDetSim — Three Orthogonal Cross-Sections | Realistic R1307 + PMMA Vessel",
                 color="white",fontsize=11,y=1.02)
    fig.tight_layout()
    fname=os.path.join(out_dir,"geo_cross_sections_R1307.png")
    fig.savefig(fname,facecolor=fig.get_facecolor(),dpi=200)
    plt.close(fig)
    print(f"  [✓] Cross-sections    → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",default="./plots/output")
    args=parser.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    print(f"\n NeutrinoDetSim — Realistic R1307 Geometry Visualisation")
    print(f" Output: {args.out_dir}\n")
    plot_3d_geometry(args.out_dir)
    plot_r1307_qe_and_physics(args.out_dir)
    plot_cross_sections(args.out_dir)
    print("\n Done.\n")
"""
    """
    #!/usr/bin/env python3

visualize_geometry.py
=====================
Standalone matplotlib visualisation of the 30-cm spherical neutrino detector.
Draws:
  • Spherical detector vessel (wireframe + transparent fill)
  • 6 x 3-inch PMTs on ±x, ±y, ±z axes (coloured discs facing inward)
  • PMT active faces (gold) with inward-pointing arrows
  • Coordinate axes and dimension annotations
  • Simulated particle tracks (electron + secondaries + optical photons)
  • Cherenkov cone schematic
  • PPO/bis-MSB WLS re-emission arrows
  • Legend and physics annotations

No Geant4 or ROOT required — pure matplotlib/numpy.


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Circle, Wedge, Arc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
try:
    from scipy.special import sph_harm
except ImportError:
    from scipy.special import sph_harm_y as sph_harm
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
#  Geometry parameters (cm)
# ─────────────────────────────────────────────────────────────────────────────
R_DET   = 15.0   # detector sphere radius [cm]
R_PMT   = 3.81   # PMT sensitive area radius [cm] (3-inch = 7.62 cm diam / 2)
T_PMT   = 0.1    # PMT disc thickness [cm]
N_PMT   = 6

# PMT positions and orientations (unit inward-pointing normal)
PMT_AXES = [
    (+1, 0, 0), (-1, 0, 0),
    (0, +1, 0), (0, -1, 0),
    (0,  0,+1), (0,  0,-1),
]
PMT_LABELS = ["+x", "−x", "+y", "−y", "+z", "−z"]
PMT_COLORS = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261","#a8dadc"]

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: draw a 3D disc (PMT face)
# ─────────────────────────────────────────────────────────────────────────────
def make_disc_verts(centre, normal, radius, n_pts=40):
    #Return vertices of a disc (closed polygon) perpendicular to `normal`.
    cx, cy, cz = centre
    nx, ny, nz = normal / np.linalg.norm(normal)
    # Build orthonormal basis in disc plane
    if abs(nx) < 0.9:
        u = np.cross([nx,ny,nz], [1,0,0])
    else:
        u = np.cross([nx,ny,nz], [0,1,0])
    u /= np.linalg.norm(u)
    v  = np.cross([nx,ny,nz], u)
    theta = np.linspace(0, 2*np.pi, n_pts)
    xs = cx + radius*(np.cos(theta)*u[0] + np.sin(theta)*v[0])
    ys = cy + radius*(np.cos(theta)*u[1] + np.sin(theta)*v[1])
    zs = cz + radius*(np.cos(theta)*u[2] + np.sin(theta)*v[2])
    return list(zip(xs, ys, zs))

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: sphere wireframe
# ─────────────────────────────────────────────────────────────────────────────
def sphere_wireframe(ax, R, n_lines=18, color="steelblue", alpha=0.18, lw=0.6):
    u = np.linspace(0, 2*np.pi, 120)
    v = np.linspace(0,   np.pi,  60)
    # Meridians
    for phi in np.linspace(0, np.pi, n_lines//2):
        x = R * np.sin(v) * np.cos(phi)
        y = R * np.sin(v) * np.sin(phi)
        z = R * np.cos(v)
        ax.plot(x, y, z, color=color, alpha=alpha, lw=lw)
    # Parallels
    for theta in np.linspace(0, np.pi, n_lines//2+1)[1:-1]:
        x = R * np.sin(theta) * np.cos(u)
        y = R * np.sin(theta) * np.sin(u)
        z = R * np.cos(theta) * np.ones_like(u)
        ax.plot(x, y, z, color=color, alpha=alpha, lw=lw)

# ─────────────────────────────────────────────────────────────────────────────
#  Simulated track generator (deterministic for reproducibility)
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)

def gen_electron_track(start, direction, energy_MeV, n_steps=80):
   
   # Approximate electron track with multiple-scattering deviation.
    #Returns list of (x,y,z) points.
    
    pos  = np.array(start, dtype=float)
    dirn = np.array(direction, dtype=float)
    dirn /= np.linalg.norm(dirn)
    pts  = [pos.copy()]
    # Mean free path scales roughly with energy (simplified)
    step = 0.20   # cm per step
    scatter_sigma = 0.12 / max(energy_MeV, 0.5)  # less scatter at high E
    # Range approximation: R(E) ~ 0.412 * E^(1.27) cm in water (for < 20 MeV)
    max_range = 0.412 * energy_MeV**1.27
    cumlen    = 0.
    for _ in range(n_steps):
        if cumlen > max_range:
            break
        # Add multiple scattering
        perp1 = np.cross(dirn, [0,1,0])
        if np.linalg.norm(perp1) < 1e-6:
            perp1 = np.cross(dirn, [1,0,0])
        perp1 /= np.linalg.norm(perp1)
        perp2  = np.cross(dirn, perp1)
        deflect = scatter_sigma * (np.random.randn()*perp1 + np.random.randn()*perp2)
        dirn   = dirn + deflect
        dirn  /= np.linalg.norm(dirn)
        pos    = pos + step * dirn
        pts.append(pos.copy())
        cumlen += step
        if np.linalg.norm(pos) > R_DET - 0.5:
            break
    return np.array(pts)

def gen_cherenkov_photons(track_pts, n_photons=18, spread_angle_deg=41.2):
    #Generate Cherenkov photon directions along the track.
    photons = []
    theta_c = np.radians(spread_angle_deg)  # ~41° in water (n=1.34)
    step = max(1, len(track_pts)//n_photons)
    for i in range(0, len(track_pts)-step, step):
        origin = track_pts[i]
        local_dir = track_pts[i+1] - track_pts[i]
        if np.linalg.norm(local_dir) < 1e-9:
            continue
        local_dir /= np.linalg.norm(local_dir)
        # Random azimuth around track direction
        phi = np.random.uniform(0, 2*np.pi)
        # Build rotation: find perpendicular
        perp = np.cross(local_dir, [0,0,1])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(local_dir, [1,0,0])
        perp /= np.linalg.norm(perp)
        perp2 = np.cross(local_dir, perp)
        photon_dir = (np.cos(theta_c)*local_dir
                      + np.sin(theta_c)*(np.cos(phi)*perp + np.sin(phi)*perp2))
        photon_dir /= np.linalg.norm(photon_dir)
        # Propagate to sphere wall
        # solve |origin + t*photon_dir|² = R² for t > 0
        a = np.dot(photon_dir, photon_dir)
        b = 2*np.dot(origin, photon_dir)
        c = np.dot(origin, origin) - R_DET**2
        disc = b*b - 4*a*c
        if disc < 0:
            continue
        t = (-b + np.sqrt(disc)) / (2*a)
        if t < 0:
            t = (-b - np.sqrt(disc)) / (2*a)
        if t < 0:
            continue
        end = origin + t * photon_dir
        photons.append((origin, end))
    return photons

def gen_scint_photons(track_pts, n_photons=25):
    #Isotropic scintillation photons from random track points.
    photons = []
    idxs = np.random.choice(len(track_pts), min(n_photons, len(track_pts)),
                             replace=False)
    for i in idxs:
        origin = track_pts[i]
        dirn   = np.random.randn(3); dirn /= np.linalg.norm(dirn)
        a  = np.dot(dirn,dirn)
        b  = 2*np.dot(origin, dirn)
        c  = np.dot(origin,origin) - R_DET**2
        disc = b*b - 4*a*c
        if disc < 0:
            continue
        t = (-b + np.sqrt(disc)) / (2*a)
        if t < 0:
            continue
        end = origin + t*dirn
        photons.append((origin, end))
    return photons

def gen_wls_photons(scint_photons, n_wls=8):
    #bis-MSB re-emission: photon absorbed mid-path, re-emitted isotropically.
    wls = []
    chosen = np.random.choice(len(scint_photons), min(n_wls, len(scint_photons)),
                               replace=False)
    for i in chosen:
        orig, end = scint_photons[i]
        # Absorption point at ~30-60% of path
        frac = np.random.uniform(0.3, 0.6)
        abs_pt = orig + frac*(end - orig)
        # Re-emit in new direction (WLS shifts wavelength, re-emits isotropically)
        new_dir = np.random.randn(3); new_dir /= np.linalg.norm(new_dir)
        a  = np.dot(new_dir,new_dir)
        b  = 2*np.dot(abs_pt, new_dir)
        c  = np.dot(abs_pt,abs_pt) - R_DET**2
        disc = b*b - 4*a*c
        if disc < 0:
            continue
        t = (-b + np.sqrt(disc)) / (2*a)
        if t < 0:
            continue
        new_end = abs_pt + t*new_dir
        wls.append((orig, abs_pt, new_end))   # (primary, wls_point, secondary)
    return wls

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1 — 3D Geometry View
# ─────────────────────────────────────────────────────────────────────────────
def plot_3d_geometry(out_dir):
    fig = plt.figure(figsize=(16, 13))

    # ── Main 3D view ──────────────────────────────────────────────────────────
    ax = fig.add_axes([0.02, 0.15, 0.62, 0.80], projection='3d')
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.set_xlim(-R_DET*1.3, R_DET*1.3)
    ax.set_ylim(-R_DET*1.3, R_DET*1.3)
    ax.set_zlim(-R_DET*1.3, R_DET*1.3)
    ax.set_xlabel("x [cm]", color="white", labelpad=6)
    ax.set_ylabel("y [cm]", color="white", labelpad=6)
    ax.set_zlabel("z [cm]", color="white", labelpad=6)
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333333")
    ax.yaxis.pane.set_edgecolor("#333333")
    ax.zaxis.pane.set_edgecolor("#333333")
    ax.grid(True, color="#222222", alpha=0.4)

    # ── Sphere wireframe ─────────────────────────────────────────────────────
    sphere_wireframe(ax, R_DET, n_lines=22, color="#4a9eff", alpha=0.22, lw=0.7)

    # Transparent sphere shell
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0,   np.pi, 30)
    xs = R_DET * np.outer(np.cos(u), np.sin(v))
    ys = R_DET * np.outer(np.sin(u), np.sin(v))
    zs = R_DET * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.04, color="#4a9eff", shade=False,
                    linewidth=0)

    # ── PMT discs ─────────────────────────────────────────────────────────────
    for idx, (ax_vec, label, color) in enumerate(zip(PMT_AXES, PMT_LABELS, PMT_COLORS)):
        axv = np.array(ax_vec, dtype=float)
        pos = (R_DET - T_PMT) * axv         # PMT centre position
        inward = -axv                         # face points inward

        # Disc vertices
        verts = make_disc_verts(pos, inward, R_PMT, n_pts=50)
        poly = Poly3DCollection([verts], alpha=0.85,
                                 facecolor=color, edgecolor="gold",
                                 linewidth=1.2)
        ax.add_collection3d(poly)

        # Inward-pointing arrow (PMT normal → centre)
        arr_start = pos
        arr_end   = pos + inward * 3.5
        ax.quiver(*arr_start, *(arr_end - arr_start),
                  color="gold", alpha=0.85, lw=1.5,
                  arrow_length_ratio=0.3, length=1)

        # PMT label
        lbl_pos = axv * (R_DET + 2.0)
        ax.text(*lbl_pos, f"PMT {idx+1}\n({label})",
                color=color, fontsize=7.5, ha='center', va='center',
                fontweight='bold')

    # ── Generate tracks ───────────────────────────────────────────────────────
    e_start  = np.array([0., 0., 0.])
    e_dir    = np.array([0.55, 0.25, 0.80])
    e_dir   /= np.linalg.norm(e_dir)
    e_track  = gen_electron_track(e_start, e_dir, energy_MeV=5.0, n_steps=120)

    cer_phot = gen_cherenkov_photons(e_track, n_photons=20)
    sci_phot = gen_scint_photons(e_track,     n_photons=28)
    wls_phot = gen_wls_photons(sci_phot,      n_wls=9)

    # ── Draw electron track ───────────────────────────────────────────────────
    ax.plot(e_track[:,0], e_track[:,1], e_track[:,2],
            color="#00ff88", lw=2.2, alpha=0.95, zorder=10,
            label="Primary e⁻ (5 MeV)")
    # Highlight start
    ax.scatter(*e_start, color="lime", s=80, zorder=15, edgecolors="white", lw=1)

    # ── Draw Cherenkov photons ─────────────────────────────────────────────────
    for i, (orig, end) in enumerate(cer_phot):
        ax.plot([orig[0], end[0]], [orig[1], end[1]], [orig[2], end[2]],
                color="#60c0ff", alpha=0.50, lw=0.9,
                label="Cherenkov photon" if i == 0 else "")

    # ── Draw scintillation photons ────────────────────────────────────────────
    for i, (orig, end) in enumerate(sci_phot):
        ax.plot([orig[0], end[0]], [orig[1], end[1]], [orig[2], end[2]],
                color="#ffaa00", alpha=0.40, lw=0.8, ls="--",
                label="Scintillation (PPO)" if i == 0 else "")

    # ── Draw WLS photons (bis-MSB re-emission) ────────────────────────────────
    for i, (orig, abs_pt, new_end) in enumerate(wls_phot):
        # Primary segment (absorbed)
        ax.plot([orig[0], abs_pt[0]], [orig[1], abs_pt[1]], [orig[2], abs_pt[2]],
                color="#ff6b9d", alpha=0.30, lw=0.7, ls=":")
        # Re-emitted segment
        ax.plot([abs_pt[0], new_end[0]], [abs_pt[1], new_end[1]], [abs_pt[2], new_end[2]],
                color="#da70d6", alpha=0.60, lw=1.0,
                label="bis-MSB WLS re-emit" if i == 0 else "")
        # WLS node marker
        ax.scatter(*abs_pt, color="violet", s=18, alpha=0.8, zorder=8)

    # ── Draw detected photon "hits" on PMT faces ──────────────────────────────
    for orig, end in cer_phot[:6]:
        # Check if endpoint is near a PMT
        pt = end / np.linalg.norm(end) * R_DET
        ax.scatter(*end, color="#60c0ff", s=25, marker="*", alpha=0.9, zorder=12)

    # ── Coordinate axes ───────────────────────────────────────────────────────
    axis_len = R_DET * 1.15
    for vec, col, lbl in [([1,0,0],"#ff4444","X"),
                           ([0,1,0],"#44ff44","Y"),
                           ([0,0,1],"#4488ff","Z")]:
        ax.quiver(0,0,0, vec[0]*axis_len, vec[1]*axis_len, vec[2]*axis_len,
                  color=col, lw=2.0, arrow_length_ratio=0.08)
        ax.text(vec[0]*(axis_len+1.5), vec[1]*(axis_len+1.5),
                vec[2]*(axis_len+1.5), lbl, color=col, fontsize=11, fontweight='bold')

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title("NeutrinoDetSim — 30 cm Spherical Detector, 6 × 3\" PMTs\n"
                 "5 MeV e⁻ track with Cherenkov, Scintillation (PPO), WLS (bis-MSB)",
                 color="white", fontsize=11, pad=10)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color="#00ff88", label="Primary e⁻ track"),
        mpatches.Patch(color="#60c0ff", label="Cherenkov photons (UV/blue)"),
        mpatches.Patch(color="#ffaa00", label="Scintillation – PPO fast (~360 nm)"),
        mpatches.Patch(color="#da70d6", label="WLS re-emit – bis-MSB slow (~420 nm)"),
        mpatches.Patch(color="gold",    label="PMT face (3\" bialkali)"),
        plt.scatter([], [], color="lime",   s=40, label="Vertex (interaction pt)"),
        plt.scatter([], [], color="violet", s=18, label="WLS absorption node"),
        plt.scatter([], [], color="#60c0ff",marker="*", s=30, label="PMT hit"),
    ]
    ax.legend(handles=handles, loc="lower left",
              facecolor="#111111", edgecolor="#444444",
              labelcolor="white", fontsize=7.5, framealpha=0.85)

    ax.view_init(elev=22, azim=-55)

    # ── Right panel: 2D cross-section ─────────────────────────────────────────
    ax2 = fig.add_axes([0.66, 0.52, 0.32, 0.44])
    ax2.set_facecolor("#0d1117")
    ax2.set_aspect("equal")
    ax2.set_xlim(-R_DET*1.35, R_DET*1.35)
    ax2.set_ylim(-R_DET*1.35, R_DET*1.35)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444444")
    ax2.set_title("XZ Cross-Section", color="white", fontsize=9)
    ax2.set_xlabel("x [cm]", color="white", fontsize=8)
    ax2.set_ylabel("z [cm]", color="white", fontsize=8)

    # Sphere circle
    theta_arr = np.linspace(0, 2*np.pi, 300)
    ax2.plot(R_DET*np.cos(theta_arr), R_DET*np.sin(theta_arr),
             color="#4a9eff", lw=2.0, alpha=0.7)
    ax2.fill(R_DET*np.cos(theta_arr), R_DET*np.sin(theta_arr),
             alpha=0.05, color="#4a9eff")

    # PMTs in XZ plane: ±x and ±z
    xz_pmts = [(+R_DET, 0,"#e63946","+x"),(- R_DET, 0,"#457b9d","−x"),
               (0,+R_DET,"#a8dadc","+z"),(0,-R_DET,"#f4a261","−z")]
    for px, pz, col, lbl in xz_pmts:
        norm = np.array([-px, -pz])
        n = np.linalg.norm(norm)
        if n>0: norm /= n
        perp = np.array([-norm[1], norm[0]])
        p1 = np.array([px,pz]) + R_PMT*perp
        p2 = np.array([px,pz]) - R_PMT*perp
        ax2.plot([p1[0],p2[0]],[p1[1],p2[1]], color=col, lw=5, alpha=0.85,
                 solid_capstyle='round')
        ax2.annotate("", xy=np.array([px,pz])+norm*3.5,
                     xytext=np.array([px,pz]),
                     arrowprops=dict(arrowstyle="->", color="gold", lw=1.5))
        ax2.text(px*1.18, pz*1.18, lbl, color=col,
                 ha='center', va='center', fontsize=7.5, fontweight='bold')

    # Track in XZ (project e_track)
    ax2.plot(e_track[:,0], e_track[:,2],
             color="#00ff88", lw=2.0, alpha=0.9, label="e⁻ track")
    ax2.scatter(0, 0, color="lime", s=50, zorder=10)

    # Cherenkov photons in XZ
    for orig, end in cer_phot:
        ax2.plot([orig[0],end[0]],[orig[2],end[2]],
                 color="#60c0ff", alpha=0.40, lw=0.7)
    for orig, end in sci_phot[:12]:
        ax2.plot([orig[0],end[0]],[orig[2],end[2]],
                 color="#ffaa00", alpha=0.30, lw=0.6, ls="--")
    for _, abs_pt, new_end in wls_phot[:5]:
        ax2.plot([abs_pt[0],new_end[0]],[abs_pt[2],new_end[2]],
                 color="#da70d6", alpha=0.55, lw=0.9)
        ax2.scatter(abs_pt[0], abs_pt[2], color="violet", s=10, alpha=0.8)

    # Dimension annotation
    ax2.annotate("", xy=(R_DET, -R_DET*1.20), xytext=(-R_DET, -R_DET*1.20),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.2))
    ax2.text(0, -R_DET*1.28, "Ø = 30 cm", ha='center', color="white",
             fontsize=8, va='top')
    ax2.annotate("", xy=(R_DET*1.20, R_DET), xytext=(R_DET*1.20, -R_DET),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.2))
    ax2.text(R_DET*1.30, 0, "15 L", ha='left', color="white",
             fontsize=8, va='center', rotation=90)

    # ── Bottom right: PMT angular coverage schematic ──────────────────────────
    ax3 = fig.add_axes([0.66, 0.06, 0.32, 0.40])
    ax3.set_facecolor("#0d1117")
    ax3.set_aspect("equal")
    ax3.set_xlim(-1.5, 1.5); ax3.set_ylim(-1.5, 1.5)
    ax3.set_title("PMT Solid Angle Coverage", color="white", fontsize=9)
    ax3.axis("off")

    # Unit sphere
    circ = plt.Circle((0,0), 1.0, fill=False, color="#4a9eff", lw=1.5, alpha=0.6)
    ax3.add_patch(circ)

    pmt_angular = [( 1,0,"#e63946"),(- 1,0,"#457b9d"),(0, 1,"#2a9d8f"),
                   (0,-1,"#e9c46a"),(0.7,0.7,"#f4a261"),(-0.7,-0.7,"#a8dadc")]
    half_angle = np.degrees(np.arcsin(R_PMT / R_DET))  # ~14.7°

    for nx, nz, col in pmt_angular:
        n = np.sqrt(nx**2+nz**2)
        if n > 0: nx/=n; nz/=n
        # Wedge for PMT angular coverage
        ang = np.degrees(np.arctan2(nz, nx))
        wedge = Wedge((0,0), 1.0, ang-half_angle, ang+half_angle,
                      facecolor=col, alpha=0.35, edgecolor=col, lw=1.0)
        ax3.add_patch(wedge)
        ax3.annotate("", xy=(nx*1.0, nz*1.0), xytext=(nx*0.55, nz*0.55),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    ax3.text(0, 0, f"Coverage\n{6*(half_angle/90.*50):.0f}%\nsolid angle",
             ha='center', va='center', color="white", fontsize=8)
    ax3.text(0, -1.38, f"PMT half-angle = {half_angle:.1f}°  (R_PMT={R_PMT:.1f}cm / R_det={R_DET:.0f}cm)",
             ha='center', va='top', color="lightgrey", fontsize=7)

    # ── Global text annotations ───────────────────────────────────────────────
    fig.text(0.02, 0.06,
             "NeutrinoDetSim  ·  Geant4 Simulation\n"
             "Medium: D₂O + LAB + PPO (1 g/L) + bis-MSB (0.05 g/L)\n"
             "6 × Hamamatsu R7081 3\" PMTs  ·  QE ~ 30% @ 420 nm\n"
             "Cherenkov threshold in H₂O: 0.775 MeV  |  D₂O: 0.786 MeV",
             color="lightgrey", fontsize=8, va='bottom',
             bbox=dict(facecolor="#111111", edgecolor="#333333",
                       boxstyle="round,pad=0.4", alpha=0.8))

    import os
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, "geo_3d_detector.png")
    fig.savefig(fname, facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)
    print(f"  [✓] 3D geometry plot  → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2 — Detailed physics annotation panel
# ─────────────────────────────────────────────────────────────────────────────
def plot_physics_detail(out_dir):
    #Four-panel figure showing Cherenkov cone, track profile,
    #PPO/bis-MSB spectra, and PMT QE.

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flatten():
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # ─── Panel A: Cherenkov cone (2D schematic) ────────────────────────────
    axA = axes[0,0]
    axA.set_xlim(-1, 14); axA.set_ylim(-6, 6)
    axA.set_aspect("equal")
    axA.set_title("A  –  Cherenkov Radiation Cone (e⁻ in H₂O/D₂O)")
    axA.set_xlabel("Track direction [arb. units]")
    axA.axis("off")

    # Track line
    axA.annotate("", xy=(12,0), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color="#00ff88", lw=2.5,
                                mutation_scale=16))
    axA.text(12.2, 0.2, "e⁻  (β ≈ 0.94)", color="#00ff88", fontsize=9)

    # Cherenkov cone
    theta_c = np.radians(41.2)
    xs = np.array([0, 8])
    axA.fill_between(xs, xs*np.tan(theta_c), -xs*np.tan(theta_c),
                     alpha=0.15, color="#60c0ff")
    axA.plot([0,8], [0, 8*np.tan(theta_c)], color="#60c0ff", lw=2.0)
    axA.plot([0,8], [0,-8*np.tan(theta_c)], color="#60c0ff", lw=2.0)

    # Individual photon rays
    np.random.seed(7)
    for pt in np.linspace(1, 7, 8):
        phi = np.random.choice([1,-1]) * np.random.uniform(0.85,1.15)
        angle = phi * theta_c
        ray_len = 5.0
        ex = pt + ray_len*np.cos(angle)
        ey =      ray_len*np.sin(angle)
        axA.annotate("", xy=(ex,ey), xytext=(pt, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#60c0ff",
                                   lw=1.2, mutation_scale=10, alpha=0.7))

    # Angle annotation
    arc = Arc((0,0), 5, 5, angle=0,
              theta1=0, theta2=np.degrees(theta_c),
              color="yellow", lw=1.5)
    axA.add_patch(arc)
    axA.text(3.2, 0.9, f"θ_c = {np.degrees(theta_c):.1f}°\n(n=1.34, H₂O)",
             color="yellow", fontsize=8.5)
    axA.text(0.2, -5.2,
             "cos θ_c = 1/(βn)   |   Threshold: E_thr(e⁻) ≈ 0.78 MeV in H₂O",
             color="lightgrey", fontsize=8.5)
    axA.text(0.2, -4.5,
             "Cherenkov yield: dN/dE ~ 370 sin²θ_c photons/(eV·cm)",
             color="lightgrey", fontsize=8.5)

    # ─── Panel B: 2D track profile with Bragg-like energy loss ────────────
    axB = axes[0,1]
    axB.set_title("B  –  Track Length vs Photon Production Rate")
    axB.set_xlabel("Track position [cm]")
    axB.set_ylabel("dN_photon / ds  [arb. units]")

    s = np.linspace(0, 1, 300)  # normalised track coordinate
    # Bethe-Bloch-like Cherenkov: ~constant then drops at end of range
    cer_rate  = 1.0 * (1 - 0.3*s**2) * np.exp(-3.*(s-1.)**4)
    # Scintillation: peaks near end of track (Bragg peak)
    sci_rate  = 0.8 + 2.5 * np.exp(-50*(s-0.92)**2)
    # Total
    total_rate = cer_rate + sci_rate

    # Scale to actual range at 5 MeV (~2.5 cm in water)
    track_s = s * 2.5
    axB.fill_between(track_s, cer_rate,  alpha=0.4, color="#60c0ff", label="Cherenkov")
    axB.fill_between(track_s, sci_rate,  alpha=0.4, color="#ffaa00", label="Scintillation (PPO)")
    axB.plot(track_s, total_rate, color="white", lw=2.0, label="Total")
    axB.axvline(2.3, color="#00ff88", ls="--", lw=1.5, label="End of range")
    axB.text(2.32, 2.8, "Bragg\npeak", color="#ffaa00", fontsize=8)
    axB.legend(facecolor="#111111", edgecolor="#444444",
               labelcolor="white", fontsize=8)
    for spine in axB.spines.values():
        spine.set_color("#444444")

    # ─── Panel C: PPO + bis-MSB emission/absorption spectra ───────────────
    axC = axes[1,0]
    axC.set_title("C  –  PPO + bis-MSB (b-MIS) Spectral Chain")
    axC.set_xlabel("Wavelength [nm]")
    axC.set_ylabel("Normalised intensity")

    wl = np.linspace(260, 600, 500)

    def gauss(wl, mu, sig, amp=1.): return amp*np.exp(-0.5*((wl-mu)/sig)**2)

    ppo_abs  = gauss(wl,305,14) + 0.3*gauss(wl,330,18,0.5)
    ppo_emit = gauss(wl,360,20) + 0.4*gauss(wl,380,14,0.5)
    bms_abs  = gauss(wl,368,18)
    bms_emit = gauss(wl,420,22) + 0.28*gauss(wl,448,16,0.4)
    pmt_qe   = np.interp(wl,
                [270,290,310,330,350,370,390,410,420,440,460,480,500,520,540,560,580,600],
                [.05,.10,.17,.23,.27,.29,.30,.30,.30,.29,.26,.22,.17,.13,.09,.06,.04,.02])
    cher_spec = gauss(wl, 360, 55, 0.6)  # broad Cherenkov spectrum

    axC.fill_between(wl, ppo_abs,  alpha=0.25, color="#4488ff")
    axC.plot(wl, ppo_abs,  color="#4488ff", lw=1.8, ls="--", label="PPO abs. (~305 nm)")
    axC.fill_between(wl, ppo_emit, alpha=0.30, color="#00ccff")
    axC.plot(wl, ppo_emit, color="#00ccff", lw=2.0, label="PPO emit. (FAST, τ≈2 ns)")
    axC.fill_between(wl, bms_abs,  alpha=0.20, color="#ff8800")
    axC.plot(wl, bms_abs,  color="#ff8800", lw=1.8, ls="--", label="bis-MSB abs. (~368 nm)")
    axC.fill_between(wl, bms_emit, alpha=0.35, color="#ffcc00")
    axC.plot(wl, bms_emit, color="#ffcc00", lw=2.2, label="bis-MSB emit. (SLOW, τ≈1.5 ns)")
    axC.fill_between(wl, cher_spec,alpha=0.15, color="#60c0ff")
    axC.plot(wl, cher_spec,color="#60c0ff", lw=1.5, ls=":", label="Cherenkov spectrum")
    axC.plot(wl, pmt_qe,   color="white",   lw=2.0, ls="-.",label="PMT QE (bialkali)")

    # Transfer arrows
    axC.annotate("", xy=(368, 0.55), xytext=(360, 0.55),
                arrowprops=dict(arrowstyle="->", color="lime", lw=2))
    axC.text(350, 0.58, "WLS\ntransfer", color="lime", fontsize=7.5, ha='right')
    axC.axvspan(330, 415, alpha=0.06, color="lime")
    axC.text(370, 0.92, "Förster\nresonance\nwindow", color="lime",
             fontsize=7, ha='center')

    axC.set_xlim(260, 600)
    axC.legend(facecolor="#111111", edgecolor="#444444",
               labelcolor="white", fontsize=7.5, loc="upper right",
               framealpha=0.85)
    for spine in axC.spines.values():
        spine.set_color("#444444")

    # ─── Panel D: PMT hit timing — Cherenkov vs Scintillation ─────────────
    axD = axes[1,1]
    axD.set_title("D  –  Expected PMT Hit Timing Structure")
    axD.set_xlabel("Photon arrival time [ns]")
    axD.set_ylabel("Hits / ns  [arb. units]")

    t = np.linspace(0, 80, 800)
    # Cherenkov: prompt, spread by transit time in 30-cm sphere (~0.5 ns)
    cer_time = 5.0  * np.exp(-0.5*((t - 0.7)/0.4)**2)
    # PPO fast: τ_fast ~ 2.1 ns, rise ~0.5 ns
    ppo_time = 3.0  * (1 - np.exp(-t/0.5)) * np.exp(-t/2.1)
    # bis-MSB slow: τ_slow ~ 12 ns, rise ~1.5 ns
    bms_time = 1.5  * (1 - np.exp(-t/1.5)) * np.exp(-t/12.0)
    # Scattered / reflected late photons
    late     = 0.3  * np.exp(-t/25.)

    axD.fill_between(t, cer_time, alpha=0.45, color="#60c0ff", label="Cherenkov (prompt, t₀≈0.7 ns)")
    axD.fill_between(t, ppo_time, alpha=0.45, color="#00ffaa", label="PPO fast (τ=2.1 ns)")
    axD.fill_between(t, bms_time, alpha=0.45, color="#ffcc00", label="bis-MSB WLS (τ=12 ns)")
    axD.fill_between(t, late,     alpha=0.30, color="#888888", label="Scattered / reflected")
    axD.plot(t, cer_time+ppo_time+bms_time+late, color="white", lw=2.0, label="Total")

    # Time windows
    axD.axvspan(0, 5,  alpha=0.12, color="#60c0ff")
    axD.axvspan(5, 30, alpha=0.08, color="#ffcc00")
    axD.text(2.0, 7.5, "Early\n(Cherenkov)", color="#60c0ff", fontsize=7.5, ha='center')
    axD.text(15., 4.0, "Late\n(Scintillation)", color="#ffcc00", fontsize=7.5, ha='center')
    axD.axvline(5, color="white", ls=":", lw=1.0)

    axD.set_xlim(0, 80)
    axD.legend(facecolor="#111111", edgecolor="#444444",
               labelcolor="white", fontsize=8, framealpha=0.85)
    for spine in axD.spines.values():
        spine.set_color("#444444")

    fig.suptitle("NeutrinoDetSim — Optical Physics Detail\n"
                 "Cherenkov Cone · Track dE/dx · PPO+bis-MSB Chain · Timing Structure",
                 color="white", fontsize=12, y=1.01)
    fig.tight_layout()

    import os
    fname = os.path.join(out_dir, "geo_physics_detail.png")
    fig.savefig(fname, facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)
    print(f"  [✓] Physics detail plot → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3 — Top-down and side cross-sections with full PMT labels
# ─────────────────────────────────────────────────────────────────────────────
def plot_cross_sections(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor("#0d1117")
    titles = ["XY Plane (top view)", "XZ Plane (side view)", "YZ Plane (front view)"]
    plane_idx = [(0,1), (0,2), (1,2)]   # which dims to plot
    plane_pmts = [
        [(+1,0,0,"#e63946","+x"),(-1,0,0,"#457b9d","−x"),
         (0,+1,0,"#2a9d8f","+y"),(0,-1,0,"#e9c46a","−y")],
        [(+1,0,0,"#e63946","+x"),(-1,0,0,"#457b9d","−x"),
         (0,0,+1,"#a8dadc","+z"),(0,0,-1,"#f4a261","−z")],
        [(0,+1,0,"#2a9d8f","+y"),(0,-1,0,"#e9c46a","−y"),
         (0,0,+1,"#a8dadc","+z"),(0,0,-1,"#f4a261","−z")],
    ]

    e_start = np.array([0.,0.,0.])
    e_dir   = np.array([0.55, 0.25, 0.80]); e_dir /= np.linalg.norm(e_dir)
    e_track = gen_electron_track(e_start, e_dir, 5.0, 120)
    cer_ph  = gen_cherenkov_photons(e_track, 20)
    sci_ph  = gen_scint_photons(e_track, 22)
    wls_ph  = gen_wls_photons(sci_ph, 7)

    axis_lbl = ["x [cm]", "y [cm]", "z [cm]"]

    for col_idx, (ax, title, (d0, d1), pmts) in enumerate(
            zip(axes, titles, plane_idx, plane_pmts)):

        ax.set_facecolor("#0d1117")
        ax.set_aspect("equal")
        ax.set_xlim(-R_DET*1.35, R_DET*1.35)
        ax.set_ylim(-R_DET*1.35, R_DET*1.35)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444444")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel(axis_lbl[d0], color="white")
        ax.set_ylabel(axis_lbl[d1], color="white")
        ax.grid(True, color="#222222", alpha=0.4)

        # Sphere
        theta_arr = np.linspace(0, 2*np.pi, 300)
        ax.plot(R_DET*np.cos(theta_arr), R_DET*np.sin(theta_arr),
                color="#4a9eff", lw=2.0, alpha=0.7)
        ax.fill(R_DET*np.cos(theta_arr), R_DET*np.sin(theta_arr),
                alpha=0.05, color="#4a9eff")

        # PMTs in this plane
        for px,py,pz,col,lbl in pmts:
            pos_2d = np.array([px,py,pz], dtype=float)[[d0,d1]] * R_DET
            norm_2d = -np.array([px,py,pz], dtype=float)[[d0,d1]]
            n = np.linalg.norm(norm_2d)
            if n > 0: norm_2d = norm_2d / n
            perp_2d = np.array([-norm_2d[1], norm_2d[0]])
            p1 = pos_2d + R_PMT * perp_2d
            p2 = pos_2d - R_PMT * perp_2d
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],
                    color=col, lw=7, alpha=0.85, solid_capstyle='round')
            # Arrow (inward normal)
            arr_end = pos_2d + norm_2d * 3.5
            ax.annotate("", xy=arr_end, xytext=pos_2d,
                        arrowprops=dict(arrowstyle="->",color="gold",lw=1.5))
            ax.text(pos_2d[0]*1.22, pos_2d[1]*1.22, lbl,
                    color=col, fontsize=8, ha='center', va='center',
                    fontweight='bold')

        # Electron track
        ax.plot(e_track[:,d0], e_track[:,d1],
                color="#00ff88", lw=2.2, alpha=0.9, label="e⁻")
        ax.scatter(0, 0, color="lime", s=60, zorder=10)

        # Cherenkov
        for i,(orig,end) in enumerate(cer_ph):
            ax.plot([orig[d0],end[d0]],[orig[d1],end[d1]],
                    color="#60c0ff", alpha=0.45, lw=0.9,
                    label="Cherenkov" if i==0 else "")

        # Scintillation
        for i,(orig,end) in enumerate(sci_ph):
            ax.plot([orig[d0],end[d0]],[orig[d1],end[d1]],
                    color="#ffaa00", alpha=0.35, lw=0.7, ls="--",
                    label="Scintillation" if i==0 else "")

        # WLS
        for i,(_,abs_pt,new_end) in enumerate(wls_ph):
            ax.plot([abs_pt[d0],new_end[d0]],[abs_pt[d1],new_end[d1]],
                    color="#da70d6", alpha=0.60, lw=1.1,
                    label="WLS (bis-MSB)" if i==0 else "")
            ax.scatter(abs_pt[d0], abs_pt[d1],
                       color="violet", s=15, alpha=0.8, zorder=8)

        # Dimension arrow
        ax.annotate("", xy=(R_DET,-R_DET*1.22), xytext=(-R_DET,-R_DET*1.22),
                   arrowprops=dict(arrowstyle="<->",color="white",lw=1.2))
        ax.text(0,-R_DET*1.30,"Ø = 30 cm",ha='center',color="white",
                fontsize=8,va='top')

        if col_idx == 0:
            ax.legend(facecolor="#111111", edgecolor="#444444",
                      labelcolor="white", fontsize=7.5, loc="upper right")

    fig.suptitle("NeutrinoDetSim — Three Orthogonal Cross-Sections  |  5 MeV e⁻  |  D₂O+LAB+PPO+bis-MSB",
                 color="white", fontsize=11, y=1.02)
    fig.tight_layout()

    import os
    fname = os.path.join(out_dir, "geo_cross_sections.png")
    fig.savefig(fname, facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)
    print(f"  [✓] Cross-section plot  → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="NeutrinoDetSim geometry visualiser")
    parser.add_argument("--out-dir", default="./plots", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n NeutrinoDetSim Geometry Visualisation")
    print(f" Output: {args.out_dir}\n")

    plot_3d_geometry   (args.out_dir)
    plot_physics_detail(args.out_dir)
    plot_cross_sections(args.out_dir)

    print("\n Done. Three geometry figures written.\n")
"""
