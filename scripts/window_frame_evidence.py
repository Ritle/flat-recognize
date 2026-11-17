"""Raster evidence for narrow window frames on an already supported wall gap.

Parallel rails alone do not establish a window: callers must also establish
the segmentation candidate, both wall jambs and a free structural span.
"""
import math

import cv2
import numpy as np


def narrow_frame_rails(gray, start, end, width):
    """Measure thin parallel dark bands with clear space between outer rails.

    Sample in the local axis frame, so vertical and diagonal frames use the
    same test. Avoid the jambs and never sample outside the source raster.
    """
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(b-a))
    if not math.isfinite(width) or width <= 0 or not math.isfinite(length) or length < 8:
        return None
    direction = (b-a)/length
    normal = np.array([-direction[1], direction[0]])
    along = np.linspace(length*.15, length*.85, max(6, math.ceil(length*.7)))
    # A one-pixel antialiased rail can disappear between integer-offset probes
    # when the model axis lies at a fractional pixel. Sample the normal at half
    # pixel spacing, retaining all width limits in physical source pixels.
    step = .5
    offsets = np.arange(-math.ceil(width/2+2), math.ceil(width/2+2)+step, step, dtype=float)
    points = a + along[None, :, None]*direction + offsets[:, None, None]*normal
    height, image_width = gray.shape
    if (points[..., 0].min() < 0 or points[..., 0].max() > image_width-1
            or points[..., 1].min() < 0 or points[..., 1].max() > height-1):
        return None
    strip = cv2.remap(gray, points[..., 0].astype(np.float32), points[..., 1].astype(np.float32),
                      cv2.INTER_LINEAR)
    support = (strip < 140).mean(axis=1)
    rows = np.flatnonzero(support >= .8)
    bands = [group for group in np.split(rows, np.flatnonzero(np.diff(rows) > 1)+1) if len(group)]
    if not 2 <= len(bands) <= 4 or any(len(group)*step > max(2, width*.25) for group in bands):
        return None
    first, last = bands[0], bands[-1]
    separation = float(offsets[last].mean()-offsets[first].mean())
    if not 2 <= separation <= width+4 or not offsets[first].mean() < 0 < offsets[last].mean():
        return None
    between = np.arange(first[-1]+1, last[0])
    # Intermediate thin rails are allowed; a solid dark strip is not a frame.
    if len(between) < 1 or float(np.mean(support[between])) > .35:
        return None
    return dict(method='narrow_parallel_raster_rails',
                rail_offsets_px=[round(float(offsets[group].mean()),3) for group in bands],
                rail_support=[round(float(support[group].max()),3) for group in bands],
                inner_dark_fraction=round(float(np.mean(support[between])),3),
                sampled_length_px=round(float(along[-1]-along[0]),3))


def bounded_frame_bay(gray, start, end, width, carrier_span):
    """Measure a short window between dark raster jambs, inside graph tips.

    The candidate supplies location, not the final aperture width. Graph tips
    can retreat into the jambs; keep the measured clear span separate from the
    longer carrier span needed to join the wall axes.
    """
    frame = narrow_frame_rails(gray,start,end,width)
    if frame is None:
        return None
    a,b = np.asarray(start,float),np.asarray(end,float)
    length = float(np.linalg.norm(b-a))
    axis = (b-a)/length
    normal = np.array([-axis[1],axis[0]])
    limits = sorted(float((np.asarray(p)-a) @ axis) for p in carrier_span)
    along = np.arange(math.ceil(limits[0]),math.floor(limits[1])+1,dtype=float)
    rails = frame['rail_offsets_px']
    # Keep a full source pixel away from both rail centers; their antialiased
    # edges must not be mistaken for a jamb inside the clear aperture.
    offsets = np.arange(math.ceil(rails[0]+1),math.floor(rails[-1]-1)+1,dtype=float)
    if len(along)<3 or not len(offsets):
        return None
    points = a + along[None,:,None]*axis + offsets[:,None,None]*normal
    if (points[...,0].min()<0 or points[...,0].max()>gray.shape[1]-1
            or points[...,1].min()<0 or points[...,1].max()>gray.shape[0]-1):
        return None
    strip = cv2.remap(gray,points[...,0].astype(np.float32),points[...,1].astype(np.float32),
                      cv2.INTER_LINEAR)
    darkness = (strip<140).mean(axis=0)
    clear = np.flatnonzero(darkness<.25)
    options = []
    for group in np.split(clear,np.flatnonzero(np.diff(clear)>1)+1):
        if len(group)<max(3,length*.6):
            continue
        left,right = int(group[0]),int(group[-1])
        if left==0 or right==len(along)-1 or not along[left]-.5<=length/2<=along[right]+.5:
            continue
        jambs = [float(darkness[max(0,left-3):left].max()),
                 float(darkness[right+1:min(len(along),right+4)].max())]
        if min(jambs)<.8:
            continue
        center_offset = (rails[0]+rails[-1])/2
        span = [(a+t*axis+center_offset*normal).tolist() for t in (along[left]-.5,along[right]+.5)]
        options.append(dict(clear_span=span,raster_jamb_support=jambs,frame=frame))
    return options[0] if len(options)==1 else None
