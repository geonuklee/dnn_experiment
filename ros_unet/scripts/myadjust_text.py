#!/usr/bin/python2
#-*- coding:utf-8 -*-
from adjustText import *

"""
    adjustText.adjust_text() -> myadjust_text()
    차이점 : arrowprops 의 화살표끝이 bounding box 안에 있을 경우 생략하는 기능 추가.
"""

# intersection between line(p1, p2) and line(p3, p4)
def intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)

def get_boundpoint(bbox, xy):
    cx = (bbox.x0+bbox.x1)/2.
    cy = (bbox.y0+bbox.y1)/2.
    dx = xy[0]-cx
    dy = xy[1]-cy
    dxy = (dx,dy)
    rad = np.arctan2(dy,dx)
    r = 20.
    ddxy = (r*np.cos(rad), r*np.sin(rad) )
    l0 = np.linalg.norm(dxy)

    assert(bbox.x0 < bbox.x1)
    assert(bbox.y0 < bbox.y1)
    bb = []
    bb.append( (bbox.x0-cx,bbox.y1-cy) )
    bb.append( (bbox.x1-cx,bbox.y1-cy) )
    bb.append( (bbox.x1-cx,bbox.y0-cy) )
    bb.append( (bbox.x0-cx,bbox.y0-cy) )

    for i in range(4):
        b0, b1 = bb[i], bb[(i+1)%4]
        ipt = intersect((0,0), ddxy, b0,b1)
        if ipt is None:
            continue
        l1 = 1.05*np.linalg.norm(ipt)
        l = max(l0,l1)
        return (ipt[0]+cx, ipt[1]+cy), (l*np.cos(rad)+cx, l*np.sin(rad)+cy )

    return None, None

def myadjust_text(texts, x=None, y=None, add_objects=None, ax=None,
                expand_text=(1.05, 1.2), expand_points=(1.05, 1.2),
                expand_objects=(1.05, 1.2), expand_align=(1.05, 1.2),
                autoalign='xy',  va='center', ha='center',
                force_text=(0.1, 0.25), force_points=(0.2, 0.5),
                force_objects=(0.1, 0.25),
                lim=500, precision=0.01,
                only_move={'points':'xy', 'text':'xy', 'objects':'xy'},
                text_from_text=True, text_from_points=True,
                save_steps=False, save_prefix='', save_format='png',
                add_step_numbers=True, on_basemap=False,
                *args, **kwargs):
    """Iteratively adjusts the locations of texts.
    """
    plt.draw()
    if ax is None:
        ax = plt.gca()
    r = get_renderer(ax.get_figure())
    orig_xy = [get_text_position(text, ax=ax) for text in texts]
    orig_x = [xy[0] for xy in orig_xy]
    orig_y = [xy[1] for xy in orig_xy]
    force_objects = float_to_tuple(force_objects)
    force_text = float_to_tuple(force_text)
    force_points = float_to_tuple(force_points)

    bboxes = get_bboxes(texts, r, (1.0, 1.0), ax)
    sum_width = np.sum(list(map(lambda bbox: bbox.width, bboxes)))
    sum_height = np.sum(list(map(lambda bbox: bbox.height, bboxes)))
    if not any(list(map(lambda val: 'x' in val, only_move.values()))):
        precision_x = np.inf
    else:
        precision_x = precision*sum_width
#
    if not any(list(map(lambda val: 'y' in val, only_move.values()))):
        precision_y = np.inf
    else:
        precision_y = precision*sum_height

    if x is None:
        if y is None:
            x, y = orig_x, orig_y
        else:
            raise ValueError('Please specify both x and y, or neither')
    if y is None:
        raise ValueError('Please specify both x and y, or neither')
    if add_objects is None:
        text_from_objects = False
        add_bboxes = []
    else:
        try:
            add_bboxes = get_bboxes(add_objects, r, (1, 1), ax)
        except:
            raise ValueError("Can't get bounding boxes from add_objects - is'\
                             it a flat list of matplotlib objects?")
            return
        text_from_objects = True
    for text in texts:
        text.set_va(va)
        text.set_ha(ha)
    if save_steps:
        if add_step_numbers:
            plt.title('Before')
        plt.savefig('%s%s.%s' % (save_prefix,
                            '000a', save_format), format=save_format, dpi=150)
    elif on_basemap:
        ax.draw(r)

    if autoalign:
        if autoalign is True:
            autoalign='xy'
        for i in range(2):
            texts = optimally_align_text(x, y, texts, expand=expand_align,
                                         add_bboxes=add_bboxes,
                                         direction=autoalign, renderer=r,
                                         ax=ax)

    if save_steps:
        if add_step_numbers:
            plt.title('Autoaligned')
        plt.savefig('%s%s.%s' % (save_prefix,
                            '000b', save_format), format=save_format, dpi=150)
    elif on_basemap:
        ax.draw(r)

    texts = repel_text_from_axes(texts, ax, renderer=r, expand=expand_points)
    history = [(np.inf, np.inf)]*10
    for i in xrange(lim):
#        q1, q2 = [np.inf, np.inf], [np.inf, np.inf]

        if text_from_text:
            d_x_text, d_y_text, q1 = repel_text(texts, renderer=r, ax=ax,
                                                expand=expand_text)
        else:
            d_x_text, d_y_text, q1 = [0]*len(texts), [0]*len(texts), (0, 0)

        if text_from_points:
            d_x_points, d_y_points, q2 = repel_text_from_points(x, y, texts,
                                                   ax=ax, renderer=r,
                                                   expand=expand_points)
        else:
            d_x_points, d_y_points, q2 = [0]*len(texts), [0]*len(texts), (0, 0)

        if text_from_objects:
            d_x_objects, d_y_objects, q3 = repel_text_from_bboxes(add_bboxes,
                                                                  texts,
                                                             ax=ax, renderer=r,
                                                         expand=expand_objects)
        else:
            d_x_objects, d_y_objects, q3 = [0]*len(texts), [0]*len(texts), (0, 0)

        if only_move:
            if 'text' in only_move:
                if 'x' not in only_move['text']:
                    d_x_text = np.zeros_like(d_x_text)
                if 'y' not in only_move['text']:
                    d_y_text = np.zeros_like(d_y_text)
            if 'points' in only_move:
                if 'x' not in only_move['points']:
                    d_x_points = np.zeros_like(d_x_points)
                if 'y' not in only_move['points']:
                    d_y_points = np.zeros_like(d_y_points)
            if 'objects' in only_move:
                if 'x' not in only_move['objects']:
                    d_x_objects = np.zeros_like(d_x_objects)
                if 'y' not in only_move['objects']:
                    d_y_objects = np.zeros_like(d_y_objects)

        dx = (np.array(d_x_text) * force_text[0] +
              np.array(d_x_points) * force_points[0] +
              np.array(d_x_objects) * force_objects[0])
        dy = (np.array(d_y_text) * force_text[1] +
              np.array(d_y_points) * force_points[1] +
              np.array(d_y_objects) * force_objects[1])
        qx = np.sum([q[0] for q in [q1, q2, q3]])
        qy = np.sum([q[1] for q in [q1, q2, q3]])
        histm = np.max(np.array(history), axis=0)
        history.pop(0)
        history.append((qx, qy))
        move_texts(texts, dx, dy,
                   bboxes = get_bboxes(texts, r, (1, 1), ax), ax=ax)
        if save_steps:
            if add_step_numbers:
                plt.title(i+1)
            plt.savefig('%s%s.%s' % (save_prefix,
                        '{0:03}'.format(i+1), save_format),
                        format=save_format, dpi=150)
        elif on_basemap:
            ax.draw(r)
        # Stop if we've reached the precision threshold, or if the x and y displacement
        # are both greater than the max over the last 10 iterations (suggesting a
        # failure to converge)
        if (qx < precision_x and qy < precision_y) or np.all([qx, qy] >= histm):
            break
        # Now adding arrows from texts to their original locations if required
    if 'arrowprops' in kwargs:
        mx = .02 # Margin in text box
        my = .02
        bboxes = get_bboxes(texts, r, (1, 1), ax)
        kwap = kwargs.pop('arrowprops')
        for j, (bbox, text) in enumerate(zip(bboxes, texts)):
            cpx, cpy =  orig_xy[j]
            xy = (orig_xy[j])
            xytext, xy = get_boundpoint(bbox, xy) # TODO midpoint가 아니라 bound point
            if xy is None:
                continue
            if xy[0] > bbox.x0-mx and xy[0] < bbox.x1+mx and xy[1] > bbox.y0-my and xy[1] < bbox.y1+my:
                continue
            ap = {'patchA':text} # Ensure arrow is clipped by the text
            ap.update(kwap) # Add arrowprops from kwargs
            ax.annotate("", # Add an arrow from the text to the point
                        xy = xy,
                        xytext=xytext,
                        arrowprops=ap,
                        *args, **kwargs)

    if save_steps:
        if add_step_numbers:
            plt.title(i+1)
            plt.savefig('%s%s.%s' % (save_prefix,
                        '{0:03}'.format(i+1), save_format),
                        format=save_format, dpi=150)
    elif on_basemap:
        ax.draw(r)

    return i+1


