"""Validate publication-size text bounds without changing scientific content."""
from pathlib import Path
import json
from matplotlib.text import Text


def check_publication_layout(fig, directory):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    entries = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        entries.append({'text': artist.get_text(), 'font_size_pt': artist.get_fontsize(),
                        'bounds': list(box.bounds),
                        'inside_canvas': bool(box.x0 >= -.5 and box.y0 >= -.5 and
                        box.x1 <= fig.bbox.width+.5 and box.y1 <= fig.bbox.height+.5)})
    axes = list(fig.axes)
    for ax in fig.axes:
        axes.extend(ax.child_axes)
    ticks = [text for ax in axes for text in [*ax.get_xticklabels(), *ax.get_yticklabels()]
             if text.get_visible() and text.get_text().strip()]
    boxes = [text.get_window_extent(renderer) for text in ticks]
    overlaps = [[ticks[i].get_text(), ticks[j].get_text()] for i in range(len(boxes))
                for j in range(i+1, len(boxes)) if boxes[i].overlaps(boxes[j])]
    result = {'figure_size_mm': [v*25.4 for v in fig.get_size_inches()],
              'minimum_font_size_pt': min(e['font_size_pt'] for e in entries),
              'clipped_text': [e['text'] for e in entries if not e['inside_canvas']],
              'coordinate_tick_overlaps': overlaps, 'text': entries}
    path = Path(directory)/'publication_layout.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    assert not result['clipped_text'], result['clipped_text']
    assert not overlaps, overlaps
    return result
