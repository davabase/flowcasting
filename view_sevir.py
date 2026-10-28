"""Browse SEVIR frames interactively, or export a selected grid to PNG."""
import argparse
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

DEFAULT_FILE = Path(__file__).resolve().parent / 'data/sevir/data/processed/nowcast_testing_000.h5'


def select_frames(text, count):
    """Accept comma-separated indices and Python-style start:stop[:step] ranges."""
    result = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            fields = item.split(':')
            if len(fields) not in (2, 3):
                raise ValueError('Use start:stop[:step] for a frame range.')
            values = [int(value) if value else None for value in fields]
            if any(value is not None and value < 0 for value in values[:2]):
                raise ValueError('Frame indices must be nonnegative.')
            if len(values) == 3 and values[2] is not None and values[2] <= 0:
                raise ValueError('Frame step must be positive.')
            result.extend(range(count)[slice(*values)])
        else:
            result.append(int(item))
    if not result or any(index < 0 or index >= count for index in result):
        raise ValueError(f'Choose frame indices between 0 and {count - 1}.')
    if len(result) > 64:
        raise ValueError('Display at most 64 frames at a time.')
    return result


class RadarViewer:
    def __init__(self, source, dataset, sample, frames, columns):
        self.source = source
        self.dataset = dataset
        self.sample = sample
        self.frames = frames
        self.columns = columns
        self.figure = plt.figure(figsize=(13, 8))
        self.axes = []
        self.colorbar = None

    def draw(self):
        data = self.source[self.dataset]
        if not 0 <= self.sample < data.shape[0]:
            raise ValueError(f'Example must be between 0 and {data.shape[0] - 1}.')
        indices = select_frames(self.frames, data.shape[-1])
        # Validate before replacing the existing plot. Read only the selected frames.
        images = [data[self.sample, :, :, frame] for frame in indices]
        if self.colorbar is not None:
            self.colorbar.remove()
        for axis in self.axes:
            axis.remove()
        self.axes = []
        cols = min(self.columns, len(indices))
        rows = math.ceil(len(indices) / cols)
        grid = self.figure.add_gridspec(rows, cols, left=.03, right=.89,
                                        bottom=.22, top=.88, wspace=.06, hspace=.25)
        cmap = plt.get_cmap('turbo').copy()
        cmap.set_bad('#eeeeee')
        radar = self.dataset.lower().endswith('vil')
        low = 0 if radar else min(float(p.min()) for p in images)
        high = 255 if radar else max(float(p.max()) for p in images)
        if high == low:
            high = low + 1
        for slot, (frame, pixels) in enumerate(zip(indices, images)):
            axis = self.figure.add_subplot(grid[slot // cols, slot % cols])
            image = axis.imshow(pixels, cmap=cmap,
                                vmin=low, vmax=high, interpolation='nearest')
            axis.set_title(f'Frame {frame}', fontsize=10)
            axis.set_axis_off()
            self.axes.append(axis)
        color_axis = self.figure.add_axes([.92, .28, .015, .50])
        self.colorbar = self.figure.colorbar(image, cax=color_axis)
        self.colorbar.set_label('SEVIR encoded value (raw)')
        self.figure.suptitle(f'SEVIR · {self.dataset} · example {self.sample}\n'
                             f'{len(indices)} selected frames · {data.shape[1]} × {data.shape[2]} pixels')
        self.figure.canvas.draw_idle()

    def interactive(self):
        self.draw()
        self.example_box = TextBox(self.figure.add_axes([.09, .12, .12, .04]), 'Example ', initial=str(self.sample))
        self.dataset_box = TextBox(self.figure.add_axes([.34, .12, .15, .04]), 'Dataset ', initial=self.dataset)
        self.frames_box = TextBox(self.figure.add_axes([.63, .12, .29, .04]), 'Frames ', initial=self.frames)
        self.count_box = TextBox(self.figure.add_axes([.09, .055, .12, .04]), 'N frames ', initial='6')
        self.first_button = Button(self.figure.add_axes([.25, .055, .14, .04]), 'Show first N')
        self.apply_button = Button(self.figure.add_axes([.43, .055, .14, .04]), 'Apply selection')
        self.save_button = Button(self.figure.add_axes([.61, .055, .14, .04]), 'Save PNG')
        self.status = self.figure.text(.03, .015, 'Indices start at 0. Frames: 0,3,8 or 0:12:2. Range stop is exclusive.', fontsize=9)
        self.first_button.on_clicked(lambda event: self.apply(first_n=True))
        self.apply_button.on_clicked(lambda event: self.apply())
        self.save_button.on_clicked(self.save)
        plt.show()

    def apply(self, first_n=False):
        previous = self.dataset, self.sample, self.frames
        try:
            key = self.dataset_box.text.strip()
            if key not in self.source or not isinstance(self.source[key], h5py.Dataset) or self.source[key].ndim != 4:
                raise ValueError('Choose an available 4D dataset: ' + ', '.join(self.source.keys()))
            self.dataset = key
            self.sample = int(self.example_box.text)
            self.frames = self.frames_box.text
            if first_n:
                count = int(self.count_box.text)
                if not 1 <= count <= min(64, self.source[key].shape[-1]):
                    raise ValueError(f'N must be between 1 and {min(64, self.source[key].shape[-1])}.')
                self.frames = f'0:{count}'
            self.draw()
            self.frames_box.set_val(self.frames)
            self.status.set_text('Selection updated. Save PNG exports this view to renders/.')
        except (ValueError, KeyError) as error:
            self.dataset, self.sample, self.frames = previous
            self.status.set_text(str(error))
        self.figure.canvas.draw_idle()

    def save(self, event=None):
        directory = Path('renders')
        directory.mkdir(exist_ok=True)
        path = directory / f'sevir_{self.dataset}_{self.sample}.png'
        self.figure.savefig(path, dpi=150)
        self.status.set_text(f'Saved {path.resolve()}')
        self.figure.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, default=DEFAULT_FILE)
    parser.add_argument('--dataset', help='HDF5 dataset key (default: first 4D dataset)')
    parser.add_argument('--sample', type=int, default=0, help='Zero-based storm example index')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('-n', type=int, default=None, help='Show the first N frames')
    selection.add_argument('--frames', help='Frame indices, e.g. 0,3,8 or 0:12:2')
    parser.add_argument('--columns', type=int, default=4)
    parser.add_argument('--output', type=Path, help='Save a PNG grid without opening a window')
    args = parser.parse_args()
    if args.columns < 1 or args.columns > 16:
        parser.error('--columns must be between 1 and 16')
    if args.output:
        plt.switch_backend('Agg')
    try:
        with h5py.File(args.file, 'r') as source:
            keys = [key for key in source if isinstance(source[key], h5py.Dataset) and source[key].ndim == 4]
            if not keys:
                raise ValueError('No datasets with shape (examples, height, width, frames) found.')
            key = args.dataset or ('IN_vil' if 'IN_vil' in keys else keys[0])
            if key not in keys:
                raise ValueError(f'Choose a dataset from: {keys}')
            print('Available datasets:', ', '.join(f'{k}: {source[k].shape}' for k in keys))
            count = args.n if args.n is not None else min(6, source[key].shape[-1])
            if args.frames is None and not 1 <= count <= min(64, source[key].shape[-1]):
                raise ValueError(f'N must be between 1 and {min(64, source[key].shape[-1])}.')
            viewer = RadarViewer(source, key, args.sample, args.frames or f'0:{count}', args.columns)
            if args.output:
                viewer.draw()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                viewer.figure.savefig(args.output, dpi=150)
                plt.close(viewer.figure)
                print(f'Saved {args.output.resolve()}')
            else:
                viewer.interactive()
    except (OSError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
