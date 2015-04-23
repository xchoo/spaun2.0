# import time
import numpy as np
import numpy.ma as ma

from matplotlib_animation.matplotlib_anim import MatplotlibAnim


class NengoAnim(MatplotlibAnim):
    def __init__(self, data_gen_func, subplot_shape, fig=None):
        super(NengoAnim, self).__init__(data_gen_func, subplot_shape, fig)

    # ----------------------------------------------------------------------- #
    def _add_rasterplot_common(self, key, tl_loc, br_loc, n_neurons,
                               buffer_size_s, style, spike_disp_height,
                               spike_dot_size, dt, ax, plot_type, **plot_args):
        plot_args.setdefault('color', 'k')

        if style in ['.', 'dot']:
            dims = 1
        else:
            dims = n_neurons

        ax = self._add_plot_common(key, tl_loc, br_loc, dims,
                                   buffer_size_s, False,
                                   plot_type, ax, **plot_args)
        self.subplot_data[key][plot_type]['style'] = style
        self.subplot_data[key][plot_type]['spike_disp_height'] = \
            spike_disp_height
        self.subplot_data[key][plot_type]['spike_value'] = 1. / dt
        self.subplot_data[key][plot_type]['buffer_len'] = buffer_size_s / dt
        self.subplot_data[key][plot_type]['n_neurons'] = n_neurons
        ax.set_ylim(0.5, n_neurons + 0.5)

        if style in ['.', 'dot']:
            for line in self.subplot_data[key][plot_type]['lines']:
                line.set_marker('.')
                line.set_markersize(spike_dot_size)

        # self.subplot_data[key][plot_type]['time'] = 0
        return ax

    def _rasterplot_common_update(self, key, t_data, data, plot_type,
                                  draw_artists):
        subplot_data = self.subplot_data[key][plot_type]

        # print time.time() - subplot_data['time']
        # subplot_data['time'] = time.time()

        n_neurons = subplot_data['n_neurons']
        style = subplot_data['style']
        s_h = subplot_data['spike_disp_height'] / 2.0

        max_val = subplot_data['spike_value']
        t_len = t_data.shape[0]

        if style in ['.', 'dot']:
            tdata = t_data

            spike_heights = np.array([[n] * t_len for n in range(n_neurons)])

            sdata = ma.array(data[:n_neurons, :].flatten())
            sdata[sdata == 0] = ma.masked
            sdata[0::t_len] = ma.masked
            sdata = np.multiply(sdata, spike_heights.flatten()) / max_val

            tdata = np.tile(tdata, n_neurons)

            subplot_data['lines'][0].set_data(tdata, sdata)
            draw_artists.append(subplot_data['lines'][0])
        else:
            # tdata = t_data.repeat(3)

            # spike_heights = ma.array([[(1 + n - s_h), (1 + n + s_h), ma.masked]
            #                           * t_len for n in range(n_neurons)])

            # sdata = ma.array(data[:n_neurons, :].repeat(3).flatten())
            # sdata = np.multiply(sdata, spike_heights.flatten()) / max_val

            tdata = t_data.repeat(3)
            for n in range(n_neurons):
                sdata = ma.array((data[n]).repeat(3))
                sdata[0::3] *= (1 + n - s_h) / max_val
                sdata[1::3] *= (1 + n + s_h) / max_val
                sdata[2::3] = ma.masked
                subplot_data['lines'][n].set_data(tdata, sdata)
                draw_artists.append(subplot_data['lines'][n])

    # ----------------------------------------------------------------------- #
    def add_rasterplot_static_x(self, key, dt, tl_loc=(0, 0), br_loc=None,
                                n_neurons=10, buffer_size_s=0.5, style='|',
                                spike_disp_height=0.75, spike_dot_size=2,
                                ax=None, **plot_args):
        return self._add_rasterplot_common(key, tl_loc, br_loc, n_neurons,
                                           buffer_size_s, style,
                                           spike_disp_height, spike_dot_size,
                                           dt, ax, 'rasterplot_static_x',
                                           **plot_args)

    def rasterplot_static_x_update(self, key, t_data, data, draw_artists):
        subplot_data = self.subplot_data[key]['rasterplot_static_x']

        ax = subplot_data['ax']
        num_ticks = len(ax.xaxis.get_major_ticks())

        buffer_size = subplot_data['buffer_size']

        t_min = max(t_data[0], t_data[-1] - buffer_size)
        t_max = max(ax.get_xlim()[1], t_data[-1])
        new_tick_labels = np.linspace(t_min, t_max, num=num_ticks,
                                      endpoint=True)
        ax.xaxis.set_ticklabels(map(lambda t: "%0.2f" % t, new_tick_labels))

        t_indexes = subplot_data['buffer_len']
        self._rasterplot_common_update(key, t_data[-t_indexes:] - t_min,
                                       data[:, -t_indexes:],
                                       'rasterplot_static_x', draw_artists)

    # ----------------------------------------------------------------------- #
    def add_rasterplot_scroll_x(self, key, dt, tl_loc=(0, 0), br_loc=None,
                                n_neurons=10, buffer_size_s=0.5, style='|',
                                spike_disp_height=0.75, spike_dot_size=2,
                                ax=None, **plot_args):
        return self._add_rasterplot_common(key, tl_loc, br_loc, n_neurons,
                                           buffer_size_s, style,
                                           spike_disp_height, spike_dot_size,
                                           dt, ax, 'rasterplot_scroll_x',
                                           **plot_args)

    def rasterplot_scroll_x_update(self, key, t_data, data, draw_artists):
        subplot_data = self.subplot_data[key]['rasterplot_scroll_x']

        ax = subplot_data['ax']

        buffer_size = subplot_data['buffer_size']

        t_min = max(t_data[0], t_data[-1] - buffer_size)
        t_max = max(ax.get_xlim()[1], t_data[-1])

        ax.set_xlim(t_min, t_max)
        ax.figure.canvas.draw()

        t_indexes = subplot_data['buffer_len']
        self._rasterplot_common_update(key, t_data[-t_indexes:],
                                       data[:, -t_indexes:],
                                       'rasterplot_scroll_x', draw_artists)
