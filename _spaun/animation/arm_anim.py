from matplotlib_animation.matplotlib_anim import MatplotlibAnim


class ArmAnim(MatplotlibAnim):
    def __init__(self, data_gen_func, subplot_shape, fig=None):
        super(ArmAnim, self).__init__(data_gen_func, subplot_shape, fig)

    def add_arm_path_plot(self, key, tl_loc=(0, 0), br_loc=None,
                          xlim=(-1, 1), ylim=(-1, 1),
                          ee_point_down_plot_args={}, ee_point_up_plot_args={},
                          ee_down_plot_args={}, ee_up_plot_args={},
                          target_point_plot_args={}, target_down_plot_args={},
                          target_up_plot_args={}, arm_plot_args={},
                          show_tick_labels=False, ax=None):
        plot_type = 'arm_path_plot'

        if ax is None:
            ax = self.subplot_init(key, tl_loc, br_loc, plot_type)
        else:
            self.subplot_key_check(key, plot_type)
            self.subplot_data[key][plot_type]['ax'] = ax

        subplot_data = self.subplot_data[key][plot_type]
        subplot_data['ee_down_data'] = [[], []]
        subplot_data['ee_up_data'] = [[], []]
        subplot_data['ee_point'] = []
        subplot_data['target_down_data'] = [[], []]
        subplot_data['target_up_data'] = [[], []]
        subplot_data['target_point'] = []
        subplot_data['last_up_down_state'] = 0  # 0: up, 1: down
        subplot_data['last_t'] = 0

        ee_point_down_plot_args.setdefault('color', '#0000DD')
        ee_point_down_plot_args.setdefault('marker', 'x')
        ee_point_down_plot_args.setdefault('mew', '4')
        ee_point_up_plot_args.setdefault('color', '#AAAADD')
        ee_point_up_plot_args.setdefault('marker', 'x')
        ee_point_up_plot_args.setdefault('mew', '2')
        ee_down_plot_args.setdefault('color', '#0000FF')
        ee_up_plot_args.setdefault('color', '#AAAAFF')

        target_point_plot_args.setdefault('color', '#DD0000')
        target_point_plot_args.setdefault('marker', 'o')
        target_down_plot_args.setdefault('color', '#FF0000')
        target_up_plot_args.setdefault('color', '#FFAAAA')

        arm_plot_args.setdefault('color', '#00DD00')
        arm_plot_args.setdefault('marker', 'o')
        arm_plot_args.setdefault('mew', '2')

        target_down_line, = ax.plot([], [], **target_down_plot_args)
        target_up_line, = ax.plot([], [], **target_up_plot_args)
        target_point_line, = ax.plot([], [], **target_point_plot_args)
        arm_line, = ax.plot([], [], **arm_plot_args)
        ee_down_line, = ax.plot([], [], **ee_down_plot_args)
        ee_up_line, =  ax.plot([], [], **ee_up_plot_args)
        ee_point_down_line, = ax.plot([], [], **ee_point_down_plot_args)
        ee_point_up_line, = ax.plot([], [], **ee_point_up_plot_args)

        subplot_data['lines'] = (ee_point_down_line, ee_point_up_line,
                                 ee_down_line, ee_up_line,
                                 target_point_line, target_down_line,
                                 target_up_line, arm_line)

        if not show_tick_labels:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
        ax.xaxis.grid()
        ax.yaxis.grid()
        ax.set_aspect(1)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        return ax

    def arm_path_plot_update(self, key, t_data, data, draw_artists):
        subplot_data = self.subplot_data[key]['arm_path_plot']
        ax = self.subplot_data[key]['arm_path_plot']['ax']
        ax.set_title('Sim Time: %0.3fs' % t_data, fontsize=10)

        (ee_point_down_line, ee_point_up_line, ee_down_line, ee_up_line,
         target_point_line, target_down_line, target_up_line, arm_line) = \
            subplot_data['lines']

        ee_x = data[0]
        ee_y = data[1]
        target_x = data[2]
        target_y = data[3]
        up_down = data[4]
        arm_pos_x_data = data[5:9]
        arm_pos_y_data = data[9:13]

        # Last up-down state
        last_up_down_state = subplot_data['last_up_down_state']

        # Update EE (end effector) and target path plots
        ee_down_data = subplot_data['ee_down_data']
        ee_up_data = subplot_data['ee_up_data']
        target_down_data = subplot_data['target_down_data']
        target_up_data = subplot_data['target_up_data']

        # Need to manually erase data on animation loop restart
        if t_data < subplot_data['last_t']:
            del ee_down_data[0][:]
            del ee_down_data[1][:]
            del target_down_data[0][:]
            del target_down_data[1][:]
            del ee_up_data[0][:]
            del ee_up_data[1][:]
            del target_up_data[0][:]
            del target_up_data[1][:]
            subplot_data['last_up_down_state'] = 0

            # Update plots
            ee_point_down_line.set_data(None, None)
            ee_point_up_line.set_data(None, None)
            target_point_line.set_data(None, None)
            ee_down_line.set_data(ee_down_data[0], ee_down_data[1])
            target_down_line.set_data(target_down_data[0], target_down_data[1])
            ee_up_line.set_data(ee_up_data[0], ee_up_data[1])
            target_up_line.set_data(target_up_data[0], target_up_data[1])
        subplot_data['last_t'] = t_data

        if up_down > 0.5:
            # EE is in down state
            if last_up_down_state == 0:
                subplot_data['last_up_down_state'] = 1
                del ee_down_data[0][:]
                del ee_down_data[1][:]
                del target_down_data[0][:]
                del target_down_data[1][:]
                ee_point_up_line.set_data(None, None)

            ee_down_data[0].append(ee_x)
            ee_down_data[1].append(ee_y)
            target_down_data[0].append(target_x)
            target_down_data[1].append(target_y)

            ee_point_down_line.set_data(ee_x, ee_y)
            ee_down_line.set_data(ee_down_data[0], ee_down_data[1])
            target_down_line.set_data(target_down_data[0], target_down_data[1])
        else:
            # EE is in up state
            if last_up_down_state == 1:
                subplot_data['last_up_down_state'] = 0
                del ee_up_data[0][:]
                del ee_up_data[1][:]
                del target_up_data[0][:]
                del target_up_data[1][:]
                ee_point_down_line.set_data(None, None)

            ee_up_data[0].append(ee_x)
            ee_up_data[1].append(ee_y)
            target_up_data[0].append(target_x)
            target_up_data[1].append(target_y)

            ee_point_up_line.set_data(ee_x, ee_y)
            ee_up_line.set_data(ee_up_data[0], ee_up_data[1])
            target_up_line.set_data(target_up_data[0], target_up_data[1])

        target_point_line.set_data(target_x, target_y)

        arm_line.set_data(arm_pos_x_data, arm_pos_y_data)

        draw_artists.append(ee_point_down_line)
        draw_artists.append(ee_point_up_line)
        draw_artists.append(ee_down_line)
        draw_artists.append(ee_up_line)
        draw_artists.append(target_point_line)
        draw_artists.append(target_down_line)
        draw_artists.append(target_up_line)
