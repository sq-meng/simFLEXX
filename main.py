import matplotlib

import matplotlib.path as mpath
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import flexxtools as ft
import re
import time

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
mpl.rcParams.update({'axes.labelsize': 16,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16})



def onpick(event):
    if plt.get_current_fig_manager().toolbar.mode != '':
        return

    ind = event.ind[0]
    if event.mouseevent.button == 1:
        evt_artist = event.artist
        if not evt_artist.highlighted[ind]:
            evt_artist.text_scatter[ind].set_visible(True)
            evt_artist.dot_colors[ind] = [0, 1, 0]
            evt_artist.set_facecolor(evt_artist.dot_colors)
            evt_artist.highlighted[ind] = True
        else:
            evt_artist.text_scatter[ind].set_visible(False)
            evt_artist.dot_colors[ind] = [0.66, 0.66, 0.66]
            evt_artist.set_facecolor(evt_artist.dot_colors)
            evt_artist.highlighted[ind] = False
        plt.draw_all()


def onclick(event):
    if plt.get_current_fig_manager().toolbar.mode != '':
        return

    if event.button == 3:
        x, y = (event.xdata, event.ydata)
        core = event.canvas.figure.handles_hm['core']
        ssr = core.conf.horizontal_magnet['sample_stick_rotation']
        QR = core.convert(np.array((x, y, 0)), 'PR')
        QS = core.convert(QR, 'RS')
        try:
            kiS, kfS = core.find_kS(QR, event.canvas.figure.handles_hm['active_channel'])
            refS = core.convert(core.conf.alignment['hkl1'], 'RS')
            northR = core.conf.horizontal_magnet['north_along']
            northS = ft.rotZ(core.convert(northR, 'RS'), -ssr / 180 * np.pi)
            az_ki = ft.azimuthS(northS, -kiS)
            az_kf = ft.azimuthS(northS, kfS)
            az_Q = ft.azimuthS(northS, QS)
            A3 = ft.azimuthS(-kiS, refS) * 180 / np.pi
            A4 = ft.azimuthS(kiS, kfS) * 180 / np.pi

            event.canvas.figure.handles_hm['ki'][0].set_data([az_ki, az_ki], [0, np.linalg.norm(kiS)])
            event.canvas.figure.handles_hm['kf'][0].set_data([az_kf, az_kf], [0, np.linalg.norm(kfS)])
            event.canvas.figure.handles_hm['Q'][0].set_data([az_Q, az_Q], [0, np.linalg.norm(QS)])
            print('Q: %s, A3: %3.1f, A4: %3.1f' % (QR, A3, A4))
            plt.draw_all()
        except ValueError as err:
            print('Q: ', QR, err.args[0])
        finally:
            pass


def onkey(event):
    number_sets = len(event.canvas.figure.hm_scatter)
    active_channel = event.canvas.figure.handles_hm['active_channel']

    if event.key == ';':
        fig = event.canvas.figure
        fig.savefig('save.png', dpi=600)

    elif event.key == ' ':
        visible = event.canvas.figure.hm_scatter[active_channel - 1].get_visible()
        if visible:
            event.canvas.figure.hm_scatter[active_channel - 1].set_visible(False)
        else:
            event.canvas.figure.hm_scatter[active_channel - 1].set_visible(True)

    else:
        if re.match('[1-5]', event.key) is not None:
            num = int(event.key)
            event.canvas.figure.handles_hm['active_channel'] = num
            for i in range(0, number_sets):
                event.canvas.figure.hm_scatter[i].set_visible(False)
            event.canvas.figure.hm_scatter[num - 1].set_visible(True)
    plt.draw_all()


def plotCoverageArea(core, ch, ax):
    inst = core.conf.instrument
    scan = core.conf.scan
    plot = core.conf.plot
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, 6))
    verts = []
    codes = []
    Path = mpath.Path

    A3 = scan['A3start']
    A4start = inst['A4'] - (inst['channels'] - 1) / 2 * inst['choffset']
    A4 = A4start
    QP = ft.angleToPlot(core, inst['ki'], inst['kf'][ch - 1], A3, A4)
    codes.append(Path.MOVETO)
    verts.append((QP[0], QP[1]))

    for _ in range(1, inst['channels']):
        A4 = A4 + inst['choffset']
        QP = ft.angleToPlot(core, inst['ki'], inst['kf'][ch - 1], A3, A4)
        codes.append(Path.LINETO)
        verts.append((QP[0], QP[1]))

    for A3 in np.linspace(scan['A3start'], scan['A3end'], 100):
        QP = ft.angleToPlot(core, inst['ki'], inst['kf'][ch - 1], A3, A4)
        codes.append(Path.LINETO)
        verts.append((QP[0], QP[1]))

    for _ in np.arange(1, inst['channels']):
        A4 = A4 - inst['choffset']
        QP = ft.angleToPlot(core, inst['ki'], inst['kf'][ch - 1], A3, A4)
        codes.append(Path.LINETO)
        verts.append((QP[0], QP[1]))

    for A3 in np.linspace(scan['A3end'], scan['A3start'], 100):
        QP = ft.angleToPlot(core, inst['ki'], inst['kf'][ch - 1], A3, A4)
        codes.append(Path.LINETO)
        verts.append((QP[0], QP[1]))

    path = mpath.Path(verts, codes)
    ax.plot(list(zip(*verts))[0], list(zip(*verts))[1], linewidth=0.5, color='black', alpha=0.9, zorder=100)
    labeltext = 'Ef=' + str(inst['ef'][ch - 1]) + 'meV, ' + 'dE=' + '{:.1f}'.format(
        float(inst['ei']) - float(inst['ef'][ch - 1])) + 'meV'
    coverage_patch = mpatches.PathPatch(path, facecolor=colors[ch], alpha=0.12, zorder=10 - ch, label=labeltext)
    ax.add_patch(coverage_patch)

    x_axis_unitlen = np.linalg.norm(core.convert(np.array(plot['x']), 'RS'))
    y_axis_unitlen = np.linalg.norm(core.convert(np.array(plot['y']), 'RS'))
    aspect = y_axis_unitlen / x_axis_unitlen
    ax.axes.set_aspect(aspect, adjustable='box')

    xlabel = plot['xlabel'] + '(r.l.u.)'
    ylabel = plot['ylabel'] + '(r.l.u.)'
    ax.axes.set_xlabel(xlabel)
    ax.axes.set_ylabel(ylabel)

    ax.dataLim.update_from_data_xy(verts + [[0, 0]])
    ax.autoscale_view()

    return coverage_patch


def plotScatter(core, ch, ax):

    a3start = core.conf.scan['A3start']
    a3end = core.conf.scan['A3end']
    a4start = core.conf.instrument['A4'] - (core.conf.instrument['channels'] - 1) / 2 * core.conf.instrument['choffset']
    a4end = core.conf.instrument['A4'] + (core.conf.instrument['channels'] - 1) / 2 * core.conf.instrument['choffset']
    ssr = core.conf.horizontal_magnet['sample_stick_rotation']
    points = []
    colors = []
    for a3 in np.linspace(a3start, a3end, int(np.abs(a3end - a3start) / 3)):
        for a4 in np.linspace(a4start, a4end, 20):
            QP, t = ft.angleToPlot(core, core.conf.instrument['ki'], core.conf.instrument['kf'][ch - 1], a3, a4,
                                   hm=True, ssr=ssr)
            points.append((QP[0], QP[1]))
            if t > 0.99998:
                colors.append(np.array([0.2, 0.2, 1, 0.8]))
            else:
                colors.append(np.array([1, t, 0.2, 0.8]))
    points_array = np.array(points)
    hm_scatter = ax.scatter(points_array[:, 0], points_array[:, 1], facecolor=colors, edgecolor=[0, 0, 0, 0],
                            zorder=190, visible=False)
    return hm_scatter


def plotLattice(core, ax):
    #
    # Takes core and target axis, plots clickable low-index lattice points on screen
    # Returns handle for the artisting drawing points.
    #
    x, y = np.mgrid[-3:3:0.5, -3:3:0.5]
    xr = np.reshape(x, -1)
    yr = np.reshape(y, -1)
    plot_coords = np.zeros((xr.shape[0], 3))
    text_scatter = []
    point_highlighted = []
    colors = np.ones((xr.shape[0], 3)) * 0.66
    xR = np.array(core.conf.alignment['hkl1'])
    yR = np.array(core.conf.alignment['hkl2'])
    for i, (x, y) in enumerate(zip(xr, yr)):
        coord_R = xR * x + yR * y
        plot_coords[i] = core.convert(coord_R, 'RP')
        text = '{:.1f}'.format(coord_R[0]) + ', ' + \
               '{:.1f}'.format(coord_R[1]) + ', ' + \
               '{:.1f}'.format(coord_R[2])
        text_scatter.append(ax.text(plot_coords[i][0] + 0.04, plot_coords[i][1], text, visible=False, zorder=200,
                                    bbox=dict(facecolor='white', alpha=0.4, linewidth=0, boxstyle='round'),
                                    fontdict={'weight': 'bold'}))
        point_highlighted.append(False)
    if (plot_coords[:, 2] > 1e-6).any():
        print('Warning: Plotting vectors out of plane.')
    lattice_scatter = ax.scatter(plot_coords[:, 0], plot_coords[:, 1], s=60, facecolor=colors, picker=True, zorder=1)
    lattice_scatter.text_scatter = text_scatter
    lattice_scatter.highlighted = point_highlighted
    lattice_scatter.dot_colors = colors
    return lattice_scatter


def main():
    conf = ft.config('simFLEXX.conf')
    core = ft.UBmatrix(conf)

    fig_qspace = plt.figure()
    fig_qspace.canvas.mpl_connect('pick_event', onpick)
    fig_qspace.canvas.mpl_connect('button_press_event', onclick)
    fig_qspace.canvas.mpl_connect('key_press_event', onkey)
    ax_qspace = fig_qspace.add_subplot(1, 1, 1)

    fig_hm = plt.figure()
    ax_hm = fig_hm.add_subplot(1, 1, 1, projection='polar')
    fig_qspace.handles_hm = {'core': core, 'ax_hm': ax_hm, 'active_channel': 1}
    drawOcclusion(core, ax_hm, fig_qspace.handles_hm)

    fig_qspace.handles_hm['ki'] = ax_hm.plot([1, 1], [0, 1.5], linewidth=2, color=[1, 0.2, 0.2, 1])
    fig_qspace.handles_hm['kf'] = ax_hm.plot([-1, -1], [0, 1.5], linewidth=2, color=[0.2, 1, 0.2, 1])
    fig_qspace.handles_hm['Q'] = ax_hm.plot([0, 0], [0, 1.5], linewidth=2, color=[0.2, 0.2, 1, 1])

    patch = []
    for i in range(1, 6):
        patch.append(plotCoverageArea(core, i, ax_qspace))

    ax_qspace.autoscale(False)
    ax_qspace.legend(handles=patch)
    plotLattice(core, ax_qspace)
    fig_qspace.hm_scatter = []
    for i in range(1, 6):
        fig_qspace.hm_scatter.append(plotScatter(core, i, ax_qspace))

    ymin, ymax = ax_qspace.axes.get_ylim()
    xmin, xmax = ax_qspace.axes.get_xlim()
    ycenter, yspan = [(ymin + ymax) / 2, ymax - ymin]
    xcenter, xspan = [(xmin + xmax) / 2, xmax - xmin]
    disp_ratio = yspan * ax_qspace.axes.get_aspect() / xspan

    if disp_ratio > 1:
        ax_qspace.axes.set_xlim([xcenter - xspan * disp_ratio * 1.33 / 2, xcenter + xspan * disp_ratio * 1 / 2])
    elif disp_ratio < 1:
        ax_qspace.axes.set_ylim([ycenter - yspan / disp_ratio * 0.75 / 2, ycenter + yspan / disp_ratio * 1 / 2])

    plt.show()


def drawOcclusion(core, ax, handles):
    red_segments, yellow_segments = findShadows(core)

    for red in red_segments:
        theta = red[0] / 180 * np.pi
        width = (red[1] - red[0]) / 180 * np.pi
        ax.bar(theta, 3, width=width, facecolor='red', alpha=0.4, zorder=10, bottom=0.2)
    for yellow in yellow_segments:
        theta = yellow[0] / 180 * np.pi
        width = (yellow[1] - yellow[0]) / 180 * np.pi
        ax.bar(theta, 2.8, width=width, facecolor=[1, 0.8, 0.4], alpha=0.7, edgecolor=[0, 0, 0, 0.2], bottom=0.3)

    ax.set_theta_zero_location('N')
    ax.axes.set_rlim([0, 3.5])
    ax.set_yticklabels([])


def findShadows(core):
    theta_firstwindow = 0
    for theta_firstwindow in range(0, 360):
        if core.conf.transmission(theta_firstwindow / 180 * np.pi) == 1:
            break
    in_red = False
    in_yellow = False
    red_begin, red_end = (0, 0)
    yellow_begin, yellow_end = (0, 0)
    red_segments = []
    yellow_segments = []
    for theta in range(theta_firstwindow, theta_firstwindow + 361):
        t = core.conf.transmission(np.remainder(theta/180*np.pi, 2*np.pi))
        if t < 1e-6:
            if not in_red:
                in_red = True
                red_begin = theta
        if t > 1e-6:
            if in_red:
                in_red = False
                red_end = theta - 1
                red_segments.append((red_begin, red_end))
        if t < 1 - 1e-5:
            if not in_yellow:
                in_yellow = True
                yellow_begin = theta - 1
        if t > 1 - 1e-5:
            if in_yellow:
                in_yellow = False
                yellow_end = theta
                yellow_segments.append((yellow_begin, yellow_end))
    if core.conf.horizontal_magnet['magnet_ident'] == 'HM-1':
            yellow_segments = []
    print('HM yellow regions:', yellow_segments)
    print('HM red regions:', red_segments)
    return red_segments, yellow_segments

main()
