
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class PlotBase():
    def __init__(self, task_name="PlotBase", *args, **kwargs):
        self.plot_colors = ['Reds', 'Greens', 'BuPu', 'YlOrRd', 'Greys', 'OrRd',
                            'PuBu', 'PuBuGn', 'PuRd', 'RdPu', 'YlGn', 'YlOrBr', ]

        self.colors = list(mcolors.TABLEAU_COLORS.keys())

    @staticmethod
    def plot_violin(df_data, cols, save_path=None, ignore_quantile=0.001, show=False):
        def plot_assist(df_data, cols, save_path=None):
            scale = min(int((len(cols) * 4) ** 0.8), 50)
            plt.figure(figsize=(scale, int(scale * 1.2)))
            all_data = [df_data[coli].values for coli in cols]
            plt.violinplot(
                all_data,
                showmeans=False,
                showextrema=True,
                showmedians=False,
                # quantiles=[[0.01, 0.1, 0.5, 0.9, 0.99] for i in range(len(all_data))],
                quantiles=[[0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999] for i in range(len(all_data))],
            )
            plt.xticks(ticks=range(1, len(all_data) + 1), labels=cols, rotation=90)
            if save_path:
                plt.savefig(save_path, dpi=100)
            if show:
                plt.show()
            plt.close()

        # 1.raw
        plot_assist(df_data, cols, save_path)
        # 2.clip
        df_data = df_data.copy()
        if save_path:
            save_path = save_path.split('/')
            save_path[-1] = 'clip_' + save_path[-1]
            save_path = '/'.join(save_path)
        for coli in cols:
            lower_bound = np.quantile(df_data[coli].values, ignore_quantile)
            upper_bound = np.quantile(df_data[coli].values, 1 - ignore_quantile)
            df_data.loc[df_data[coli] <= lower_bound, coli] = lower_bound
            df_data.loc[df_data[coli] >= upper_bound, coli] = upper_bound
        plot_assist(df_data, cols, save_path)

    def plot_curve(self, itemname, df_data, target_col, change_col, save_path, title=None, plot_intervals=1,
                   axes_shape=(1, 1), normalize=True, color_col=['color']):
        def plot_assist(fig, pos, df_data, change_col):
            ax_fig = fig.add_subplot(*pos)
            df_data[target_col].plot(ax=ax_fig, color=self.colors[:df_data.shape[1]], lw=2)
            x = np.array(df_data.index)
            y = df_data[change_col]
            for i, color_coli in enumerate(color_col):
                indexi = (df_data[color_coli] != 0)
                xi = x[indexi]
                yi = y[indexi]
                ci = df_data[color_coli][indexi]
                # xi = x
                # yi = y
                # ci = df_data[color_coli]
                ax_fig.scatter(x=xi, y=yi, c=ci, s=ci * fig_size[0] / 1,
                               cmap=self.plot_colors[i + 2])

        fig_size = (30 * axes_shape[0], 15 * axes_shape[1])
        fig = plt.figure(figsize=fig_size)

        # 1.data preparation
        data = df_data[target_col]
        if normalize:
            data = (data - data.min()) / (data.max() - data.min())
        for color_coli in color_col:
            data[color_coli] = df_data[color_coli]
            data[color_coli].fillna(0, inplace=True)
            norm = plt.Normalize(data[color_coli].min(), data[color_coli].max())
            data.loc[:, color_coli] = norm(data[color_coli])
        data = data.loc[range(0, data.shape[0], plot_intervals), :]
        n_divs = axes_shape[0] * axes_shape[1]
        n_each_div = df_data.shape[0] / n_divs
        divided_df_data = [data.loc[int(i * (n_each_div)):int((i + 1) * n_each_div), :] for i in range(n_divs)]
        for i in range(n_divs):
            plot_assist(fig, axes_shape + (i + 1,), divided_df_data[i], change_col)

        # 3.save
        if title:
            plt.suptitle(title)
        plt.savefig(save_path, dpi=int(10000 / max(fig_size)))
        plt.close()
        logging.debug(f"PLOT CURVE: {itemname} saved to {save_path}")

    def plot_bar_curve(self, df_data, plot_cols, bar_cols, save_path, title=None, plot_intervals=1, axes_shape=(8, 1)):
        def plot_assist(fig, pos, df_data):
            N_TICK = 8

            ax_fig1 = fig.add_subplot(*pos)
            ax_fig1.set_ylim([min_plot_cols * 1.25 - max_plot_cols * 0.25, max_plot_cols])
            df_data[plot_cols].plot(ax=ax_fig1, color=self.colors[4:df_data.shape[1] + 4], lw=1)
            plt.legend(loc="upper left")

            # df_data[bar_cols].plot(ax=ax_fig1, kind='bar')
            ax_fig2 = ax_fig1.twinx()
            ax_fig2.set_ylim([0, max_bar_cols * 5])
            df_data[bar_cols].plot(ax=ax_fig2, kind='area')

            n_len = df_data.shape[0]

            labels = [df_data.index[min(int(k),len(df_data)-1)] for k in np.linspace(0,n_len,N_TICK).tolist()]
            plt.xticks(labels, [df_data.loc[l_i,'UpdateTime'] for l_i in labels])
            max_val = df_data["LastPrice"].max()
            min_val = df_data["LastPrice"].min()
            vib_rate = (max_val - min_val)/df_data["LastPrice"].values[0]
            plt.title(f"{round(vib_rate*100,4)}%", fontsize=15)

        fig_size = (20 * axes_shape[1], 4 * axes_shape[0])
        fig = plt.figure(figsize=fig_size)

        # 1.data preparation
        data = df_data[plot_cols + bar_cols +['UpdateTime']]
        max_plot_cols = df_data[plot_cols].values.max()
        min_plot_cols = df_data[plot_cols].values.min()
        max_bar_cols = np.quantile(abs(df_data[bar_cols].values),0.99)*2
        data = data.loc[range(0, data.shape[0], plot_intervals), :]
        n_divs = axes_shape[0] * axes_shape[1]
        n_each_div = df_data.shape[0] / n_divs
        divided_df_data = [data.loc[int(i * (n_each_div)):int((i + 1) * n_each_div), :] for i in range(n_divs)]
        divided_df_data.append(data)
        new_axes_shape = (axes_shape[0]+1,axes_shape[1])
        for i in range(len(divided_df_data)):
            plot_assist(fig, new_axes_shape + (i + 1,), divided_df_data[i])

        # 2.save
        if title:
            plt.suptitle(title, fontsize=50)
        save_path = os.path.join(save_path, title + ".jpg")
        plt.savefig(save_path, dpi=int(4000 / max(fig_size)))
        plt.close()
        logging.debug(f"PLOT BAR CURVE: {title} saved to {save_path}")

if __name__ == '__main__':
    PlotBase()