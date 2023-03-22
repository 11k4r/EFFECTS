import numpy as np
from matplotlib.lines import Line2D

class AggregationFunction:
    def __init__(self, agg, name=''):
        assert callable(agg)

        self.agg = agg
        self.name = name if name != "" else agg.__name__

    def explain(self):
        pass


class MaxAgg(AggregationFunction):

    def __init__(self):
        super().__init__(agg=np.max, name="max")

    def explain(self):
        def plot(classes, fig, sb, se, data, color_dict, title,
                 ylabel, title_size, label_size):
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=color_dict[cls], lw=2))
                for ts in data[cls]:
                    fig.plot(np.arange(sb, se), ts[sb:se], color=color_dict[cls], alpha=0.7)
                    fig.scatter(np.argmax(ts[sb:se]) + sb, np.max(ts[sb:se]), color=color_dict[cls])
            leg = fig.legend(custom_lines, classes, loc='upper left', ncol=2, fontsize=9)
            fig.add_artist(leg)
            legend2 = [Line2D([0], [0], marker='o', color='w', label='max',
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label='data', linestyle='solid', markersize=9)]
            fig.legend(handles=legend2, loc='upper right')

            fig.set_title(title, fontsize=title_size)
            fig.set_xlabel('Time', fontsize=label_size)
            fig.set_ylabel(ylabel, fontsize=label_size)
            fig.grid()

        return plot


class MinAgg(AggregationFunction):

    def __init__(self):
        super().__init__(agg=np.min, name="min")

    def explain(self):
        def plot(classes, fig, sb, se, data, color_dict, title,
                 ylabel, title_size, label_size):
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=color_dict[cls], lw=2))
                for ts in data[cls]:
                    fig.plot(np.arange(sb, se), ts[sb:se], color=color_dict[cls], alpha=0.7)
                    fig.scatter(np.argmin(ts[sb:se]) + sb, np.min(ts[sb:se]), color=color_dict[cls])
            leg = fig.legend(custom_lines, classes, loc='upper left', ncol=2, fontsize=9)
            fig.add_artist(leg)
            legend2 = [Line2D([0], [0], marker='o', color='w', label='min',
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label='data', linestyle='solid', markersize=9)]
            fig.legend(handles=legend2, loc='upper right')

            fig.set_title(title, fontsize=title_size)
            fig.set_xlabel('Time', fontsize=label_size)
            fig.set_ylabel(ylabel, fontsize=label_size)
            fig.grid()
        return plot


class MeanAgg(AggregationFunction):

    def __init__(self):
        super().__init__(agg=np.mean, name="mean")

    def explain(self):
        def plot(classes, fig, sb, se, data, color_dict, title,
                 ylabel, title_size, label_size):
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=color_dict[cls], lw=2))
                for ts in data[cls]:
                    fig.plot(np.arange(sb, se), ts[sb:se], color=color_dict[cls], alpha=0.5)
                    mean = np.mean(ts[sb:se])
                    fig.plot([sb, se], [mean, mean], '--', alpha=0.9, linewidth=2, color=color_dict[cls])
            leg = fig.legend(custom_lines, classes, loc='upper left', ncol=2, fontsize=9)
            fig.add_artist(leg)
            legend2 = [Line2D([0], [0], linestyle='dashed', color='black', label='mean',
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label='data', linestyle='solid', markersize=9)]
            fig.legend(handles=legend2, loc='upper right')

            fig.set_title(title, fontsize=title_size)
            fig.set_xlabel('Time', fontsize=label_size)
            fig.set_ylabel(ylabel, fontsize=label_size)
            fig.grid()

        return plot


class VarAgg(AggregationFunction):

    def __init__(self):
        super().__init__(agg=np.var, name="var")

    def explain(self):
        def plot(classes, fig, sb, se, data, color_dict, title,
                 ylabel, title_size, label_size):
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=color_dict[cls], lw=2))
                for ts in data[cls]:
                    fig.plot(np.arange(sb, se), ts[sb:se], color=color_dict[cls], alpha=0.5)
                    var = np.var(ts[sb:se])
                    fig.plot([sb, se], [var, var], '--', alpha=0.9, linewidth=2, color=color_dict[cls])
            leg = fig.legend(custom_lines, classes, loc='upper left', ncol=2, fontsize=9)
            fig.add_artist(leg)
            legend2 = [Line2D([0], [0], linestyle='dashed', color='black', label='var',
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label='data', linestyle='solid', markersize=9)]
            fig.legend(handles=legend2, loc='upper right')

            fig.set_title(title, fontsize=title_size)
            fig.set_xlabel('Time', fontsize=label_size)
            fig.set_ylabel(ylabel, fontsize=label_size)
            fig.grid()

        return plot


class PtpAgg(AggregationFunction):

    def __init__(self):
        super().__init__(agg=np.ptp, name="ptp")

    def explain(self):
        pass

def trend(X, axis=1):
    if len(X[0]) == 1:
        return np.zeros(len(X))
    x_axis = np.arange(len(np.array(X)[0]))
    return np.array([np.polyfit(x=x_axis, y=x, deg=1)[0] for x in X])

class TrendAgg(AggregationFunction):

    def __init__(self):
        super().__init__(agg=trend, name="trend")

    def explain(self):
        def plot(classes, fig, sb, se, data, color_dict, title,
                 ylabel, title_size, label_size):
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=color_dict[cls], lw=2))
                for ts in data[cls]:
                    fig.plot(np.arange(sb, se), ts[sb:se], color=color_dict[cls], alpha=0.5)
                    y0, y1 = np.polyval(np.polyfit(np.arange(sb, se), ts[sb:se], deg=1), [sb, se])
                    fig.plot([sb, se], [y0, y1], '--', alpha=1, linewidth=2, color=color_dict[cls])
            leg = fig.legend(custom_lines, classes, loc='upper left', ncol=2, fontsize=9)
            fig.add_artist(leg)
            legend2 = [Line2D([0], [0], linestyle='dashed', color='black', label='trend',
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label='data', linestyle='solid', markersize=9)]
            fig.legend(handles=legend2, loc='upper right')

            fig.set_title(title, fontsize=title_size)
            fig.set_xlabel('Time', fontsize=label_size)
            fig.set_ylabel(ylabel, fontsize=label_size)
            fig.grid()

        return plot

class StdAgg(AggregationFunction):

    def __init__(self):
        super().__init__(agg=np.std, name="std")

    def explain(self):
        def plot(classes, fig, sb, se, data, color_dict, title,
                 ylabel, title_size, label_size):
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=color_dict[cls], lw=2))
                for ts in data[cls]:
                    fig.plot(np.arange(sb, se), ts[sb:se], color=color_dict[cls], alpha=0.5)
                    std = np.std(ts[sb:se])
                    fig.plot([sb, se], [std, std], '--', alpha=0.9, linewidth=2, color=color_dict[cls])
            leg = fig.legend(custom_lines, classes, loc='upper left', ncol=2, fontsize=9)
            fig.add_artist(leg)
            legend2 = [Line2D([0], [0], linestyle='dashed', color='black', label='std',
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label='data', linestyle='solid', markersize=9)]
            fig.legend(handles=legend2, loc='upper right')

            fig.set_title(title, fontsize=title_size)
            fig.set_xlabel('Time', fontsize=label_size)
            fig.set_ylabel(ylabel, fontsize=label_size)
            fig.grid()

        return plot

class SumAgg(AggregationFunction):

    def __init__(self):
        super().__init__(agg=np.sum, name="sum")

    def explain(self):
        def plot(classes, fig, sb, se, data, color_dict, title,
                 ylabel, title_size, label_size):
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=color_dict[cls], lw=2))
                for ts in data[cls]:
                    fig.plot(np.arange(sb, se), ts[sb:se], color=color_dict[cls], alpha=0.5)
                    fig.plot(np.arange(sb, se), np.cumsum(ts[sb:se]), '--', alpha=0.9, linewidth=2, color=color_dict[cls])
            leg = fig.legend(custom_lines, classes, loc='upper left', ncol=2, fontsize=9)
            fig.add_artist(leg)
            legend2 = [Line2D([0], [0], linestyle='dashed', color='black', label='sum',
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label='data', linestyle='solid', markersize=9)]
            fig.legend(handles=legend2, loc='upper right')

            fig.set_title(title, fontsize=title_size)
            fig.set_xlabel('Time', fontsize=label_size)
            fig.set_ylabel(ylabel, fontsize=label_size)
            fig.grid()

        return plot


class MedianAgg(AggregationFunction):

    def __init__(self):
        super().__init__(agg=np.median, name="median")

    def explain(self):
        def plot(classes, fig, sb, se, data, color_dict, title,
                 ylabel, title_size, label_size):
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=color_dict[cls], lw=2))
                for ts in data[cls]:
                    fig.plot(np.arange(sb, se), ts[sb:se], color=color_dict[cls], alpha=0.5)
                    median = np.median(ts)
                    fig.plot([sb, se], [median, median], '--', alpha=0.9, linewidth=2, color=color_dict[cls])
            leg = fig.legend(custom_lines, classes, loc='upper left', ncol=2, fontsize=9)
            fig.add_artist(leg)
            legend2 = [Line2D([0], [0], linestyle='dashed', color='black', label='median',
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label='data', linestyle='solid', markersize=9)]
            fig.legend(handles=legend2, loc='upper right')

            fig.set_title(title, fontsize=title_size)
            fig.set_xlabel('Time', fontsize=label_size)
            fig.set_ylabel(ylabel, fontsize=label_size)
            fig.grid()

        return plot



def ipeak(x):
    return max(x.min(), x.max(), key=abs)

def peak(X, axis=1):
    if len(X[0]) == 1:
        return np.zeros(len(X))
    x_axis = np.arange(len(np.array(X)[0]))
    return np.array([ipeak(x) for x in X])


def argpeak(x):
    p = ipeak(x)
    return np.where(x == p)[0][0]

class PeakAgg(AggregationFunction):

    def __init__(self):
        super().__init__(agg=peak, name="peak")

    def explain(self):
        def plot(classes, fig, sb, se, data, color_dict, title,
                 ylabel, title_size, label_size):
            custom_lines = []
            for cls in classes:
                custom_lines.append(Line2D([0], [0], color=color_dict[cls], lw=2))
                for ts in data[cls]:
                    fig.plot(np.arange(sb, se), ts[sb:se], color=color_dict[cls], alpha=0.7)
                    fig.scatter(argpeak(ts[sb:se]) + sb, ipeak(ts[sb:se]), color=color_dict[cls])
            leg = fig.legend(custom_lines, classes, loc='upper left', ncol=2, fontsize=9)
            fig.add_artist(leg)
            legend2 = [Line2D([0], [0], marker='o', color='w', label='peak',
                              markerfacecolor='black', markersize=9),
                       Line2D([0], [0], color='black', lw=2, label='data', linestyle='solid', markersize=9)]
            fig.legend(handles=legend2, loc='upper right')

            fig.set_title(title, fontsize=title_size)
            fig.set_xlabel('Time', fontsize=label_size)
            fig.set_ylabel(ylabel, fontsize=label_size)
            fig.grid()
        return plot